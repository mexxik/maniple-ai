"""PPO: clipped surrogate objective with GAE.

The importance ratio uses the behaviour log-prob that travelled with each row (computed by <algo>_infer under the
policy that actually acted), so a few versions of policy lag are handled by the clipping. Rows older than
spec.ppo.max_policy_lag versions never reach this class (dropped by the buffer).

Observations are dicts (one tensor per spec input); actions are the flat row over all groups, and the
log-prob / entropy are summed over groups by ActionDistribution.
"""

from __future__ import annotations

import numpy as np
import torch

from ..nets import ActorCritic, migrate_v1_state_dict
from ..spec import AgentSpec
from .base import Algorithm


def gae(rewards, values, dones, last_value, gamma, lam):
    """Generalised advantage estimation over one trajectory. Returns (advantages, returns)."""
    steps = len(rewards)
    advantages = np.zeros(steps, dtype=np.float32)
    running = 0.0

    for t in reversed(range(steps)):
        next_value = last_value if t == steps - 1 else values[t + 1]
        nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        running = delta + gamma * lam * nonterminal * running
        advantages[t] = running

    return advantages, advantages + values


class PPO(Algorithm):
    def __init__(self, spec: AgentSpec, device: str):
        super().__init__(spec, device)
        self.cfg = spec.ppo
        self.rollout_size = self.cfg.rollout
        self.model = ActorCritic(spec).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)

    # ------------------------------------------------------------------ Algorithm interface

    def actor(self):
        return self.model.actor

    def state_dict(self):
        return {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state):
        # strict=False: checkpoints from before a net option existed (e.g. the obs normalizer) still load;
        # checkpoints from before encoders/heads existed are renamed first
        self.model.load_state_dict(migrate_v1_state_dict(state["model"]), strict=False)
        try:
            self.optimizer.load_state_dict(self._migrate_optimizer(state["model"], state["optimizer"]))
        except (ValueError, KeyError, RuntimeError) as e:
            print(f"[maniple] optimizer state not restored ({type(e).__name__}: {e}); Adam restarts")
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self._check_optimizer_shapes()

    def _migrate_optimizer(self, model_state: dict, opt_state: dict) -> dict:
        """Adam's moments are stored by parameter POSITION. A v1 checkpoint ordered them by the old module
        layout (actor.log_std first), so re-key them by parameter NAME onto the current layout."""
        if not any(k.startswith("actor.net.") for k in model_state):
            return opt_state
        # parameters in the old parameters() order = the old state_dict order minus its buffers
        old_names = [k for k in model_state if not k.startswith("actor.normalizer.")]
        if len(old_names) != len(opt_state["state"]):
            raise ValueError(
                f"{len(old_names)} old parameters vs {len(opt_state['state'])} optimizer entries"
            )
        renamed = list(migrate_v1_state_dict({k: i for i, k in enumerate(old_names)}).items())
        by_new_name = {new: old_index for new, old_index in renamed}
        state, params = {}, []
        for new_index, (name, _) in enumerate(self.model.named_parameters()):
            if name in by_new_name and by_new_name[name] in opt_state["state"]:
                state[new_index] = opt_state["state"][by_new_name[name]]
            params.append(new_index)
        groups = [dict(opt_state["param_groups"][0], params=params)]
        return {"state": state, "param_groups": groups}

    def _check_optimizer_shapes(self) -> None:
        """A moment of the wrong shape would crash the first update; restart Adam instead."""
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                st = self.optimizer.state.get(param, {})
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in st and st[key].shape != param.shape:
                        print(
                            f"[maniple] optimizer moment {key} {tuple(st[key].shape)} vs param {tuple(param.shape)}; Adam restarts"
                        )
                        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
                        return

    def update(self, trajectories):
        # optimise under the normaliser the actions were sampled with (the sent log-probs assume it),
        # then let the running statistics absorb this batch for the next version
        batch = self._prepare_batch(trajectories)
        stats = self._optimize(batch)
        self.model.actor.update_normalizers(batch["inputs"])
        return stats

    # ------------------------------------------------------------------ internals

    def _to_device(self, inputs: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        return {name: torch.as_tensor(array, device=self.device) for name, array in inputs.items()}

    def _prepare_batch(self, trajectories):
        """Compute advantages/returns per trajectory and concatenate everything into flat tensors."""
        device = self.device
        input_parts: dict[str, list[torch.Tensor]] = {i.name: [] for i in self.spec.inputs}
        action_parts, old_logp_parts, adv_parts, ret_parts = [], [], [], []

        with torch.no_grad():
            for trajectory in trajectories:
                inputs = self._to_device(trajectory["inputs"])
                action = torch.as_tensor(trajectory["action"], device=device)

                values = self.model.value(inputs).cpu().numpy()
                last_value = 0.0 if trajectory["done"][-1] else float(values[-1])
                advantages, returns = gae(
                    trajectory["reward"], values, trajectory["done"], last_value, self.cfg.gamma, self.cfg.lam
                )

                # behaviour log-prob: what the served policy assigned to the action. If the client did not send
                # one (NaN), fall back to recomputing it under the current network.
                sent = torch.as_tensor(trajectory["logp"], device=device)
                recomputed = self.model.actor.distribution(inputs).log_prob(action)
                old_logp = torch.where(torch.isnan(sent), recomputed, sent)

                for name, tensor in inputs.items():
                    input_parts[name].append(tensor)
                action_parts.append(action)
                old_logp_parts.append(old_logp)
                adv_parts.append(torch.as_tensor(advantages, device=device))
                ret_parts.append(torch.as_tensor(returns, device=device))

        advantages = torch.cat(adv_parts)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return {
            "inputs": {name: torch.cat(parts) for name, parts in input_parts.items()},
            "action": torch.cat(action_parts),
            "old_logp": torch.cat(old_logp_parts),
            "advantage": advantages,
            "return": torch.cat(ret_parts),
        }

    def _optimize(self, batch):
        """Several epochs of minibatch clipped-PPO updates. Returns averaged stats."""
        cfg = self.cfg
        self.model.train()

        n = batch["action"].shape[0]
        stats = {"samples": int(n), "loss_pi": 0.0, "loss_v": 0.0, "kl": 0.0}
        steps = 0

        for _ in range(cfg.epochs):
            permutation = torch.randperm(n, device=self.device)

            for start in range(0, n, cfg.minibatch):
                idx = permutation[start : start + cfg.minibatch]
                inputs = {name: tensor[idx] for name, tensor in batch["inputs"].items()}
                action = batch["action"][idx]
                old_logp = batch["old_logp"][idx]
                advantage = batch["advantage"][idx]
                target_return = batch["return"][idx]

                distribution = self.model.actor.distribution(inputs)
                logp = distribution.log_prob(action)
                ratio = torch.exp(logp - old_logp)

                clipped_ratio = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip)
                loss_pi = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()
                loss_v = ((self.model.value(inputs) - target_return) ** 2).mean()
                entropy = distribution.entropy().mean()
                loss = loss_pi + cfg.value_coef * loss_v - cfg.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                stats["loss_pi"] += loss_pi.item()
                stats["loss_v"] += loss_v.item()
                stats["kl"] += (old_logp - logp).mean().item()
                steps += 1

        for key in ("loss_pi", "loss_v", "kl"):
            stats[key] /= max(steps, 1)
        log_stds = [head.log_std for head in self.model.actor.heads.values() if hasattr(head, "log_std")]
        if log_stds:
            stats["std"] = torch.cat(log_stds).detach().exp().mean().item()
        return stats
