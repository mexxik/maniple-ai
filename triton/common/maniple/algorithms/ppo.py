"""PPO: clipped surrogate objective with GAE.

The importance ratio uses the behaviour log-prob that travelled with each row (computed by <algo>_infer under the
policy that actually acted), so a few versions of policy lag are handled by the clipping. Rows older than
spec.ppo.max_policy_lag versions never reach this class (dropped by the buffer).
"""

from __future__ import annotations

import numpy as np
import torch

from ..nets import ActorCritic
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
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])

    def update(self, trajectories):
        batch = self._prepare_batch(trajectories)
        return self._optimize(batch)

    # ------------------------------------------------------------------ internals

    def _log_prob(self, distribution, action):
        """Log-prob of `action` under `distribution`, summed over action dims for continuous policies."""
        if self.model.actor.continuous:
            logp = distribution.log_prob(action)
        else:
            logp = distribution.log_prob(action.argmax(-1))  # one-hot -> index
        return logp.sum(-1) if logp.dim() > 1 else logp

    def _prepare_batch(self, trajectories):
        """Compute advantages/returns per trajectory and concatenate everything into flat tensors."""
        device = self.device
        obs_parts, action_parts, old_logp_parts, adv_parts, ret_parts = [], [], [], [], []

        with torch.no_grad():
            for trajectory in trajectories:
                obs = torch.as_tensor(trajectory["obs"], device=device)
                action = torch.as_tensor(trajectory["action"], device=device)

                values = self.model.value(obs).cpu().numpy()
                last_value = 0.0 if trajectory["done"][-1] else float(values[-1])
                advantages, returns = gae(
                    trajectory["reward"], values, trajectory["done"], last_value, self.cfg.gamma, self.cfg.lam
                )

                # behaviour log-prob: what the served policy assigned to the action. If the client did not send
                # one (NaN), fall back to recomputing it under the current network.
                sent = torch.as_tensor(trajectory["logp"], device=device)
                recomputed = self._log_prob(self.model.actor.distribution(obs), action)
                old_logp = torch.where(torch.isnan(sent), recomputed, sent)

                obs_parts.append(obs)
                action_parts.append(action)
                old_logp_parts.append(old_logp)
                adv_parts.append(torch.as_tensor(advantages, device=device))
                ret_parts.append(torch.as_tensor(returns, device=device))

        advantages = torch.cat(adv_parts)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return {
            "obs": torch.cat(obs_parts),
            "action": torch.cat(action_parts),
            "old_logp": torch.cat(old_logp_parts),
            "advantage": advantages,
            "return": torch.cat(ret_parts),
        }

    def _optimize(self, batch):
        """Several epochs of minibatch clipped-PPO updates. Returns averaged stats."""
        cfg = self.cfg
        self.model.train()

        n = batch["obs"].shape[0]
        stats = {"samples": int(n), "loss_pi": 0.0, "loss_v": 0.0, "kl": 0.0}
        steps = 0

        for _ in range(cfg.epochs):
            permutation = torch.randperm(n, device=self.device)

            for start in range(0, n, cfg.minibatch):
                idx = permutation[start : start + cfg.minibatch]
                obs = batch["obs"][idx]
                action = batch["action"][idx]
                old_logp = batch["old_logp"][idx]
                advantage = batch["advantage"][idx]
                target_return = batch["return"][idx]

                distribution = self.model.actor.distribution(obs)
                logp = self._log_prob(distribution, action)
                ratio = torch.exp(logp - old_logp)

                clipped_ratio = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip)
                loss_pi = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()
                loss_v = ((self.model.value(obs) - target_return) ** 2).mean()
                entropy = distribution.entropy().mean()
                loss = loss_pi + cfg.value_coef * loss_v - cfg.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                stats["loss_pi"] += float(loss_pi)
                stats["loss_v"] += float(loss_v)
                stats["kl"] += float((old_logp - logp).mean())
                steps += 1

        for key in ("loss_pi", "loss_v", "kl"):
            stats[key] /= max(steps, 1)
        return stats
