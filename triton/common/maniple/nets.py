"""Networks built from an AgentSpec. The Actor alone is what gets exported (inputs -> flat action row).

inputs (dict name -> tensor)
  |-- one encoder per input (VectorEncoder | ImageEncoder)  -> features, concatenated in spec order
  |-- torso (net.hidden)
  |-- one head per action group (ContinuousHead | DiscreteHead) -> flat action row, in spec order
critic: MLP on the same features (net.separate_critic) or a linear head on the torso
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .spec import ActionGroup, AgentSpec, InputSpec, NetConfig

_ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "gelu": nn.GELU}


def hidden_stack(inp: int, hidden: list[int], activation: str, layernorm: bool) -> tuple[nn.Sequential, int]:
    """Linear (+ LayerNorm) + activation per hidden width. Returns (module, output width)."""
    layers: list[nn.Module] = []
    width = inp
    for h in hidden:
        layers.append(nn.Linear(width, h))
        if layernorm:
            layers.append(nn.LayerNorm(h))
        layers.append(_ACTIVATIONS[activation]())
        width = h
    return nn.Sequential(*layers), width


def mlp(
    inp: int, hidden: list[int], out: int, activation: str = "tanh", layernorm: bool = False
) -> nn.Sequential:
    stack, width = hidden_stack(inp, hidden, activation, layernorm)
    return nn.Sequential(*stack, nn.Linear(width, out))


class ObsNormalizer(nn.Module):
    """Running mean/std of a vector input. Updated by the trainer, frozen into the export as plain buffers."""

    def __init__(self, dim: int, enabled: bool, clip: float = 10.0):
        super().__init__()
        self.enabled = enabled
        self.clip = clip
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, obs: torch.Tensor) -> None:
        if not self.enabled:
            return
        obs = obs.float()
        batch_mean = obs.mean(0)
        batch_var = obs.var(0, unbiased=False)
        batch_count = obs.shape[0]

        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total

        self.mean.copy_(new_mean)
        self.var.copy_(m2 / total)
        self.count.copy_(total)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return obs
        normalized = (obs - self.mean) / torch.sqrt(self.var + 1e-8)
        return torch.clamp(normalized, -self.clip, self.clip)


# ---------------------------------------------------------------- encoders


class VectorEncoder(nn.Module):
    """'mlp': normalise, then optional hidden layers. With no hidden layers the vector passes straight through."""

    def __init__(self, spec: InputSpec, net: NetConfig):
        super().__init__()
        self.normalizer = ObsNormalizer(spec.size, net.normalize_obs)
        self.net, self.out_dim = hidden_stack(spec.size, spec.hidden, net.activation, net.layernorm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.normalizer(x.float().flatten(1)))


class ImageEncoder(nn.Module):
    """'cnn': the Nature-DQN convolutions on a channels-last uint8 frame, then fully connected layers."""

    def __init__(self, spec: InputSpec, net: NetConfig):
        super().__init__()
        height, width, channels = (int(d) for d in spec.shape)
        act = _ACTIVATIONS[net.activation]
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, 8, stride=4),
            act(),
            nn.Conv2d(32, 64, 4, stride=2),
            act(),
            nn.Conv2d(64, 64, 3, stride=1),
            act(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, channels, height, width)).shape[1]
        self.fc, self.out_dim = hidden_stack(flat, spec.hidden, net.activation, net.layernorm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2).float() / 255.0  # NHWC uint8 -> NCHW [0, 1]
        return self.fc(self.conv(x))


def build_encoder(spec: InputSpec, net: NetConfig) -> nn.Module:
    if spec.encoder == "cnn":
        return ImageEncoder(spec, net)
    return VectorEncoder(spec, net)


# ---------------------------------------------------------------- heads


class ContinuousHead(nn.Module):
    """Gaussian with a state-independent log std. Row contribution: the mean (export) or a sample."""

    def __init__(self, width: int, group: ActionGroup, log_std_init: float):
        super().__init__()
        self.mean = nn.Linear(width, group.dim)
        self.log_std = nn.Parameter(torch.full((group.dim,), float(log_std_init)))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.mean(h)

    def distribution(self, h: torch.Tensor) -> torch.distributions.Distribution:
        return torch.distributions.Normal(self.mean(h), self.log_std.exp())


class DiscreteHead(nn.Module):
    """Categorical over n choices. Row contribution: the logits (export) or a one-hot."""

    def __init__(self, width: int, group: ActionGroup):
        super().__init__()
        self.logits = nn.Linear(width, group.n)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.logits(h)

    def distribution(self, h: torch.Tensor) -> torch.distributions.Distribution:
        return torch.distributions.Categorical(logits=self.logits(h))


class ActionDistribution:
    """The joint distribution over all action groups, working on the flat action row."""

    def __init__(self, spec: AgentSpec, dists: dict[str, torch.distributions.Distribution]):
        self.spec = spec
        self.dists = dists
        self.slices = spec.action_slices

    def _row(self, parts: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([parts[g.name] for g in self.spec.actions], dim=-1)

    def _one_hot(self, group: ActionGroup, index: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(index, group.n).float()

    def sample(self) -> torch.Tensor:
        parts = {}
        for g in self.spec.actions:
            s = self.dists[g.name].sample()
            parts[g.name] = s if g.continuous else self._one_hot(g, s)
        return self._row(parts)

    def mode(self) -> torch.Tensor:
        parts = {}
        for g in self.spec.actions:
            d = self.dists[g.name]
            parts[g.name] = d.mean if g.continuous else self._one_hot(g, d.probs.argmax(-1))
        return self._row(parts)

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """Summed over groups (and over the values of a continuous group). [N]."""
        total = None
        for g in self.spec.actions:
            part = action[:, self.slices[g.name]]
            lp = (
                self.dists[g.name].log_prob(part).sum(-1)
                if g.continuous
                else self.dists[g.name].log_prob(part.argmax(-1))
            )
            total = lp if total is None else total + lp
        return total

    def entropy(self) -> torch.Tensor:
        total = None
        for g in self.spec.actions:
            e = self.dists[g.name].entropy()
            e = e.sum(-1) if g.continuous else e
            total = e if total is None else total + e
        return total

    def indices(self, action: torch.Tensor) -> torch.Tensor:
        """Chosen index per discrete group, [N, G] (G may be 0)."""
        cols = [action[:, self.slices[g.name]].argmax(-1) for g in self.spec.discrete_groups]
        if not cols:
            return torch.zeros((action.shape[0], 0), dtype=torch.int64, device=action.device)
        return torch.stack(cols, dim=-1)


# ---------------------------------------------------------------- actor / critic


class Actor(nn.Module):
    """inputs -> flat action row (means / logits).
    Exported as ONNX: one input per spec input (spec order); outputs 'action' [N, action_dim] and, when any
    group is continuous, 'log_std' [N, continuous_dim] (the continuous groups' log stds, spec order)."""

    def __init__(self, spec: AgentSpec):
        super().__init__()
        self.spec = spec
        net = spec.net
        self.encoders = nn.ModuleDict({i.name: build_encoder(i, net) for i in spec.inputs})
        self.torso, width = hidden_stack(spec.feature_dim, spec.hidden, net.activation, net.layernorm)
        self.torso_dim = width
        self.heads = nn.ModuleDict(
            {
                g.name: ContinuousHead(width, g, net.log_std_init) if g.continuous else DiscreteHead(width, g)
                for g in spec.actions
            }
        )
        self.continuous = spec.continuous_dim > 0  # any continuous group -> the export has 'log_std'

    def features(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([self.encoders[i.name](inputs[i.name]) for i in self.spec.inputs], dim=-1)

    def hidden(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.torso(self.features(inputs))

    def forward(self, *tensors: torch.Tensor):
        """Export signature: positional inputs in spec order -> action row (, log_std)."""
        inputs = {i.name: t for i, t in zip(self.spec.inputs, tensors, strict=True)}
        h = self.hidden(inputs)
        row = torch.cat([self.heads[g.name](h) for g in self.spec.actions], dim=-1)
        if not self.continuous:
            return row
        n = row.shape[0]
        log_std = torch.cat(
            [
                self.heads[g.name].log_std.unsqueeze(0).expand(n, -1)
                for g in self.spec.actions
                if g.continuous
            ],
            dim=-1,
        )
        return row, log_std

    def distribution(self, inputs: dict[str, torch.Tensor]) -> ActionDistribution:
        h = self.hidden(inputs)
        return ActionDistribution(
            self.spec, {g.name: self.heads[g.name].distribution(h) for g in self.spec.actions}
        )

    def update_normalizers(self, inputs: dict[str, torch.Tensor]) -> None:
        for name, encoder in self.encoders.items():
            if isinstance(encoder, VectorEncoder) and name in inputs:
                encoder.normalizer.update(inputs[name].flatten(1))


class ActorCritic(nn.Module):
    """Actor plus a value head. separate_critic=True gives the critic its own MLP on the encoder features
    (default); False attaches a linear value head to the actor's torso."""

    def __init__(self, spec: AgentSpec):
        super().__init__()
        self.actor = Actor(spec)
        net = spec.net
        self.separate_critic = net.separate_critic
        if self.separate_critic:
            self.critic = mlp(spec.feature_dim, spec.hidden, 1, net.activation, net.layernorm)
        else:
            self.critic = nn.Linear(self.actor.torso_dim, 1)

    def value(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.separate_critic:
            return self.critic(self.actor.features(inputs)).squeeze(-1)
        return self.critic(self.actor.hidden(inputs)).squeeze(-1)


def migrate_v1_state_dict(state: dict) -> dict:
    """Rename the keys of a checkpoint written before encoders/heads existed (one 'obs' vector, one group
    'action'): actor.net.* was normaliser -> hidden stack -> output Linear in one Sequential."""
    if not any(k.startswith("actor.net.") for k in state):
        return state
    last = max(int(k.split(".")[2]) for k in state if k.startswith("actor.net."))
    out = {}
    for key, value in state.items():
        parts = key.split(".")
        if key.startswith("actor.normalizer."):
            out["actor.encoders.obs.normalizer." + ".".join(parts[2:])] = value
        elif key == "actor.log_std":
            out["actor.heads.action.log_std"] = value
        elif key.startswith("actor.net."):
            index = int(parts[2])
            if index == last:
                out["actor.heads.action.mean." + ".".join(parts[3:])] = value
                out["actor.heads.action.logits." + ".".join(parts[3:])] = (
                    value  # discrete twin; one is dropped
                )
            else:
                out[f"actor.torso.{index}." + ".".join(parts[3:])] = value
        else:
            out[key] = value
    return out
