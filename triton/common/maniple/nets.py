"""Networks built from an AgentSpec. The Actor alone is what gets exported (obs -> action)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .spec import AgentSpec

_ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "gelu": nn.GELU}


def mlp(
    inp: int, hidden: list[int], out: int, activation: str = "tanh", layernorm: bool = False
) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = inp
    for h in hidden:
        layers.append(nn.Linear(width, h))
        if layernorm:
            layers.append(nn.LayerNorm(h))
        layers.append(_ACTIVATIONS[activation]())
        width = h
    layers.append(nn.Linear(width, out))
    return nn.Sequential(*layers)


class ObsNormalizer(nn.Module):
    """Running mean/std of observations. Updated by the trainer, frozen into the export as plain buffers."""

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


class Actor(nn.Module):
    """obs -> action mean (continuous) or logits (discrete).
    Exported as ONNX: input 'obs'; outputs 'action' (mean/logits) and, for continuous, 'log_std'."""

    def __init__(self, spec: AgentSpec):
        super().__init__()
        net = spec.net
        self.normalizer = ObsNormalizer(spec.obs_dim, net.normalize_obs)
        self.net = mlp(spec.obs_dim, spec.hidden, spec.action.out_dim, net.activation, net.layernorm)
        self.continuous = spec.action.type == "continuous"
        if self.continuous:
            self.log_std = nn.Parameter(torch.zeros(spec.action.out_dim))

    def forward(self, obs):
        """Export signature. Continuous: (mean, log_std broadcast to [batch, dim]); discrete: logits."""
        out = self.net(self.normalizer(obs))
        if self.continuous:
            return out, self.log_std.unsqueeze(0).repeat(obs.shape[0], 1)
        return out

    def distribution(self, obs):
        out = self.net(self.normalizer(obs))
        if self.continuous:
            return torch.distributions.Normal(out, self.log_std.exp())
        return torch.distributions.Categorical(logits=out)


class ActorCritic(nn.Module):
    """Actor plus a value head. separate_critic=True gives the critic its own MLP (default);
    False attaches a linear value head to the actor's last hidden layer."""

    def __init__(self, spec: AgentSpec):
        super().__init__()
        self.actor = Actor(spec)
        net = spec.net
        self.separate_critic = net.separate_critic
        if self.separate_critic:
            self.critic = mlp(spec.obs_dim, spec.hidden, 1, net.activation, net.layernorm)
        else:
            self.critic = nn.Linear(spec.hidden[-1], 1)

    def value(self, obs):
        x = self.actor.normalizer(obs)
        if self.separate_critic:
            return self.critic(x).squeeze(-1)
        torso = self.actor.net[:-1]  # everything but the output layer
        return self.critic(torso(x)).squeeze(-1)
