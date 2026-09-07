"""Networks built from an AgentSpec. Actor alone is what gets exported (obs -> action)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .spec import AgentSpec


def mlp(inp, hidden, out):
    layers, d = [], inp
    for h in hidden:
        layers += [nn.Linear(d, h), nn.Tanh()]
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """obs -> action mean (continuous) or logits (discrete).
    Exported as ONNX: input 'obs'; outputs 'action' (mean/logits) and, for continuous, 'log_std'."""

    def __init__(self, spec: AgentSpec):
        super().__init__()
        self.net = mlp(spec.obs_dim, spec.hidden, spec.action.out_dim)
        self.continuous = spec.action.type == "continuous"
        if self.continuous:
            self.log_std = nn.Parameter(torch.zeros(spec.action.out_dim))

    def forward(self, obs):
        """Export signature. Continuous: (mean, log_std broadcast to [batch, dim]); discrete: (logits,)."""
        out = self.net(obs)
        if self.continuous:
            return out, self.log_std.unsqueeze(0).repeat(obs.shape[0], 1)
        return out

    def distribution(self, obs):
        out = self.net(obs)
        if self.continuous:
            return torch.distributions.Normal(out, self.log_std.exp())
        return torch.distributions.Categorical(logits=out)


class ActorCritic(nn.Module):
    def __init__(self, spec: AgentSpec):
        super().__init__()
        self.actor = Actor(spec)
        self.critic = mlp(spec.obs_dim, spec.hidden, 1)

    def value(self, obs):
        return self.critic(obs).squeeze(-1)
