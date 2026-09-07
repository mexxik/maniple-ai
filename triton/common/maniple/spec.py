"""AgentSpec: the ONE contract shared by game, trainer and exported policy.

JSON example (this is what the client sends with command=register):
{
  "obs":    {"dim": 24},
  "action": {"type": "continuous", "dim": 6, "low": -1.0, "high": 1.0},
  "hidden": [64, 64],
  "ppo":    {"gamma": 0.99, "lam": 0.95, "clip": 0.2, "lr": 3e-4, "epochs": 4, "minibatch": 256, "rollout": 2048}
}
Discrete actions: {"type": "discrete", "n": 5}. The exported policy always outputs FP32 [batch, action_dim]
(continuous: the mean; discrete: logits) so the client stays generic.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class ActionSpace:
    type: str = "continuous"  # continuous | discrete
    dim: int = 0  # continuous
    n: int = 0  # discrete
    low: float = -1.0
    high: float = 1.0

    @property
    def out_dim(self) -> int:
        return self.dim if self.type == "continuous" else self.n


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    epochs: int = 4
    minibatch: int = 256
    rollout: int = 2048  # transitions per update
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    max_policy_lag: int = 4  # drop rows produced by policies older than this many versions


@dataclass
class AgentSpec:
    obs_dim: int
    action: ActionSpace
    hidden: list = field(default_factory=lambda: [64, 64])
    ppo: PPOConfig = field(default_factory=PPOConfig)

    @staticmethod
    def from_json(text: str) -> AgentSpec:
        d = json.loads(text)
        return AgentSpec(
            obs_dim=int(d["obs"]["dim"]),
            action=ActionSpace(**d.get("action", {})),
            hidden=list(d.get("hidden", [64, 64])),
            ppo=PPOConfig(**d.get("ppo", {})),
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "obs": {"dim": self.obs_dim},
                "action": asdict(self.action),
                "hidden": self.hidden,
                "ppo": asdict(self.ppo),
            }
        )
