"""AgentSpec: the ONE contract shared by game, trainer and exported policy.

JSON example (this is what the client sends with command=register):
{
  "obs":    {"dim": 24},
  "action": {"type": "continuous", "dim": 6, "low": -1.0, "high": 1.0},
  "net":    {"preset": "medium", "activation": "relu", "layernorm": true, "normalize_obs": true},
  "ppo":    {"gamma": 0.99, "lam": 0.95, "clip": 0.2, "lr": 3e-4, "epochs": 4, "minibatch": 256, "rollout": 2048}
}

Network size, three ways (net.preset):
  "auto"                     width derived from the observation size, two layers   (default)
  "small" | "medium" | "large"   fixed presets, see PRESETS
  "custom"                   use net.hidden as given, e.g. [512, 512, 256]
Giving net.hidden without a preset implies "custom". The resolved layers are reported back in status().

Discrete actions: {"type": "discrete", "n": 5}. The exported policy always outputs FP32 [batch, action_dim]
(continuous: the mean; discrete: logits) so the client stays generic.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

PRESETS = {
    "small": [64, 64],
    "medium": [256, 256],
    "large": [512, 512, 256],
}

ACTIVATIONS = ("tanh", "relu", "elu", "gelu")


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
class NetConfig:
    preset: str = "auto"  # auto | small | medium | large | custom
    hidden: list[int] = field(default_factory=list)  # used when preset == custom
    activation: str = "tanh"  # tanh | relu | elu | gelu
    layernorm: bool = False  # LayerNorm after every hidden layer
    separate_critic: bool = True  # False = critic head on the actor torso
    normalize_obs: bool = True  # running mean/std of observations, baked into the export

    def resolve_hidden(self, obs_dim: int) -> list[int]:
        """The hidden layer sizes this config means for a given observation size."""
        if self.preset == "custom":
            if not self.hidden:
                raise ValueError("net.preset=custom needs net.hidden")
            return list(self.hidden)
        if self.preset in PRESETS:
            return list(PRESETS[self.preset])
        if self.preset == "auto":
            width = int(min(512, max(64, round(4 * obs_dim / 32) * 32)))
            return [width, width]
        raise ValueError(f"unknown net.preset '{self.preset}' (auto, custom, {', '.join(PRESETS)})")

    def validate(self) -> None:
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"unknown net.activation '{self.activation}' ({', '.join(ACTIVATIONS)})")


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
    net: NetConfig = field(default_factory=NetConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    @property
    def hidden(self) -> list[int]:
        return self.net.resolve_hidden(self.obs_dim)

    @staticmethod
    def from_json(text: str) -> AgentSpec:
        data = json.loads(text)

        net_data = dict(data.get("net", {}))
        if "hidden" in data and "net" not in data:  # older clients: top-level hidden list
            net_data = {"preset": "custom", "hidden": list(data["hidden"])}
        if net_data.get("hidden") and "preset" not in net_data:
            net_data["preset"] = "custom"

        spec = AgentSpec(
            obs_dim=int(data["obs"]["dim"]),
            action=ActionSpace(**data.get("action", {})),
            net=NetConfig(**net_data),
            ppo=PPOConfig(**data.get("ppo", {})),
        )
        spec.net.validate()
        spec.net.resolve_hidden(spec.obs_dim)  # raises early on a bad preset
        return spec

    def to_json(self) -> str:
        return json.dumps(
            {
                "obs": {"dim": self.obs_dim},
                "action": asdict(self.action),
                "net": asdict(self.net),
                "ppo": asdict(self.ppo),
            }
        )

    def describe(self) -> dict:
        """What the trainer actually built from this spec; reported in status()."""
        return {
            "obs_dim": self.obs_dim,
            "action": asdict(self.action),
            "hidden": self.hidden,
            "activation": self.net.activation,
            "layernorm": self.net.layernorm,
            "separate_critic": self.net.separate_critic,
            "normalize_obs": self.net.normalize_obs,
        }
