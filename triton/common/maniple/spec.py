"""AgentSpec: the ONE contract shared by game, trainer and exported policy.

A policy takes several NAMED INPUTS and returns several NAMED ACTION GROUPS. JSON example (this is what the
client sends with command=register):
{
  "inputs": {
    "obs":   {"shape": [32],         "dtype": "fp32",  "encoder": "mlp"},
    "frame": {"shape": [84, 84, 4],  "dtype": "uint8", "encoder": "cnn"}
  },
  "actions": {
    "move": {"type": "continuous", "dim": 4, "low": -1.0, "high": 1.0},
    "fire": {"type": "discrete",   "n": 2}
  },
  "net":    {"preset": "medium", "activation": "relu", "layernorm": true, "normalize_obs": true},
  "ppo":    {"gamma": 0.99, "lam": 0.95, "clip": 0.2, "lr": 3e-4, "epochs": 4, "minibatch": 256, "rollout": 2048},
  "versioning": {"score_window": 20, "export_on_improvement": true, "keep_exported": 3, "trt_on_promote": true,
                 "score": "return"}
}

Inputs are the tensors on the wire (command=act / observe, and the exported model's inputs). The names are
fixed, each optional, at most once per spec:
  obs     FP32  [N, D]          a vector (numbers the game computes)        encoder "mlp"
  frame   UINT8 [N, H, W, C]    an image as captured, channels last         encoder "cnn"
  audio   FP32  [N, S]          a window of samples                         reserved (no encoder yet)
  text    STRING [N]            a line of text                              reserved (no encoder yet)
Encoders: "mlp" = running mean/std normalisation (net.normalize_obs) then `hidden` layers (default none, the
vector goes straight to the torso); "cnn" = Nature-DQN convolutions on the frame scaled to [0, 1], then
`hidden` fully connected layers (default [512]). Every encoder's output is concatenated and fed to the torso
(net.hidden), which has one head per action group.

Action groups come back as ONE flat row per agent, in spec order: a continuous group contributes its values
(the mean, or a sample), a discrete group its one-hot; `action_index` holds the chosen index of every
discrete group ([N, number of discrete groups]) and `logp` is the log-prob summed over groups.

The v1 shorthand is still accepted and means exactly one input and one group:
  {"obs": {"dim": 24}, "action": {"type": "continuous", "dim": 6, "low": -1.0, "high": 1.0}, ...}
  == {"inputs": {"obs": {"shape": [24]}}, "actions": {"action": {"type": "continuous", "dim": 6, ...}}, ...}

versioning.score picks what "better" means: "return" (default) ranks versions by the mean return of finished
training episodes; "report" ranks them by the score the client reports with command=report (a game-defined
number such as kills per agent-minute), so exports on improvement wait for reports and shaping never leaks
into the ranking.

Network size (net.preset), the torso between the encoders and the heads:
  "auto"                     width derived from the encoder output size, two layers   (default)
  "small" | "medium" | "large"   fixed presets, see PRESETS
  "custom"                   use net.hidden as given, e.g. [512, 512, 256]
  "none"                     no torso, the heads sit on the encoder output (a cnn input's fc layer is the torso)
Giving net.hidden without a preset implies "custom". The resolved layers are reported back in status().
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

# wire name -> (dtype, rank of one row, default encoder). None = accepted on the wire, no encoder yet.
WIRE_INPUTS = {
    "obs": ("fp32", 1, "mlp"),
    "frame": ("uint8", 3, "cnn"),
    "audio": ("fp32", 1, None),
    "text": ("string", 0, None),
}

ENCODERS = ("mlp", "cnn")
CNN_DEFAULT_HIDDEN = [512]


@dataclass
class InputSpec:
    name: str
    shape: list[int]
    dtype: str = "fp32"  # fp32 | uint8 | string, fixed per wire name
    encoder: str = ""  # mlp | cnn, default per wire name
    hidden: list[int] = field(default_factory=list)  # layers inside the encoder (see module docstring)

    @property
    def size(self) -> int:
        n = 1
        for d in self.shape:
            n *= int(d)
        return n

    def validate(self) -> None:
        if self.name not in WIRE_INPUTS:
            raise ValueError(f"unknown input '{self.name}' ({', '.join(WIRE_INPUTS)})")
        dtype, rank, default_encoder = WIRE_INPUTS[self.name]
        if self.dtype != dtype:
            raise ValueError(f"input '{self.name}' is {dtype} on the wire, not {self.dtype}")
        if len(self.shape) != rank or any(int(d) <= 0 for d in self.shape):
            raise ValueError(f"input '{self.name}' needs a shape of {rank} positive dims, got {self.shape}")
        if not self.encoder:
            self.encoder = default_encoder or ""
        if self.encoder not in ENCODERS:
            what = f"'{self.encoder}'" if self.encoder else "no encoder yet"
            raise ValueError(f"input '{self.name}': {what} ({', '.join(ENCODERS)})")
        if self.encoder == "cnn" and self.name != "frame":
            raise ValueError(f"input '{self.name}': encoder cnn needs an image (use 'frame')")
        if self.encoder == "cnn" and not self.hidden:
            self.hidden = list(CNN_DEFAULT_HIDDEN)

    def to_json(self) -> dict:
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "encoder": self.encoder,
            "hidden": list(self.hidden),
        }


@dataclass
class ActionGroup:
    name: str
    type: str = "continuous"  # continuous | discrete
    dim: int = 0  # continuous
    n: int = 0  # discrete
    low: float = -1.0
    high: float = 1.0

    @property
    def continuous(self) -> bool:
        return self.type == "continuous"

    @property
    def out_dim(self) -> int:
        """Width of this group in the flat action row (values, or one-hot)."""
        return self.dim if self.continuous else self.n

    def validate(self) -> None:
        if self.type not in ("continuous", "discrete"):
            raise ValueError(f"action '{self.name}': unknown type '{self.type}' (continuous, discrete)")
        if self.out_dim <= 0:
            raise ValueError(f"action '{self.name}': needs dim > 0 (continuous) or n > 0 (discrete)")

    def to_json(self) -> dict:
        if self.continuous:
            return {"type": "continuous", "dim": self.dim, "low": self.low, "high": self.high}
        return {"type": "discrete", "n": self.n}


# kept for callers that only know the v1 name
ActionSpace = ActionGroup


@dataclass
class NetConfig:
    preset: str = "auto"  # auto | small | medium | large | custom | none
    hidden: list[int] = field(default_factory=list)  # used when preset == custom
    activation: str = "tanh"  # tanh | relu | elu | gelu
    layernorm: bool = False  # LayerNorm after every hidden layer
    separate_critic: bool = True  # False = critic head on the actor torso
    normalize_obs: bool = True  # running mean/std of vector inputs, baked into the export
    log_std_init: float = (
        0.0  # continuous actions: initial log std of the Gaussian (0 = std 1; -1 = std 0.37)
    )

    def resolve_hidden(self, feature_dim: int) -> list[int]:
        """The torso layer sizes this config means for a given encoder output size."""
        if self.preset == "custom":
            if not self.hidden:
                raise ValueError("net.preset=custom needs net.hidden")
            return list(self.hidden)
        if self.preset == "none":
            return []
        if self.preset in PRESETS:
            return list(PRESETS[self.preset])
        if self.preset == "auto":
            width = int(min(512, max(64, round(4 * feature_dim / 32) * 32)))
            return [width, width]
        raise ValueError(f"unknown net.preset '{self.preset}' (auto, custom, none, {', '.join(PRESETS)})")

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
class VersioningConfig:
    score_window: int = 20  # finished training episodes averaged to score the current version
    export_on_improvement: bool = True  # export the current version when its training score beats 'best'
    keep_exported: int = 3  # newest exported versions kept on disk (best/stable are always kept)
    trt_on_promote: bool = True  # build a TensorRT engine for a version when it is promoted to 'stable'
    score: str = (
        "return"  # what ranks versions and triggers exports: "return" = mean training episode return,
    )
    #                        "report" = the score the game reports (command=report), e.g. kills per agent-minute

    def validate(self) -> None:
        if self.score not in ("return", "report"):
            raise ValueError(f"unknown versioning.score '{self.score}' (return, report)")


@dataclass
class AgentSpec:
    inputs: list[InputSpec]
    actions: list[ActionGroup]
    net: NetConfig = field(default_factory=NetConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    versioning: VersioningConfig = field(default_factory=VersioningConfig)

    # ------------------------------------------------------------------ derived layout

    def input(self, name: str) -> InputSpec | None:
        for i in self.inputs:
            if i.name == name:
                return i
        return None

    @property
    def input_names(self) -> list[str]:
        return [i.name for i in self.inputs]

    @property
    def action_dim(self) -> int:
        """Width of the flat action row."""
        return sum(g.out_dim for g in self.actions)

    @property
    def continuous_dim(self) -> int:
        """Width of 'log_std': the continuous groups' values only."""
        return sum(g.dim for g in self.actions if g.continuous)

    @property
    def discrete_groups(self) -> list[ActionGroup]:
        return [g for g in self.actions if not g.continuous]

    @property
    def action_slices(self) -> dict[str, slice]:
        """Where each group sits in the flat action row."""
        out, start = {}, 0
        for g in self.actions:
            out[g.name] = slice(start, start + g.out_dim)
            start += g.out_dim
        return out

    @property
    def feature_dim(self) -> int:
        """Width of the concatenated encoder outputs (what the torso sees)."""
        total = 0
        for i in self.inputs:
            total += i.hidden[-1] if i.hidden else i.size
        return total

    @property
    def hidden(self) -> list[int]:
        return self.net.resolve_hidden(self.feature_dim)

    @property
    def is_v1(self) -> bool:
        """One plain vector in, one group out: what every client before the generic pipe sent."""
        return (
            len(self.inputs) == 1
            and self.inputs[0].name == "obs"
            and not self.inputs[0].hidden
            and len(self.actions) == 1
            and self.actions[0].name == "action"
        )

    @property
    def obs_dim(self) -> int:
        """Vector input size; only meaningful for v1 specs (kept for callers that predate inputs)."""
        obs = self.input("obs")
        return obs.size if obs else 0

    @property
    def action(self) -> ActionGroup:
        """The single action group of a v1 spec (first group otherwise)."""
        return self.actions[0]

    # ------------------------------------------------------------------ json

    @staticmethod
    def from_json(text: str) -> AgentSpec:
        data = json.loads(text)

        net_data = dict(data.get("net", {}))
        if "hidden" in data and "net" not in data:  # older clients: top-level hidden list
            net_data = {"preset": "custom", "hidden": list(data["hidden"])}
        if net_data.get("hidden") and "preset" not in net_data:
            net_data["preset"] = "custom"

        if "inputs" in data or "actions" in data:
            inputs = [InputSpec(name=k, **v) for k, v in _named(data.get("inputs", {}), "inputs")]
            actions = [ActionGroup(name=k, **v) for k, v in _named(data.get("actions", {}), "actions")]
        else:  # v1 shorthand
            if "obs" not in data:
                raise ValueError("spec needs 'inputs' (or the v1 'obs' + 'action')")
            inputs = [InputSpec(name="obs", shape=[int(data["obs"]["dim"])])]
            actions = [ActionGroup(name="action", **data.get("action", {}))]

        spec = AgentSpec(
            inputs=inputs,
            actions=actions,
            net=NetConfig(**net_data),
            ppo=PPOConfig(**data.get("ppo", {})),
            versioning=VersioningConfig(**data.get("versioning", {})),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if not self.inputs:
            raise ValueError("spec needs at least one input")
        if not self.actions:
            raise ValueError("spec needs at least one action group")
        seen = set()
        for i in self.inputs:
            if i.name in seen:
                raise ValueError(f"input '{i.name}' given twice")
            seen.add(i.name)
            i.validate()
        seen = set()
        for g in self.actions:
            if g.name in seen:
                raise ValueError(f"action '{g.name}' given twice")
            seen.add(g.name)
            g.validate()
        self.net.validate()
        self.versioning.validate()
        self.net.resolve_hidden(self.feature_dim)  # raises early on a bad preset

    def to_json(self) -> str:
        return json.dumps(
            {
                "inputs": {i.name: i.to_json() for i in self.inputs},
                "actions": {g.name: g.to_json() for g in self.actions},
                "net": asdict(self.net),
                "ppo": asdict(self.ppo),
                "versioning": asdict(self.versioning),
            }
        )

    def describe(self) -> dict:
        """What the trainer actually built from this spec; reported in status()."""
        return {
            "inputs": {i.name: i.to_json() for i in self.inputs},
            "actions": {g.name: g.to_json() for g in self.actions},
            "action_dim": self.action_dim,
            "feature_dim": self.feature_dim,
            "hidden": self.hidden,
            "activation": self.net.activation,
            "layernorm": self.net.layernorm,
            "separate_critic": self.net.separate_critic,
            "normalize_obs": self.net.normalize_obs,
            "log_std_init": self.net.log_std_init,
        }


def _named(section, what: str) -> list[tuple[str, dict]]:
    """'inputs' / 'actions' as a name -> fields dict (in order) or a list of dicts with a 'name' field."""
    if isinstance(section, dict):
        return [(str(k), dict(v)) for k, v in section.items()]
    if isinstance(section, list):
        out = []
        for entry in section:
            entry = dict(entry)
            name = entry.pop("name", None)
            if not name:
                raise ValueError(f"{what}: every list entry needs a 'name'")
            out.append((str(name), entry))
        return out
    raise ValueError(f"{what} must be an object or a list")
