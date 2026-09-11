"""Algorithm interface: what the registry needs from any RL algorithm."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn

from ..spec import AgentSpec


class Algorithm(ABC):
    """One instance per named policy. Owns its networks and optimizer."""

    #: how many transitions the buffer must hold before update() is called
    rollout_size: int

    def __init__(self, spec: AgentSpec, device: str):
        self.spec = spec
        self.device = device

    @abstractmethod
    def actor(self) -> nn.Module:
        """Inference network: inputs -> flat action row (means / logits). This is what gets exported as ONNX."""

    @abstractmethod
    def update(self, trajectories: list[dict]) -> dict:
        """One training step on trajectories from TransitionBuffer.take(). Returns stats for status()."""

    @abstractmethod
    def state_dict(self) -> dict: ...

    @abstractmethod
    def load_state_dict(self, state: dict) -> None: ...
