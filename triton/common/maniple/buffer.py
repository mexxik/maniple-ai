"""Transition buffer fed by observe() rows from many game instances.

A row is one agent step: (agent_id, episode_id, inputs, action, reward, done, policy_version, logp), where
`inputs` is one array per spec input (a vector, a frame, ...) kept in its wire dtype (frames stay uint8).
take() groups rows into trajectories per (agent_id, episode_id), in arrival order, and empties the buffer.

Policy lag: rows produced by a policy version older than `max_lag` versions behind the trainer are dropped
(counted in `dropped_stale`). Rows within the allowed lag are kept; PPO corrects for them with the logp.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .spec import AgentSpec


@dataclass
class Row:
    agent_id: int
    episode_id: int
    inputs: dict[str, np.ndarray]
    action: np.ndarray
    reward: float
    done: bool
    policy_version: int
    logp: float  # NaN when the client did not send one


class TransitionBuffer:
    def __init__(self, spec: AgentSpec):
        self.spec = spec
        self.dropped_stale = 0
        self._rows: list[Row] = []
        self._lock = threading.Lock()

    def add(
        self,
        inputs: dict[str, np.ndarray],
        action,
        reward,
        done,
        agent_id,
        episode_id,
        policy_version,
        logp,
        current_version: int,
        max_lag: int,
    ) -> int:
        """Append a batch of rows (one per agent). Returns how many were accepted."""
        self._check(inputs, action)
        n = int(action.shape[0])
        accepted = 0

        with self._lock:
            for i in range(n):
                version = int(policy_version[i]) if policy_version is not None else current_version
                if current_version - version > max_lag:
                    self.dropped_stale += 1
                    continue

                self._rows.append(
                    Row(
                        agent_id=int(agent_id[i]) if agent_id is not None else 0,
                        episode_id=int(episode_id[i]) if episode_id is not None else 0,
                        inputs={name: np.array(array[i], copy=True) for name, array in inputs.items()},
                        action=action[i].astype(np.float32),
                        reward=float(reward[i]),
                        done=bool(done[i]),
                        policy_version=version,
                        logp=float(logp[i]) if logp is not None else np.nan,
                    )
                )
                accepted += 1

        return accepted

    def _check(self, inputs: dict[str, np.ndarray], action) -> None:
        """Every spec input present with the right row shape, rows aligned with the action."""
        n = int(action.shape[0])
        for i in self.spec.inputs:
            array = inputs.get(i.name)
            if array is None:
                raise ValueError(f"observe: input '{i.name}' missing")
            if array.shape[0] != n or list(array.shape[1:]) != [int(d) for d in i.shape]:
                raise ValueError(
                    f"observe: input '{i.name}' is {list(array.shape)}, expected [{n}, {', '.join(map(str, i.shape))}]"
                )
        if action.shape[1:] != (self.spec.action_dim,):
            raise ValueError(
                f"observe: action is {list(action.shape)}, expected [{n}, {self.spec.action_dim}]"
            )

    def __len__(self) -> int:
        with self._lock:
            return len(self._rows)

    def take(self) -> list[dict]:
        """Return all buffered rows as per-trajectory arrays and clear the buffer."""
        with self._lock:
            rows, self._rows = self._rows, []

        by_trajectory: dict[tuple[int, int], list[Row]] = defaultdict(list)
        for row in rows:
            by_trajectory[(row.agent_id, row.episode_id)].append(row)

        trajectories = []
        for trajectory in by_trajectory.values():
            trajectories.append(
                {
                    "inputs": {
                        i.name: np.stack([r.inputs[i.name] for r in trajectory]) for i in self.spec.inputs
                    },
                    "action": np.stack([r.action for r in trajectory]),
                    "reward": np.array([r.reward for r in trajectory], dtype=np.float32),
                    "done": np.array([r.done for r in trajectory], dtype=bool),
                    "logp": np.array([r.logp for r in trajectory], dtype=np.float32),
                }
            )
        return trajectories
