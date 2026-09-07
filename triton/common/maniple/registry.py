"""PolicyRegistry: named policies living inside an '<algo>_train' Triton model.

Policy = spec + Algorithm instance + TransitionBuffer + a background thread that calls algorithm.update()
whenever the buffer holds algorithm.rollout_size transitions, then exports '<name>_policy/<version>'
and checkpoints. Algorithm-agnostic: pass the Algorithm class at construction.
"""

from __future__ import annotations

import os
import threading
import time

import torch

from .algorithms.base import Algorithm
from .buffer import TransitionBuffer
from .export import export_actor, latest_exported_version
from .spec import AgentSpec


class Policy:
    def __init__(self, name: str, spec: AgentSpec, registry: PolicyRegistry):
        self.name, self.spec, self.reg = name, spec, registry
        self.algo: Algorithm = registry.algorithm_cls(spec, registry.device)
        self.buffer = TransitionBuffer(spec.obs_dim, spec.action.out_dim)
        # exported version == Triton model version of '<name>_policy'; continue after whatever is already there
        self.version = latest_exported_version(name, registry.model_repository)
        self.updates = 0
        self.total_samples = 0
        self.last_stats = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name=f"train-{name}", daemon=True)

    def start(self):
        if self.version == 0:
            self.export()  # version 1 = untrained, so '<name>_policy' exists immediately
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self):
        self._stop.set()

    def observe(self, obs, action, reward, done, agent_id, episode_id, policy_version, logp=None) -> int:
        if obs is None or action is None or reward is None or done is None:
            raise ValueError("observe needs obs, action, reward, done")
        n = self.buffer.add(
            obs,
            action,
            reward,
            done,
            agent_id,
            episode_id,
            policy_version,
            logp,
            current_version=self.version,
            max_lag=self.spec.ppo.max_policy_lag,
        )
        self.total_samples += n
        return n

    def _loop(self):
        while not self._stop.is_set():
            if len(self.buffer) >= self.algo.rollout_size:
                self.last_stats = self.algo.update(self.buffer.take())
                self.updates += 1
                if self.updates % self.reg.export_every_updates == 0:
                    self.export()
                self.checkpoint()
            else:
                time.sleep(0.05)

    def export(self) -> int:
        self.version += 1
        with self.reg.export_lock:
            export_actor(self.algo.actor(), self.spec, self.name, self.version, self.reg.model_repository)
        self.algo.actor().to(self.reg.device)
        return self.version

    def checkpoint(self):
        d = os.path.join(self.reg.checkpoint_dir, self.name)
        os.makedirs(d, exist_ok=True)
        torch.save(
            {
                "algo": self.algo.state_dict(),
                "version": self.version,
                "updates": self.updates,
                "total_samples": self.total_samples,
            },
            os.path.join(d, "state.pt"),
        )
        with open(os.path.join(d, "spec.json"), "w") as f:
            f.write(self.spec.to_json())

    def restore(self, d: str):
        st = torch.load(os.path.join(d, "state.pt"), map_location=self.reg.device)
        self.algo.load_state_dict(st["algo"])
        self.version, self.updates, self.total_samples = st["version"], st["updates"], st["total_samples"]

    def status(self) -> dict:
        return {
            "name": self.name,
            "algorithm": type(self.algo).__name__,
            "version": self.version,
            "updates": self.updates,
            "buffered": len(self.buffer),
            "total_samples": self.total_samples,
            "dropped_stale": self.buffer.dropped_stale,
            "stats": self.last_stats,
        }


class PolicyRegistry:
    def __init__(
        self,
        algorithm_cls: type[Algorithm],
        model_repository: str,
        checkpoint_dir: str,
        export_every_updates: int,
        device: str,
    ):
        self.algorithm_cls = algorithm_cls
        self.model_repository, self.checkpoint_dir = model_repository, checkpoint_dir
        self.export_every_updates = max(1, export_every_updates)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.export_lock = threading.Lock()
        self._policies: dict[str, Policy] = {}
        self._lock = threading.Lock()

    def get(self, name: str) -> Policy | None:
        return self._policies.get(name)

    def get_or_create(self, name: str, spec: AgentSpec) -> Policy:
        with self._lock:
            p = self._policies.get(name)
            if p is None:
                p = Policy(name, spec, self)
                p.start()  # may raise; then nothing is registered
                self._policies[name] = p
            return p

    def load_all(self):
        if not os.path.isdir(self.checkpoint_dir):
            return
        for name in os.listdir(self.checkpoint_dir):
            d = os.path.join(self.checkpoint_dir, name)
            if not os.path.exists(os.path.join(d, "spec.json")):
                continue
            with open(os.path.join(d, "spec.json")) as f:
                spec = AgentSpec.from_json(f.read())
            p = Policy(name, spec, self)
            p.restore(d)
            self._policies[name] = p
            p.start()

    def shutdown(self):
        for p in self._policies.values():
            p.stop()
