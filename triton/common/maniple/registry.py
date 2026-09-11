"""PolicyRegistry: named policies living inside an '<algo>_train' Triton model.

Policy = spec + Algorithm + TransitionBuffer + VersionTracker + a background training thread.
- version number = update count; 'latest' is served from memory (act()), never from the repository
- exports happen on events only: best training score improved (optional), command=export, promote
- promote builds a TensorRT engine in the background (optional)
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback

import numpy as np
import torch

from .algorithms.base import Algorithm
from .buffer import TransitionBuffer
from .export import build_trt, export_actor, latest_exported_version, policy_model_name, prune_versions
from .spec import AgentSpec
from .versions import VersionTracker


class Policy:
    def __init__(self, name: str, spec: AgentSpec, registry: PolicyRegistry):
        self.name, self.spec, self.reg = name, spec, registry
        self.algo: Algorithm = registry.algorithm_cls(spec, registry.device)
        self.buffer = TransitionBuffer(spec)
        self.model_dir = os.path.join(registry.model_repository, policy_model_name(name))
        self.versions = VersionTracker(
            model_dir=self.model_dir,
            score_window=spec.versioning.score_window,
            keep_exported=spec.versioning.keep_exported,
        )

        self.version = max(1, self.versions.latest, latest_exported_version(name, registry.model_repository))
        self.updates = 0
        self._decided_report_seq = 0  # versioning.score == 'report': last report seq acted on
        self.total_samples = 0
        self.last_stats = {}

        self.weights_lock = threading.Lock()  # act() vs update()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name=f"train-{name}", daemon=True)

    # ------------------------------------------------------------------ lifecycle

    def start(self):
        self.versions.set_latest(self.version, save=True)
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self):
        self._stop.set()

    # ------------------------------------------------------------------ acting (channel 'latest')

    @torch.no_grad()
    def act(self, inputs: dict[str, np.ndarray], explore: bool):
        """inputs {name: [N, ...]} -> (action row, action_index, logp) with the current weights."""
        self._check_inputs(inputs)
        actor = self.algo.actor()
        with self.weights_lock:
            actor.eval()
            tensors = {name: torch.as_tensor(array, device=self.reg.device) for name, array in inputs.items()}
            dist = actor.distribution(tensors)
            action = dist.sample() if explore else dist.mode()
            logp = dist.log_prob(action)
            index = dist.indices(action)
        return (
            action.cpu().numpy().astype(np.float32),
            index.cpu().numpy().astype(np.int64),
            logp.cpu().numpy().astype(np.float32),
        )

    def _check_inputs(self, inputs: dict[str, np.ndarray]) -> None:
        rows = None
        for i in self.spec.inputs:
            array = inputs.get(i.name)
            if array is None:
                raise ValueError(
                    f"input '{i.name}' missing (this policy takes: {', '.join(self.spec.input_names)})"
                )
            if list(array.shape[1:]) != [int(d) for d in i.shape]:
                raise ValueError(
                    f"input '{i.name}' is {list(array.shape)}, expected [N, {', '.join(map(str, i.shape))}]"
                )
            if rows is not None and array.shape[0] != rows:
                raise ValueError("inputs have different row counts")
            rows = array.shape[0]

    # ------------------------------------------------------------------ data in

    def observe(self, inputs, action, reward, done, agent_id, episode_id, policy_version, logp=None) -> int:
        if not inputs or action is None or reward is None or done is None:
            raise ValueError("observe needs the policy's inputs, action, reward, done")
        n = self.buffer.add(
            inputs,
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
        if agent_id is not None and episode_id is not None:
            self.versions.observe_rows(agent_id, episode_id, reward, done)
        return n

    # ------------------------------------------------------------------ training loop

    def _loop(self):
        while not self._stop.is_set():
            if len(self.buffer) < self.algo.rollout_size:
                time.sleep(0.05)
                continue

            trajectories = self.buffer.take()
            try:
                with self.weights_lock:
                    self.last_stats = self.algo.update(trajectories)
            except Exception as e:  # noqa: BLE001 - a bad batch must not kill the trainer (the buffer would grow forever)
                self.last_stats = {"error": f"{type(e).__name__}: {e}"}
                print(
                    f"[maniple] policy '{self.name}': update failed, batch dropped ({self.last_stats['error']})"
                )
                traceback.print_exc()
                continue
            self.updates += 1
            self.version += 1
            self.versions.set_latest(self.version)

            if self.spec.versioning.export_on_improvement:
                self._export_if_improved()
            self.checkpoint()

    def _export_if_improved(self):
        if self.spec.versioning.score == "report":
            # one decision per report: the game's own score against the best exported version
            if self.versions.last_report is None or self.versions.report_seq == self._decided_report_seq:
                return
            self._decided_report_seq = self.versions.report_seq
            current = self.versions.last_report[1]
        else:
            current = self.versions.current_train_score()
        best = self.versions.best_score()
        if current is not None and (best is None or current > best):
            self.export()

    # ------------------------------------------------------------------ exports

    def export(self) -> int:
        """Write the current weights as 'policy_<name>/<version>'. Returns the version."""
        version = self.version
        with self.reg.export_lock:
            with self.weights_lock:
                export_actor(self.algo.actor(), self.spec, self.name, version, self.reg.model_repository)
            self.versions.on_export(version)
            if self.spec.versioning.score == "report":
                self.versions.attribute_last_report(version)
            self._prune()
        return version

    def _prune(self):
        keep = self.versions.exported_to_keep()
        on_disk = {int(v) for v in os.listdir(self.model_dir) if v.isdigit()}
        prune_versions(self.model_dir, keep)
        self.versions.drop_exported(on_disk - keep)

    def report(self, version: int, score: float, episodes: int) -> None:
        self.versions.report(version, score, episodes)

    def promote(self, version: int) -> None:
        if version == self.version and version not in self.versions.exported:
            self.export()
        if version not in self.versions.exported:
            raise ValueError(f"version {version} is not exported (exported: {self.versions.exported})")
        self.versions.promote(version)
        if self.spec.versioning.trt_on_promote:
            threading.Thread(
                target=self._build_trt, args=(version,), name=f"trt-{self.name}", daemon=True
            ).start()

    def _build_trt(self, version: int):
        self.versions.set_trt(version, "building")
        try:
            build_trt(self.name, version, self.spec, self.reg.model_repository)
            self.versions.set_trt(version, "ready")
        except Exception as e:  # noqa: BLE001 - reported through the manifest, never fatal
            self.versions.set_trt(version, "failed", f"{type(e).__name__}: {e}")

    # ------------------------------------------------------------------ persistence

    def checkpoint(self):
        d = os.path.join(self.reg.checkpoint_dir, self.name)
        os.makedirs(d, exist_ok=True)
        state = {
            "algo": self.algo.state_dict(),
            "version": self.version,
            "updates": self.updates,
            "total_samples": self.total_samples,
        }
        torch.save(state, os.path.join(d, "state.pt"))
        with open(os.path.join(d, "spec.json"), "w") as f:
            f.write(self.spec.to_json())

    def restore(self, d: str):
        st = torch.load(os.path.join(d, "state.pt"), map_location=self.reg.device)
        self.algo.load_state_dict(st["algo"])
        self.version = max(self.version, int(st["version"]))
        self.updates = int(st["updates"])
        self.total_samples = int(st["total_samples"])

    def status(self) -> dict:
        return {
            "name": self.name,
            "algorithm": type(self.algo).__name__,
            "version": self.version,
            "updates": self.updates,
            "buffered": len(self.buffer),
            "total_samples": self.total_samples,
            "dropped_stale": self.buffer.dropped_stale,
            "net": self.spec.describe(),
            "spec": json.loads(self.spec.to_json()),
            "versions": self.versions.summary(),
            "stats": self.last_stats,
        }


class PolicyRegistry:
    def __init__(
        self, algorithm_cls: type[Algorithm], model_repository: str, checkpoint_dir: str, device: str
    ):
        self.algorithm_cls = algorithm_cls
        self.model_repository, self.checkpoint_dir = model_repository, checkpoint_dir
        self.device = device if torch.cuda.is_available() else "cpu"
        self.export_lock = threading.Lock()
        self._policies: dict[str, Policy] = {}
        self._lock = threading.Lock()

    def get(self, name: str) -> Policy | None:
        return self._policies.get(name)

    def get_or_create(self, name: str, spec: AgentSpec) -> Policy:
        with self._lock:
            p = self._policies.get(name)
            if p is not None:
                if p.spec.to_json() != spec.to_json():
                    raise ValueError(
                        f"policy '{name}' exists with a different spec; use another name, or delete "
                        f"model_repository/policy_{name} and checkpoints/{name} to start over"
                    )
                return p
            p = Policy(name, spec, self)
            p.start()  # may raise; then nothing is registered
            self._policies[name] = p
            return p

    def load_all(self):
        """Restore checkpointed policies after a server restart."""
        if not os.path.isdir(self.checkpoint_dir):
            return
        for name in sorted(os.listdir(self.checkpoint_dir)):
            d = os.path.join(self.checkpoint_dir, name)
            if not os.path.exists(os.path.join(d, "spec.json")):
                continue
            with open(os.path.join(d, "spec.json")) as f:
                spec = AgentSpec.from_json(f.read())
            p = Policy(name, spec, self)
            try:
                p.restore(d)
            except Exception as e:  # noqa: BLE001 - a broken checkpoint must not take the server down
                print(
                    f"[maniple] policy '{name}': checkpoint not restored ({type(e).__name__}: {e}); skipped"
                )
                continue
            self._policies[name] = p
            p.start()

    def shutdown(self):
        for p in self._policies.values():
            p.stop()
