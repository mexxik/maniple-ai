"""Version bookkeeping for one named policy: scores per exported version and the channel manifest.

Channels (what clients ask for instead of numbers):
  latest   newest export, what training actors must use
  best     highest score among versions with enough episodes, what players/evaluation use
  stable   manually promoted (command=promote), never moves on its own

The manifest is written next to the model as '<name>_policy/versions.json' so the separate ppo_infer process
can resolve channels; it survives restarts and doubles as a human-readable history.

Scores come from two sources, kept apart:
  train  mean return of episodes played by that version during training (exploring policy, free)
  eval   scores reported by clients after greedy episodes (command=report), trusted over train when present
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class VersionScore:
    train_return_sum: float = 0.0
    train_episodes: int = 0
    eval_score: float | None = None
    eval_episodes: int = 0

    @property
    def train_mean(self) -> float | None:
        return self.train_return_sum / self.train_episodes if self.train_episodes else None

    def ranking_score(self, min_episodes: int) -> float | None:
        """Score used to pick 'best', or None if this version has not been measured enough."""
        if self.eval_score is not None and self.eval_episodes >= 1:
            return self.eval_score
        if self.train_episodes >= min_episodes:
            return self.train_mean
        return None

    def to_json(self) -> dict:
        return {
            "train_mean_return": self.train_mean,
            "train_episodes": self.train_episodes,
            "eval_score": self.eval_score,
            "eval_episodes": self.eval_episodes,
        }


class VersionTracker:
    """Tracks episodes per version from observe() rows, picks channels, writes the manifest."""

    def __init__(self, model_dir: str, min_episodes: int, keep_latest: int):
        self.model_dir = model_dir
        self.manifest_path = os.path.join(model_dir, "versions.json")
        self.min_episodes = min_episodes
        self.keep_latest = keep_latest

        self.latest = 0
        self.best: int | None = None
        self.stable: int | None = None
        self.scores: dict[int, VersionScore] = defaultdict(VersionScore)

        # per (agent, episode): reward accumulated per version, to attribute the episode to one version
        self._open_episodes: dict[tuple[int, int], dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self._lock = threading.Lock()
        self._load()

    # ------------------------------------------------------------------ episode accounting

    def observe_rows(self, agent_id, episode_id, reward, done, policy_version) -> None:
        """Feed one observe() batch (numpy arrays, one entry per agent)."""
        with self._lock:
            for i in range(len(reward)):
                key = (int(agent_id[i]), int(episode_id[i]))
                version = int(policy_version[i])
                self._open_episodes[key][version] += float(reward[i])
                if bool(done[i]):
                    self._close_episode(key)

    def _close_episode(self, key) -> None:
        by_version = self._open_episodes.pop(key)
        # the version that contributed most of the episode owns its return
        owner = max(by_version, key=lambda v: abs(by_version[v]))
        score = self.scores[owner]
        score.train_return_sum += sum(by_version.values())
        score.train_episodes += 1

    # ------------------------------------------------------------------ commands

    def report(self, version: int, score: float, episodes: int = 1) -> None:
        """A client's greedy evaluation of `version` (running average over reports)."""
        with self._lock:
            s = self.scores[version]
            total = s.eval_episodes + episodes
            previous = s.eval_score if s.eval_score is not None else 0.0
            s.eval_score = (previous * s.eval_episodes + score * episodes) / total
            s.eval_episodes = total
            self._recompute_best()
            self._save()

    def promote(self, version: int) -> None:
        with self._lock:
            self.stable = version
            self._save()

    def on_export(self, version: int) -> None:
        with self._lock:
            self.latest = version
            self.scores.setdefault(version, VersionScore())
            self._recompute_best()
            self._save()

    def refresh(self) -> None:
        """Recompute best from current scores and persist (called after training updates)."""
        with self._lock:
            self._recompute_best()
            self._save()

    # ------------------------------------------------------------------ channels

    def resolve(self, channel: str) -> int | None:
        with self._lock:
            if channel == "latest":
                return self.latest or None
            if channel == "best":
                return self.best or self.latest or None
            if channel == "stable":
                return self.stable or self.best or self.latest or None
            return int(channel) if channel.isdigit() else None

    def versions_to_keep(self, on_disk: list[int]) -> set[int]:
        """Which exported version directories must survive pruning."""
        with self._lock:
            keep = set(sorted(on_disk)[-self.keep_latest :])
            for pinned in (self.best, self.stable):
                if pinned is not None:
                    keep.add(pinned)
            return keep

    def summary(self) -> dict:
        with self._lock:
            return self._manifest()

    # ------------------------------------------------------------------ internals

    def _recompute_best(self) -> None:
        on_disk = self._versions_on_disk()
        candidates = []
        for version, score in self.scores.items():
            if version not in on_disk:
                continue
            value = score.ranking_score(self.min_episodes)
            if value is not None:
                candidates.append((value, version))
        if candidates:
            self.best = max(candidates)[1]
        elif self.best not in on_disk:
            self.best = None

    def _versions_on_disk(self) -> set[int]:
        if not os.path.isdir(self.model_dir):
            return set()
        return {int(v) for v in os.listdir(self.model_dir) if v.isdigit()}

    def _manifest(self) -> dict:
        return {
            "latest": self.latest,
            "best": self.best,
            "stable": self.stable,
            "min_episodes": self.min_episodes,
            "scores": {str(v): s.to_json() for v, s in sorted(self.scores.items())},
        }

    def _save(self) -> None:
        os.makedirs(self.model_dir, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".versions-", dir=self.model_dir)
        with os.fdopen(fd, "w") as f:
            json.dump(self._manifest(), f, indent=1)
        os.chmod(tmp, 0o644)
        os.replace(tmp, self.manifest_path)

    def _load(self) -> None:
        if not os.path.exists(self.manifest_path):
            return
        with open(self.manifest_path) as f:
            data = json.load(f)
        self.latest = int(data.get("latest") or 0)
        self.best = data.get("best")
        self.stable = data.get("stable")
        for version, s in data.get("scores", {}).items():
            score = self.scores[int(version)]
            mean, episodes = s.get("train_mean_return"), int(s.get("train_episodes", 0))
            score.train_episodes = episodes
            score.train_return_sum = (mean or 0.0) * episodes
            score.eval_score = s.get("eval_score")
            score.eval_episodes = int(s.get("eval_episodes", 0))


def read_manifest(model_dir: str) -> dict | None:
    """Read-only access for other processes (ppo_infer)."""
    path = os.path.join(model_dir, "versions.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
