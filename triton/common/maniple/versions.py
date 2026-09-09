"""Version bookkeeping for one named policy: scores, exported versions, channels, TensorRT builds.

A version number = the trainer's update count. Only some versions are exported as static Triton models;
'latest' is always served straight from the trainer's memory, so training never touches the repository.

Channels (what clients ask for instead of numbers):
  latest   the trainer's current weights (served by <algo>_train, command=act)
  best     exported version with the highest score
  stable   manually promoted (command=promote); gets a TensorRT engine if enabled

The manifest 'policy_<name>/versions.json' is written by the trainer and read by <algo>_infer (a separate
process); it survives restarts and doubles as a human-readable history.

Scores:
  train  mean return of the last `score_window` finished training episodes, attributed to the version that
         was current when they finished (exploring policy, free)
  eval   scores reported by clients after greedy episodes (command=report); trusted over train
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass
class VersionScore:
    train_score: float | None = None  # window mean at the time this version was current
    train_episodes: int = 0
    eval_score: float | None = None
    eval_episodes: int = 0

    def ranking_score(self) -> float | None:
        if self.eval_score is not None:
            return self.eval_score
        return self.train_score

    def to_json(self) -> dict:
        return {
            "train_score": self.train_score,
            "train_episodes": self.train_episodes,
            "eval_score": self.eval_score,
            "eval_episodes": self.eval_episodes,
        }


class VersionTracker:
    def __init__(self, model_dir: str, score_window: int, keep_exported: int):
        self.model_dir = model_dir
        self.manifest_path = os.path.join(model_dir, "versions.json")
        self.score_window = score_window
        self.keep_exported = keep_exported

        self.latest = 0  # trainer's current version (update count)
        self.best: int | None = None
        self.stable: int | None = None
        self.exported: list[int] = []
        self.trt: dict = {}  # {"version": n, "state": "building" | "ready" | "failed", "error": ...}
        self.scores: dict[int, VersionScore] = defaultdict(VersionScore)

        self._recent_returns: deque[float] = deque(maxlen=score_window)
        self.report_seq = 0  # counts report() calls; lets the trainer act once per new report
        self.last_report: tuple[int, float, int] | None = (
            None  # (version, score, episodes) of the newest report
        )
        self._open_episodes: dict[tuple[int, int], float] = defaultdict(float)
        self._lock = threading.Lock()
        self._load()

    # ------------------------------------------------------------------ training-side updates

    def set_latest(self, version: int, save: bool = False) -> None:
        with self._lock:
            self.latest = version
            self.scores.setdefault(version, VersionScore())
            self._save()

    def observe_rows(self, agent_id, episode_id, reward, done) -> None:
        """Feed one observe() batch; finished episodes score the current version."""
        with self._lock:
            for i in range(len(reward)):
                key = (int(agent_id[i]), int(episode_id[i]))
                self._open_episodes[key] += float(reward[i])
                if bool(done[i]):
                    self._recent_returns.append(self._open_episodes.pop(key))
            self._refresh_current_score()

    def _refresh_current_score(self) -> None:
        if len(self._recent_returns) < self.score_window:
            return
        score = self.scores[self.latest]
        score.train_score = sum(self._recent_returns) / len(self._recent_returns)
        score.train_episodes = len(self._recent_returns)

    def current_train_score(self) -> float | None:
        """Mean of the last `score_window` finished episodes (None until the window is full)."""
        with self._lock:
            if len(self._recent_returns) < self.score_window:
                return None
            return sum(self._recent_returns) / len(self._recent_returns)

    def best_score(self) -> float | None:
        with self._lock:
            return self.scores[self.best].ranking_score() if self.best is not None else None

    # ------------------------------------------------------------------ exports and channels

    def on_export(self, version: int) -> None:
        with self._lock:
            if version not in self.exported:
                self.exported.append(version)
            score = self.scores.setdefault(version, VersionScore())
            if score.train_score is None and len(self._recent_returns) >= self.score_window:
                score.train_score = sum(self._recent_returns) / len(self._recent_returns)
                score.train_episodes = len(self._recent_returns)
            self._recompute_best()
            self._save()

    def report(self, version: int, score: float, episodes: int = 1) -> None:
        with self._lock:
            self._add_report(version, score, episodes)
            self.report_seq += 1
            self.last_report = (version, score, episodes)
            self._recompute_best()
            self._save()

    def attribute_last_report(self, version: int) -> None:
        """Give an unscored version the newest reported score (the export it just triggered)."""
        with self._lock:
            if self.last_report is None or self.scores[version].eval_score is not None:
                return
            _, score, episodes = self.last_report
            self._add_report(version, score, episodes)
            self._recompute_best()
            self._save()

    def _add_report(self, version: int, score: float, episodes: int) -> None:
        s = self.scores[version]
        total = s.eval_episodes + episodes
        previous = s.eval_score if s.eval_score is not None else 0.0
        s.eval_score = (previous * s.eval_episodes + score * episodes) / total
        s.eval_episodes = total

    def promote(self, version: int) -> None:
        with self._lock:
            self.stable = version
            self._save()

    def set_trt(self, version: int, state: str, error: str | None = None) -> None:
        with self._lock:
            self.trt = {"version": version, "state": state}
            if error:
                self.trt["error"] = error
            self._save()

    def exported_to_keep(self) -> set[int]:
        with self._lock:
            keep = set(sorted(self.exported)[-self.keep_exported :])
            for pinned in (self.best, self.stable):
                if pinned is not None:
                    keep.add(pinned)
            return keep

    def drop_exported(self, versions: set[int]) -> None:
        with self._lock:
            self.exported = [v for v in self.exported if v not in versions]
            self._recompute_best()
            self._save()

    def summary(self) -> dict:
        with self._lock:
            return self._manifest()

    # ------------------------------------------------------------------ internals

    def _recompute_best(self) -> None:
        candidates = []
        for version in self.exported:
            value = self.scores[version].ranking_score() if version in self.scores else None
            if value is not None:
                candidates.append((value, version))
        if candidates:
            self.best = max(candidates)[1]
        elif self.best not in self.exported:
            self.best = None

    def _manifest(self) -> dict:
        return {
            "latest": self.latest,
            "best": self.best,
            "stable": self.stable,
            "exported": sorted(self.exported),
            "trt": self.trt,
            "last_report": (
                {
                    "version": self.last_report[0],
                    "score": self.last_report[1],
                    "episodes": self.last_report[2],
                }
                if self.last_report
                else None
            ),
            "scores": {
                str(v): s.to_json()
                for v, s in sorted(self.scores.items())
                if v in self.exported or v == self.latest
            },
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
        self.exported = [int(v) for v in data.get("exported", [])]
        self.trt = data.get("trt", {})
        for version, s in data.get("scores", {}).items():
            score = self.scores[int(version)]
            score.train_score = s.get("train_score")
            score.train_episodes = int(s.get("train_episodes", 0))
            score.eval_score = s.get("eval_score")
            score.eval_episodes = int(s.get("eval_episodes", 0))


def read_manifest(model_dir: str) -> dict | None:
    """Read-only access for other processes (<algo>_infer)."""
    path = os.path.join(model_dir, "versions.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
