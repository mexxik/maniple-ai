"""InferModel: base class for '<algo>_infer/<version>/model.py' — the stable inference entry point.

    from maniple.infer_model import InferModel
    class TritonPythonModel(InferModel): pass

Inputs : name (STRING [1]), obs (FP32 [N, obs_dim]), explore (BOOL [1], optional, default false),
         channel (STRING [1], optional): stable (default) | best | latest | "<version>"
Outputs: action         FP32 [N, act_dim]  continuous action, or one-hot of the chosen discrete action
         action_index   INT64 [N]          chosen index for discrete, -1 for continuous
         logp           FP32 [N]           log-prob of the returned action under the served policy
         policy_version INT64 [1]          version that produced the action
         status         STRING [1]         JSON {ok, policy, channel, served_by | error}

Routing
  latest         -> '<algo>_train' command=act (the trainer's current weights, no repository involved)
  best / stable  -> exported static model 'policy_<name>' (ONNX) or 'policy_<name>_trt' (TensorRT, when built
                    for that version), loaded on first use (server runs in explicit model-control mode) and
                    unloaded again after `idle_unload_sec` without requests
  "<version>"    -> that exported version
Exploration is sampled here (Normal for continuous, Categorical for discrete) so game clients stay generic.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time

import numpy as np
import triton_python_backend_utils as pb_utils

from .export import policy_model_name, trt_model_name
from .versions import read_manifest


def _string_input(request, name):
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is None:
        return None
    value = tensor.as_numpy().reshape(-1)[0]
    return value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)


def _bool_input(request, name, default):
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is None:
        return default
    return bool(tensor.as_numpy().reshape(-1)[0])


def _to_numpy(tensor):
    """BLS outputs may live on the GPU even with preferred_memory; go through DLPack in that case."""
    if tensor.is_cpu():
        return tensor.as_numpy()
    import torch

    return torch.from_dlpack(tensor.to_dlpack()).cpu().numpy()


class _ManifestCache:
    """versions.json per policy, re-read when its mtime changes."""

    def __init__(self, model_repository: str):
        self.model_repository = model_repository
        self._cache: dict[str, tuple[float, dict | None]] = {}

    def get(self, name: str) -> dict | None:
        model_dir = os.path.join(self.model_repository, policy_model_name(name))
        path = os.path.join(model_dir, "versions.json")
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return None
        cached = self._cache.get(name)
        if cached is None or cached[0] != mtime:
            self._cache[name] = (mtime, read_manifest(model_dir))
        return self._cache[name][1]


class _ModelLoader:
    """Loads exported models on first use (explicit model control) and unloads idle ones."""

    def __init__(self, idle_unload_sec: float):
        self.idle_unload_sec = idle_unload_sec
        self._last_used: dict[str, float] = {}
        self._loaded_dirs: dict[str, set[str]] = {}
        self._lock = threading.Lock()
        threading.Thread(target=self._reaper, name="maniple-unload", daemon=True).start()

    def ensure(self, model: str, model_dir: str) -> None:
        """Load `model`, or reload it if new version directories appeared since the last load."""
        versions = {v for v in os.listdir(model_dir) if v.isdigit()} if os.path.isdir(model_dir) else set()
        with self._lock:
            self._last_used[model] = time.time()
            if self._loaded_dirs.get(model) == versions and pb_utils.is_model_ready(model):
                return
            pb_utils.load_model(model)
            self._loaded_dirs[model] = versions

    def _reaper(self) -> None:
        while True:
            time.sleep(15)
            now = time.time()
            with self._lock:
                idle = [m for m, t in self._last_used.items() if now - t > self.idle_unload_sec]
                for model in idle:
                    with contextlib.suppress(Exception):  # best effort
                        pb_utils.unload_model(model)
                    self._last_used.pop(model, None)
                    self._loaded_dirs.pop(model, None)


class InferModel:
    train_model = "ppo_train"  # subclasses for other algorithms override

    def initialize(self, args):
        self.rng = np.random.default_rng()
        cfg = json.loads(args["model_config"])
        params = {k: v["string_value"] for k, v in cfg.get("parameters", {}).items()}
        self.model_repository = params.get("model_repository", "/models")
        self.train_model = params.get("train_model", self.train_model)
        self.manifests = _ManifestCache(self.model_repository)
        self.loader = _ModelLoader(float(params.get("idle_unload_sec", "600")))

    def execute(self, requests):
        return [self._handle(request) for request in requests]

    # ------------------------------------------------------------------ one request

    def _handle(self, request):
        name = _string_input(request, "name")
        explore = _bool_input(request, "explore", default=False)
        channel = _string_input(request, "channel") or "stable"
        obs = pb_utils.get_input_tensor_by_name(request, "obs").as_numpy().astype(np.float32)

        try:
            if channel == "latest":
                return self._from_trainer(name, obs, explore, channel)
            return self._from_export(name, obs, explore, channel)
        except Exception as e:  # noqa: BLE001 - always answer with a status
            n = obs.shape[0]
            return self._respond(
                np.zeros((n, 1), np.float32),
                np.full(n, -1, np.int64),
                np.zeros(n, np.float32),
                0,
                {"ok": False, "policy": name, "channel": channel, "error": f"{type(e).__name__}: {e}"},
            )

    def _from_trainer(self, name, obs, explore, channel):
        """'latest': ask the trainer for actions from its current weights."""
        inputs = [
            pb_utils.Tensor("name", np.array([name.encode()], dtype=np.object_)),
            pb_utils.Tensor("command", np.array([b"act"], dtype=np.object_)),
            pb_utils.Tensor("obs", obs),
            pb_utils.Tensor("explore", np.array([explore])),
        ]
        response = pb_utils.InferenceRequest(
            model_name=self.train_model,
            requested_output_names=["status", "action", "action_index", "logp", "policy_version"],
            inputs=inputs,
            preferred_memory=pb_utils.PreferredMemory(pb_utils.TRITONSERVER_MEMORY_CPU),
        ).exec()
        if response.has_error():
            raise RuntimeError(response.error().message())
        out = {t.name(): _to_numpy(t) for t in response.output_tensors()}
        status = json.loads(out["status"][0])
        if not status.get("ok"):
            raise RuntimeError(status.get("error", "trainer refused"))
        version = int(out["policy_version"].reshape(-1)[0])
        return self._respond(
            out["action"],
            out["action_index"],
            out["logp"],
            version,
            {"ok": True, "policy": name, "channel": channel, "served_by": self.train_model},
        )

    def _from_export(self, name, obs, explore, channel):
        """'best' / 'stable' / '<n>': resolve through the manifest, lazy-load, run the static model."""
        manifest = self.manifests.get(name)
        version = self._resolve(manifest, channel)
        if version is None:
            raise RuntimeError(
                f"no exported version for channel '{channel}' (train first, then export/report/promote)"
            )

        model = policy_model_name(name)
        trt = (manifest or {}).get("trt") or {}
        if channel == "stable" and trt.get("state") == "ready" and int(trt.get("version", -1)) == version:
            model = trt_model_name(name)

        self.loader.ensure(model, os.path.join(self.model_repository, model))
        outputs = self._call_policy(model, obs, version)

        served_version = (
            int(outputs["policy_version"].reshape(-1)[0]) if "policy_version" in outputs else version
        )
        if "log_std" in outputs:
            action, index, logp = self._continuous(outputs["action"], outputs["log_std"], explore)
        else:
            action, index, logp = self._discrete(outputs["action"], explore)
        status = {"ok": True, "policy": name, "channel": channel, "served_by": model, "explore": explore}
        return self._respond(action, index, logp, served_version, status)

    @staticmethod
    def _resolve(manifest, channel):
        if channel.isdigit():
            return int(channel)
        if not manifest:
            return None
        exported = manifest.get("exported") or []
        for key in {"stable": ("stable", "best"), "best": ("best",)}.get(channel, ()):
            v = manifest.get(key)
            if v is not None and v in exported:
                return int(v)
        return max(exported) if exported else None

    def _call_policy(self, model, obs, version):
        infer = pb_utils.InferenceRequest(
            model_name=model,
            model_version=int(version),
            requested_output_names=[],
            inputs=[pb_utils.Tensor("obs", obs)],
            preferred_memory=pb_utils.PreferredMemory(pb_utils.TRITONSERVER_MEMORY_CPU),
        )
        response = infer.exec()
        if response.has_error():
            raise RuntimeError(f"{model} v{version}: {response.error().message()}")
        return {t.name(): _to_numpy(t) for t in response.output_tensors()}

    # ------------------------------------------------------------------ action sampling

    def _continuous(self, mean, log_std, explore):
        std = np.exp(log_std)
        noise = self.rng.standard_normal(mean.shape).astype(np.float32) * std
        action = mean + noise if explore else mean
        logp = (-0.5 * ((action - mean) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(1)
        index = np.full(mean.shape[0], -1, np.int64)
        return action.astype(np.float32), index, logp.astype(np.float32)

    def _discrete(self, logits, explore):
        n = logits.shape[0]
        shifted = logits - logits.max(1, keepdims=True)
        probs = np.exp(shifted)
        probs /= probs.sum(1, keepdims=True)
        if explore:
            index = np.array([self.rng.choice(probs.shape[1], p=row) for row in probs], dtype=np.int64)
        else:
            index = probs.argmax(1).astype(np.int64)
        logp = np.log(probs[np.arange(n), index] + 1e-12)
        action = np.zeros_like(logits, dtype=np.float32)
        action[np.arange(n), index] = 1.0
        return action, index, logp.astype(np.float32)

    # ------------------------------------------------------------------ response

    @staticmethod
    def _respond(action, index, logp, version, status):
        return pb_utils.InferenceResponse(
            output_tensors=[
                pb_utils.Tensor("action", action),
                pb_utils.Tensor("action_index", index),
                pb_utils.Tensor("logp", logp),
                pb_utils.Tensor("policy_version", np.array([version], dtype=np.int64)),
                pb_utils.Tensor("status", np.array([json.dumps(status).encode()], dtype=np.object_)),
            ]
        )
