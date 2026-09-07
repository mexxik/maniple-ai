"""InferModel: base class for '<algo>_infer/<version>/model.py' — the stable inference entry point.

    from maniple.infer_model import InferModel
    class TritonPythonModel(InferModel): pass

Inputs : name (STRING [1]), obs (FP32 [N, obs_dim]), explore (BOOL [1], optional, default false),
         channel (STRING [1], optional): best (default) | latest | stable | "<version>" — see versions.py
Outputs: action         FP32 [N, act_dim]  continuous action, or one-hot of the chosen discrete action
         action_index   INT64 [N]          chosen index for discrete, -1 for continuous
         logp           FP32 [N]           log-prob of the returned action under the served policy;
                                           send it back with observe() so the trainer can correct policy lag
         policy_version INT64 [1]          version baked into the served model
         status         STRING [1]         JSON {ok, policy, explore | error}

Forwards to the exported static model '<name>_policy' (ONNX/TRT, dynamic batching) via BLS. With explore=true
it samples: Normal(mean, exp(log_std)) for continuous, Categorical(logits) for discrete — exploration lives
server-side so the game client stays generic. If the policy model is not ready yet, returns zeros + status.
"""

from __future__ import annotations

import json
import os

import numpy as np
import triton_python_backend_utils as pb_utils

from .export import policy_model_name
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

    def resolve(self, model: str, channel: str) -> int:
        """Concrete version for a channel; -1 = let Triton pick (newest loaded)."""
        if channel.isdigit():
            return int(channel)
        manifest = self._get(model)
        if manifest is None:
            return -1
        version = manifest.get(channel)
        if channel == "best" and version is None:
            version = manifest.get("latest")
        if channel == "stable" and version is None:
            version = manifest.get("best") or manifest.get("latest")
        return int(version) if version else -1

    def _get(self, model: str) -> dict | None:
        model_dir = os.path.join(self.model_repository, model)
        path = os.path.join(model_dir, "versions.json")
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return None
        cached = self._cache.get(model)
        if cached is None or cached[0] != mtime:
            self._cache[model] = (mtime, read_manifest(model_dir))
        return self._cache[model][1]


class InferModel:
    def initialize(self, args):
        self.rng = np.random.default_rng()
        cfg = json.loads(args["model_config"])
        params = {k: v["string_value"] for k, v in cfg.get("parameters", {}).items()}
        self.manifests = _ManifestCache(params.get("model_repository", "/models"))

    def execute(self, requests):
        return [self._handle(request) for request in requests]

    # ------------------------------------------------------------------ one request

    def _handle(self, request):
        name = _string_input(request, "name")
        explore = _bool_input(request, "explore", default=False)
        channel = _string_input(request, "channel") or "best"
        obs = pb_utils.get_input_tensor_by_name(request, "obs").as_numpy().astype(np.float32)
        n = obs.shape[0]
        model = policy_model_name(name)
        version = self.manifests.resolve(model, channel)

        outputs = self._call_policy(model, obs, version)
        if outputs is None and version > 0:
            # a just-exported (or just-pruned) version may not be loaded yet: fall back to the newest loaded one
            outputs = self._call_policy(model, obs, -1)
        if outputs is None:
            return self._respond(
                action=np.zeros((n, 1), np.float32),
                index=np.full(n, -1, np.int64),
                logp=np.zeros(n, np.float32),
                version=0,
                status={
                    "ok": False,
                    "policy": model,
                    "error": f"{model} version {version if version > 0 else 'latest'} is not ready",
                },
            )

        version = (
            int(outputs["policy_version"].reshape(-1)[0]) if "policy_version" in outputs else 0
        )  # from the model itself

        if "log_std" in outputs:
            action, index, logp = self._continuous(outputs["action"], outputs["log_std"], explore)
        else:
            action, index, logp = self._discrete(outputs["action"], explore)

        status = {"ok": True, "policy": model, "explore": explore, "channel": channel}
        return self._respond(action, index, logp, version, status)

    def _call_policy(self, model, obs, version=-1):
        """Run the exported policy model through BLS. Returns {output name: numpy} or None on error."""
        infer = pb_utils.InferenceRequest(
            model_name=model,
            model_version=version,
            requested_output_names=[],
            inputs=[pb_utils.Tensor("obs", obs)],
            preferred_memory=pb_utils.PreferredMemory(pb_utils.TRITONSERVER_MEMORY_CPU),
        )
        response = infer.exec()
        if response.has_error():
            return None
        return {t.name(): _to_numpy(t) for t in response.output_tensors()}

    # ------------------------------------------------------------------ action sampling

    def _continuous(self, mean, log_std, explore):
        """Normal(mean, std). Returns (action, index=-1, logp)."""
        std = np.exp(log_std)
        noise = self.rng.standard_normal(mean.shape).astype(np.float32) * std
        action = mean + noise if explore else mean

        logp = (-0.5 * ((action - mean) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(1)
        index = np.full(mean.shape[0], -1, np.int64)
        return action.astype(np.float32), index, logp.astype(np.float32)

    def _discrete(self, logits, explore):
        """Categorical(logits). Returns (one-hot action, index, logp)."""
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
