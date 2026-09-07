"""AlgorithmModel: base class for '<algo>_train/<version>/model.py' Triton Python-backend models.

A concrete model is two lines:
    from maniple.triton_model import AlgorithmModel
    from maniple.algorithms.ppo import PPO
    class TritonPythonModel(AlgorithmModel):
        algorithm = PPO

Protocol (every request addresses ONE named policy; rows in the tensors are agent steps):
  command=register  spec=<json>     create the policy (idempotent) and export version 1
  command=observe   obs action reward done [agent_id episode_id policy_version logp]   feed transitions
  command=status
  command=export                     force an export now
  command=report    version score [episodes]   a client's greedy evaluation of a version (drives 'best')
  command=promote   version                    pin the 'stable' channel to a version
Output: 'status' = JSON {ok, error?, name, version, updates, buffered, total_samples, ...}
"""

from __future__ import annotations

import json

import numpy as np
import triton_python_backend_utils as pb_utils

from .algorithms.base import Algorithm
from .registry import PolicyRegistry
from .spec import AgentSpec


def _str(request, name):
    t = pb_utils.get_input_tensor_by_name(request, name)
    if t is None:
        return None
    v = t.as_numpy().reshape(-1)[0]
    return v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v)


def _arr(request, name):
    t = pb_utils.get_input_tensor_by_name(request, name)
    return None if t is None else t.as_numpy()


class AlgorithmModel:
    algorithm: type[Algorithm] = None  # set by the subclass

    def initialize(self, args):
        if self.algorithm is None:
            raise RuntimeError("AlgorithmModel subclass must set `algorithm`")
        cfg = json.loads(args["model_config"])
        p = {k: v["string_value"] for k, v in cfg.get("parameters", {}).items()}
        self.registry = PolicyRegistry(
            algorithm_cls=self.algorithm,
            model_repository=p.get("model_repository", "/models"),
            checkpoint_dir=p.get("checkpoint_dir", "/checkpoints"),
            export_every_updates=int(p.get("export_every_updates", "1")),
            device="cuda" if args.get("model_instance_kind") == "GPU" else "cpu",
        )
        self.registry.load_all()

    def execute(self, requests):
        responses = []
        for req in requests:
            try:
                responses.append(self._handle(req))
            except Exception as e:  # a bad row must never take the server down
                responses.append(self._status({"ok": False, "error": f"{type(e).__name__}: {e}"}))
        return responses

    def finalize(self):
        self.registry.shutdown()

    # ---- protocol
    def _handle(self, req):
        name = _str(req, "name")
        command = _str(req, "command") or "observe"
        if command == "register":
            policy = self.registry.get_or_create(name, AgentSpec.from_json(_str(req, "spec")))
            return self._status({"ok": True, **policy.status()})
        policy = self.registry.get(name)
        if policy is None:
            return self._status(
                {"ok": False, "error": f"unknown policy '{name}': send command=register with spec first"}
            )
        if command == "observe":
            n = policy.observe(
                obs=_arr(req, "obs"),
                action=_arr(req, "action"),
                reward=_arr(req, "reward"),
                done=_arr(req, "done"),
                agent_id=_arr(req, "agent_id"),
                episode_id=_arr(req, "episode_id"),
                policy_version=_arr(req, "policy_version"),
                logp=_arr(req, "logp"),
            )
            return self._status({"ok": True, "accepted": int(n), **policy.status()})
        if command == "status":
            return self._status({"ok": True, **policy.status()})
        if command == "export":
            return self._status({"ok": True, "exported_version": policy.export(), **policy.status()})
        if command == "report":
            version = int(_arr(req, "version").reshape(-1)[0])
            score = float(_arr(req, "score").reshape(-1)[0])
            episodes_t = _arr(req, "episodes")
            episodes = int(episodes_t.reshape(-1)[0]) if episodes_t is not None else 1
            policy.report(version, score, episodes)
            return self._status({"ok": True, **policy.status()})
        if command == "promote":
            version = int(_arr(req, "version").reshape(-1)[0])
            policy.promote(version)
            return self._status({"ok": True, **policy.status()})
        return self._status({"ok": False, "error": f"unknown command '{command}'"})

    @staticmethod
    def _status(d):
        out = pb_utils.Tensor("status", np.array([json.dumps(d).encode()], dtype=np.object_))
        return pb_utils.InferenceResponse(output_tensors=[out])
