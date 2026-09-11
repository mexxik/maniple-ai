"""AlgorithmModel: base class for '<algo>_train/<version>/model.py' Triton Python-backend models.

A concrete model is two lines:
    from maniple.triton_model import AlgorithmModel
    from maniple.algorithms.ppo import PPO
    class TritonPythonModel(AlgorithmModel):
        algorithm = PPO

Protocol (every request addresses ONE named policy; rows in the tensors are agent steps; INPUTS = the
tensors the policy's spec declares: obs and/or frame, audio, text):
  command=register  spec=<json>                              create the policy (idempotent)
  command=act       INPUTS [explore]                         actions from the CURRENT weights ('latest' channel)
  command=observe   INPUTS action reward done [agent_id episode_id policy_version logp]   feed transitions
  command=status
  command=export                                             write the current version as a static model
  command=report    version score [episodes]                 a client's greedy evaluation of a version (drives 'best')
  command=promote   version                                  pin 'stable' (exports if needed, builds TensorRT if enabled)
Outputs: status (JSON) always; action [N, action_dim] / action_index [N, discrete groups] / logp [N] /
policy_version filled for command=act.
"""

from __future__ import annotations

import json

import numpy as np
import triton_python_backend_utils as pb_utils

from .algorithms.base import Algorithm
from .registry import PolicyRegistry
from .spec import AgentSpec
from .wire import array_input as _arr
from .wire import read_inputs
from .wire import scalar_input as _scalar
from .wire import string_input as _str


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
            device="cuda" if args.get("model_instance_kind") == "GPU" else "cpu",
        )
        self.registry.load_all()

    def execute(self, requests):
        responses = []
        for req in requests:
            try:
                responses.append(self._handle(req))
            except Exception as e:  # noqa: BLE001 - a bad row must never take the server down
                responses.append(self._respond({"ok": False, "error": f"{type(e).__name__}: {e}"}))
        return responses

    def finalize(self):
        self.registry.shutdown()

    # ------------------------------------------------------------------ protocol

    def _handle(self, req):
        name = _str(req, "name")
        command = _str(req, "command") or "observe"

        if command == "register":
            policy = self.registry.get_or_create(name, AgentSpec.from_json(_str(req, "spec")))
            return self._respond({"ok": True, **policy.status()})

        policy = self.registry.get(name)
        if policy is None:
            return self._respond(
                {"ok": False, "error": f"unknown policy '{name}': send command=register with spec first"}
            )

        if command == "act":
            explore = bool(_scalar(req, "explore", False))
            action, index, logp = policy.act(read_inputs(req), explore)
            status = {"ok": True, "policy": name, "channel": "latest", "explore": explore}
            return self._respond(status, action=action, index=index, logp=logp, version=policy.version)

        if command == "observe":
            n = policy.observe(
                inputs=read_inputs(req),
                action=_arr(req, "action"),
                reward=_arr(req, "reward"),
                done=_arr(req, "done"),
                agent_id=_arr(req, "agent_id"),
                episode_id=_arr(req, "episode_id"),
                policy_version=_arr(req, "policy_version"),
                logp=_arr(req, "logp"),
            )
            return self._respond({"ok": True, "accepted": int(n), **policy.status()})

        if command == "status":
            return self._respond({"ok": True, **policy.status()})

        if command == "export":
            return self._respond({"ok": True, "exported_version": policy.export(), **policy.status()})

        if command == "report":
            policy.report(
                int(_scalar(req, "version")), float(_scalar(req, "score")), int(_scalar(req, "episodes", 1))
            )
            return self._respond({"ok": True, **policy.status()})

        if command == "promote":
            policy.promote(int(_scalar(req, "version")))
            return self._respond({"ok": True, **policy.status()})

        return self._respond({"ok": False, "error": f"unknown command '{command}'"})

    @staticmethod
    def _respond(status, action=None, index=None, logp=None, version=0):
        n = 0 if action is None else action.shape[0]
        return pb_utils.InferenceResponse(
            output_tensors=[
                pb_utils.Tensor("status", np.array([json.dumps(status).encode()], dtype=np.object_)),
                pb_utils.Tensor("action", action if action is not None else np.zeros((0, 1), np.float32)),
                pb_utils.Tensor("action_index", index if index is not None else np.zeros((n, 0), np.int64)),
                pb_utils.Tensor("logp", logp if logp is not None else np.zeros((n,), np.float32)),
                pb_utils.Tensor("policy_version", np.array([version], dtype=np.int64)),
            ]
        )
