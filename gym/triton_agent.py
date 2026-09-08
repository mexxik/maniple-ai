"""Small client for the Triton training protocol, shared by run.py and play.py.

Two server models are involved:
  ppo_train  the algorithm: register a named policy, feed it transitions (observe), ask for status
  ppo_infer  the inference entry point: obs -> action for a named policy, with optional exploration

Everything is plain numpy in and out; the client has no idea what network is behind the name.
"""

import json

import numpy as np
import tritonclient.grpc as grpc
from tritonclient.utils import np_to_triton_dtype


def make_input(name, array):
    """Wrap a numpy array as a Triton input tensor."""
    tensor = grpc.InferInput(name, array.shape, np_to_triton_dtype(array.dtype))
    tensor.set_data_from_numpy(array)
    return tensor


def make_string(text):
    """Triton STRING tensors are numpy object arrays of bytes."""
    return np.array([text.encode()], dtype=np.object_)


class TritonAgent:
    """One named policy on one Triton server."""

    def __init__(self, url, name):
        self.client = grpc.InferenceServerClient(url)
        self.name = name

    # ------------------------------------------------------------------ training side

    def register(self, spec):
        """Create the policy from an AgentSpec dict (idempotent). Returns the trainer status."""
        return self._train_call("register", spec=make_string(json.dumps(spec)))

    def observe(self, obs, action, reward, done, agent_id, episode_id, policy_version, logp):
        """Send one batch of transitions (one row per agent). Returns the trainer status."""
        return self._train_call(
            "observe",
            obs=obs.astype(np.float32),
            action=action.astype(np.float32),
            reward=reward.astype(np.float32),
            done=done.astype(bool),
            agent_id=agent_id.astype(np.int64),
            episode_id=episode_id.astype(np.int64),
            policy_version=policy_version.astype(np.int64),
            logp=logp.astype(np.float32),
        )

    def status(self):
        return self._train_call("status")

    def report(self, version, score, episodes=1):
        """Tell the trainer how a version scored in greedy evaluation (drives the 'best' channel)."""
        return self._train_call(
            "report",
            version=np.array([version], dtype=np.int64),
            score=np.array([score], dtype=np.float32),
            episodes=np.array([episodes], dtype=np.int64),
        )

    def export(self):
        """Export the trainer's current version as a static model."""
        return self._train_call("export")

    def promote(self, version):
        """Pin the 'stable' channel to a version."""
        return self._train_call("promote", version=np.array([version], dtype=np.int64))

    def stored_spec(self):
        """The spec an existing policy was created with, or None if the name is unknown."""
        try:
            return self.status()["spec"]
        except RuntimeError:
            return None

    def wait_until_ready(self, timeout_sec=60):
        """The exported policy model appears a moment after register(); block until Triton serves it."""
        import time

        model = f"policy_{self.name}"
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self.client.is_model_ready(model):
                return
            time.sleep(0.5)
        raise TimeoutError(f"{model} did not become ready within {timeout_sec}s")

    def _train_call(self, command, **tensors):
        inputs = [make_input("name", make_string(self.name)), make_input("command", make_string(command))]
        for key, value in tensors.items():
            inputs.append(make_input(key, value))

        response = self.client.infer("ppo_train", inputs, outputs=[grpc.InferRequestedOutput("status")])
        status = json.loads(response.as_numpy("status")[0])
        if not status["ok"]:
            raise RuntimeError(f"ppo_train {command}: {status['error']}")
        return status

    # ------------------------------------------------------------------ inference side

    def act(self, obs, explore=True, channel="best"):
        """obs [N, obs_dim] -> (action [N, act_dim], action_index [N], logp [N], served policy version).

        channel: best (default) | latest | stable | "<version>". Training actors must use latest.

        action is the continuous action, or a one-hot of the chosen discrete action.
        action_index is the discrete index (-1 for continuous).
        logp is the log-probability of the returned action under the served policy; send it back with observe()
        so the trainer can correct for policy lag.
        """
        inputs = [
            make_input("name", make_string(self.name)),
            make_input("obs", obs.astype(np.float32)),
            make_input("explore", np.array([explore])),
            make_input("channel", make_string(str(channel))),
        ]
        wanted = ("action", "action_index", "logp", "policy_version", "status")
        response = self.client.infer(
            "ppo_infer", inputs, outputs=[grpc.InferRequestedOutput(o) for o in wanted]
        )

        status = json.loads(response.as_numpy("status")[0])
        if not status["ok"]:
            raise RuntimeError(f"ppo_infer: {status['error']}")

        return (
            response.as_numpy("action"),
            response.as_numpy("action_index"),
            response.as_numpy("logp"),
            int(response.as_numpy("policy_version")[0]),
        )
