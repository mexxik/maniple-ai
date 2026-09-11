"""Small client for the Triton training protocol, shared by run.py and play.py.

Two server models are involved:
  ppo_train  the algorithm: register a named policy, feed it transitions (observe), ask for status
  ppo_infer  the inference entry point: inputs -> action for a named policy, with optional exploration

Observations are a dict of named inputs, {"obs": [N, D] float, "frame": [N, H, W, C] uint8, ...}, matching the
inputs the policy was registered with; a bare array means {"obs": array}. Actions come back as one flat row per
agent over all action groups (continuous values, one-hot per discrete group) plus the chosen index per discrete
group. Everything is plain numpy in and out; the client has no idea what network is behind the name.
"""

import json

import numpy as np
import tritonclient.grpc as grpc
from tritonclient.utils import np_to_triton_dtype

# wire dtype per observation input (see triton/common/maniple/spec.py)
INPUT_DTYPES = {"obs": np.float32, "frame": np.uint8, "audio": np.float32, "text": np.object_}


def make_input(name, array):
    """Wrap a numpy array as a Triton input tensor."""
    tensor = grpc.InferInput(name, array.shape, np_to_triton_dtype(array.dtype))
    tensor.set_data_from_numpy(array)
    return tensor


def make_string(text):
    """Triton STRING tensors are numpy object arrays of bytes."""
    return np.array([text.encode()], dtype=np.object_)


def as_inputs(inputs):
    """{"obs": array, ...} or a bare array (= obs), cast to the wire dtypes."""
    if not isinstance(inputs, dict):
        inputs = {"obs": inputs}
    out = {}
    for name, array in inputs.items():
        if name not in INPUT_DTYPES:
            raise ValueError(f"unknown input '{name}' ({', '.join(INPUT_DTYPES)})")
        if name == "text":
            out[name] = np.array([str(t).encode() for t in array], dtype=np.object_)
        else:
            out[name] = np.asarray(array).astype(INPUT_DTYPES[name])
    return out


class TritonAgent:
    """One named policy on one Triton server."""

    def __init__(self, url, name):
        self.client = grpc.InferenceServerClient(url)
        self.name = name

    # ------------------------------------------------------------------ training side

    def register(self, spec):
        """Create the policy from an AgentSpec dict (idempotent). Returns the trainer status."""
        return self._train_call("register", spec=make_string(json.dumps(spec)))

    def observe(self, inputs, action, reward, done, agent_id, episode_id, policy_version, logp):
        """Send one batch of transitions (one row per agent). Returns the trainer status."""
        return self._train_call(
            "observe",
            **as_inputs(inputs),
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

    def act(self, inputs, explore=True, channel="best"):
        """inputs -> (action [N, action_dim], action_index [N, G], logp [N], served policy version).

        channel: best (default) | latest | stable | "<version>". Training actors must use latest.

        action is the flat row over all action groups: continuous values, one-hot per discrete group.
        action_index is the chosen index per discrete group (G columns, 0 for a continuous-only policy).
        logp is the log-probability of the returned row under the served policy; send it back with observe()
        so the trainer can correct for policy lag.
        """
        tensors = [
            make_input("name", make_string(self.name)),
            make_input("explore", np.array([explore])),
            make_input("channel", make_string(str(channel))),
        ]
        tensors += [make_input(key, value) for key, value in as_inputs(inputs).items()]
        wanted = ("action", "action_index", "logp", "policy_version", "status")
        response = self.client.infer(
            "ppo_infer", tensors, outputs=[grpc.InferRequestedOutput(o) for o in wanted]
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
