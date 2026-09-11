"""Unit tests for the generic pipe: spec parsing, networks, buffer, PPO update, ONNX export, v1 migration.

No Triton needed (nothing from triton_python_backend_utils is imported). Run inside the server image, where
torch and onnx are installed:

    docker compose exec triton python -m unittest discover -s /common/tests -v
"""

import json
import os
import tempfile
import unittest

import numpy as np
import torch

from maniple.buffer import TransitionBuffer
from maniple.nets import Actor, ActorCritic, migrate_v1_state_dict
from maniple.spec import AgentSpec

V1 = {"obs": {"dim": 32}, "action": {"type": "continuous", "dim": 5}, "net": {"preset": "medium"}}

V2 = {
    "inputs": {
        "obs": {"shape": [32]},
        "frame": {"shape": [84, 84, 4], "dtype": "uint8", "encoder": "cnn"},
    },
    "actions": {
        "move": {"type": "continuous", "dim": 4},
        "fire": {"type": "discrete", "n": 2},
        "mode": {"type": "discrete", "n": 3},
    },
    "net": {"preset": "small"},
    "ppo": {"rollout": 64, "minibatch": 32, "epochs": 1},
}

DISCRETE = {"obs": {"dim": 4}, "action": {"type": "discrete", "n": 2}, "net": {"preset": "small"}}


def spec(data):
    return AgentSpec.from_json(json.dumps(data))


def random_inputs(s, n):
    out = {}
    for i in s.inputs:
        if i.dtype == "uint8":
            out[i.name] = np.random.randint(0, 256, size=(n, *i.shape), dtype=np.uint8)
        else:
            out[i.name] = np.random.randn(n, *i.shape).astype(np.float32)
    return out


class SpecTest(unittest.TestCase):
    def test_v1_is_one_input_one_group(self):
        s = spec(V1)
        self.assertTrue(s.is_v1)
        self.assertEqual(s.input_names, ["obs"])
        self.assertEqual(s.obs_dim, 32)
        self.assertEqual([g.name for g in s.actions], ["action"])
        self.assertEqual(s.action_dim, 5)
        self.assertEqual(s.continuous_dim, 5)
        self.assertEqual(s.hidden, [256, 256])

    def test_v1_and_v2_forms_are_the_same_spec(self):
        v2 = {
            "inputs": {"obs": {"shape": [32]}},
            "actions": {"action": {"type": "continuous", "dim": 5}},
            "net": {"preset": "medium"},
        }
        self.assertEqual(spec(V1).to_json(), spec(v2).to_json())
        self.assertEqual(spec(V1).to_json(), spec(json.loads(spec(V1).to_json())).to_json())  # round trip

    def test_v2_layout(self):
        s = spec(V2)
        self.assertFalse(s.is_v1)
        self.assertEqual(s.input_names, ["obs", "frame"])
        self.assertEqual(s.input("frame").hidden, [512])  # cnn default fc
        self.assertEqual(s.feature_dim, 32 + 512)
        self.assertEqual(s.action_dim, 4 + 2 + 3)
        self.assertEqual(s.continuous_dim, 4)
        self.assertEqual([g.name for g in s.discrete_groups], ["fire", "mode"])
        self.assertEqual(s.action_slices["fire"], slice(4, 6))

    def test_actions_as_list(self):
        data = dict(V2, actions=[{"name": "fire", "type": "discrete", "n": 2}])
        self.assertEqual([g.name for g in spec(data).actions], ["fire"])

    def test_none_preset_has_no_torso(self):
        self.assertEqual(spec(dict(V1, net={"preset": "none"})).hidden, [])

    def test_errors(self):
        bad = [
            dict(V2, inputs={"text": {"shape": []}}),  # reserved, no encoder yet
            dict(V2, inputs={"depth": {"shape": [8]}}),  # unknown wire name
            dict(V2, inputs={"obs": {"shape": [8], "encoder": "cnn"}}),  # cnn needs a frame
            dict(V2, inputs={"frame": {"shape": [84, 84], "dtype": "uint8"}}),  # wrong rank
            dict(V2, actions={"fire": {"type": "discrete"}}),  # n missing
            dict(V2, actions={}),
            {"action": {"type": "discrete", "n": 2}},  # no inputs at all
        ]
        for data in bad:
            with self.assertRaises((ValueError, TypeError), msg=json.dumps(data)):
                spec(data)


class NetTest(unittest.TestCase):
    def test_v1_actor_forward_and_distribution(self):
        s = spec(V1)
        actor = Actor(s)
        x = torch.randn(3, 32)
        row, log_std = actor(x)
        self.assertEqual(tuple(row.shape), (3, 5))
        self.assertEqual(tuple(log_std.shape), (3, 5))
        d = actor.distribution({"obs": x})
        a = d.sample()
        self.assertEqual(tuple(a.shape), (3, 5))
        self.assertEqual(tuple(d.log_prob(a).shape), (3,))
        self.assertEqual(tuple(d.indices(a).shape), (3, 0))

    def test_v2_actor_mixed_groups(self):
        s = spec(V2)
        actor = Actor(s)
        inputs = {k: torch.as_tensor(v) for k, v in random_inputs(s, 3).items()}
        row, log_std = actor(inputs["obs"], inputs["frame"])
        self.assertEqual(tuple(row.shape), (3, 9))
        self.assertEqual(tuple(log_std.shape), (3, 4))
        d = actor.distribution(inputs)
        a = d.sample()
        self.assertEqual(tuple(a.shape), (3, 9))
        self.assertTrue(torch.all(a[:, 4:6].sum(-1) == 1))  # one-hot per discrete group
        self.assertTrue(torch.all(a[:, 6:9].sum(-1) == 1))
        idx = d.indices(a)
        self.assertEqual(tuple(idx.shape), (3, 2))
        self.assertTrue(torch.all(idx[:, 0] == a[:, 4:6].argmax(-1)))
        self.assertEqual(tuple(d.log_prob(a).shape), (3,))
        self.assertEqual(tuple(d.entropy().shape), (3,))
        mode = d.mode()
        self.assertEqual(tuple(mode.shape), (3, 9))

    def test_discrete_only_actor_has_no_log_std(self):
        actor = Actor(spec(DISCRETE))
        out = actor(torch.randn(2, 4))
        self.assertFalse(isinstance(out, tuple))
        self.assertEqual(tuple(out.shape), (2, 2))

    def test_critic_variants(self):
        for separate in (True, False):
            s = spec(dict(V2, net={"preset": "small", "separate_critic": separate}))
            model = ActorCritic(s)
            inputs = {k: torch.as_tensor(v) for k, v in random_inputs(s, 2).items()}
            self.assertEqual(tuple(model.value(inputs).shape), (2,))

    def test_v1_checkpoint_migrates(self):
        s = spec(V1)
        fresh = ActorCritic(s)
        # rebuild the pre-encoder key layout from a current state dict
        old = {}
        for key, value in fresh.state_dict().items():
            if key.startswith("actor.encoders.obs.normalizer."):
                old["actor.normalizer." + key.split(".", 4)[4]] = value
            elif key.startswith("actor.torso."):
                old["actor.net." + key[len("actor.torso.") :]] = value
            elif key.startswith("actor.heads.action.mean."):
                last = len(fresh.actor.torso)
                old[f"actor.net.{last}." + key[len("actor.heads.action.mean.") :]] = value
            elif key == "actor.heads.action.log_std":
                old["actor.log_std"] = value
            else:
                old[key] = value
        self.assertIn("actor.net.0.weight", old)
        migrated = migrate_v1_state_dict(old)
        loaded = ActorCritic(s)
        missing, unexpected = loaded.load_state_dict(migrated, strict=False)
        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [k for k in migrated if ".logits." in k])
        x = torch.randn(2, 32)
        self.assertTrue(torch.allclose(fresh.actor(x)[0], loaded.actor(x)[0]))
        self.assertTrue(torch.allclose(fresh.value({"obs": x}), loaded.value({"obs": x})))

    def test_v1_optimizer_state_migrates(self):
        """Adam moments are stored by position; the old layout had actor.log_std first. After migration every
        moment must sit on the parameter of the same name."""
        from maniple.algorithms.ppo import PPO

        s = spec(dict(V1, ppo={"rollout": 64, "minibatch": 32, "epochs": 1}))
        ppo = PPO(s, "cpu")
        buffer = TransitionBuffer(s)
        BufferAndPPOTest.fill(None, s, buffer, agents=4, steps=16)
        ppo.update(buffer.take())
        new_state = ppo.state_dict()

        # rebuild the old checkpoint: old key names, optimizer entries in the old parameter order
        names = [n for n, _ in ppo.model.named_parameters()]
        old_of = {"actor.heads.action.log_std": "actor.log_std"}
        last = len(ppo.model.actor.torso)
        for n in names:
            if n.startswith("actor.torso."):
                old_of[n] = "actor.net." + n[len("actor.torso.") :]
            elif n.startswith("actor.heads.action.mean."):
                old_of[n] = f"actor.net.{last}." + n[len("actor.heads.action.mean.") :]
            elif n not in old_of:
                old_of[n] = n
        old_order = ["actor.log_std"] + [old_of[n] for n in names if n != "actor.heads.action.log_std"]
        old_model = {}
        for k, v in new_state["model"].items():
            if k.startswith("actor.encoders.obs.normalizer."):
                old_model["actor.normalizer." + k.split(".", 4)[4]] = v
            else:
                old_model[old_of.get(k, k)] = v
        # state_dict order matters for the migration: params of the actor first, then buffers, then net, critic
        ordered = {"actor.log_std": old_model["actor.log_std"]}
        ordered.update({k: v for k, v in old_model.items() if k.startswith("actor.normalizer.")})
        ordered.update({k: v for k, v in old_model.items() if k not in ordered})
        new_index = {n: i for i, n in enumerate(names)}
        old_opt_state = {
            i: new_state["optimizer"]["state"][new_index[[n for n in names if old_of[n] == old_name][0]]]
            for i, old_name in enumerate(old_order)
        }
        old_opt = {
            "state": old_opt_state,
            "param_groups": [dict(new_state["optimizer"]["param_groups"][0], params=list(range(len(names))))],
        }

        loaded = PPO(s, "cpu")
        loaded.load_state_dict({"model": ordered, "optimizer": old_opt})
        for (name, param), (_, ref) in zip(
            loaded.model.named_parameters(), ppo.model.named_parameters(), strict=True
        ):
            st, ref_st = loaded.optimizer.state[param], ppo.optimizer.state[ref]
            self.assertEqual(tuple(st["exp_avg"].shape), tuple(param.shape), name)
            self.assertTrue(torch.equal(st["exp_avg"], ref_st["exp_avg"]), name)


class BufferAndPPOTest(unittest.TestCase):
    def fill(self, s, buffer, agents, steps, version=1):
        n_act = s.action_dim
        for t in range(steps):
            inputs = random_inputs(s, agents)
            action = np.zeros((agents, n_act), np.float32)
            action[:, : s.continuous_dim] = np.random.randn(agents, s.continuous_dim)
            for g in s.discrete_groups:
                sl = s.action_slices[g.name]
                action[np.arange(agents), sl.start + np.random.randint(0, g.n, agents)] = 1.0
            buffer.add(
                inputs,
                action,
                np.random.randn(agents).astype(np.float32),
                np.array([t == steps - 1] * agents),
                np.arange(agents),
                np.zeros(agents, np.int64),
                np.full(agents, version),
                np.full(agents, np.nan, np.float32),
                current_version=version,
                max_lag=4,
            )

    def test_buffer_keeps_inputs_per_trajectory(self):
        s = spec(V2)
        buffer = TransitionBuffer(s)
        self.fill(s, buffer, agents=2, steps=5)
        self.assertEqual(len(buffer), 10)
        trajectories = buffer.take()
        self.assertEqual(len(trajectories), 2)
        t = trajectories[0]
        self.assertEqual(t["inputs"]["frame"].shape, (5, 84, 84, 4))
        self.assertEqual(t["inputs"]["frame"].dtype, np.uint8)
        self.assertEqual(t["inputs"]["obs"].shape, (5, 32))
        self.assertEqual(t["action"].shape, (5, 9))
        self.assertTrue(t["done"][-1])
        self.assertEqual(len(buffer), 0)

    def test_buffer_rejects_wrong_shapes(self):
        s = spec(V1)
        buffer = TransitionBuffer(s)
        with self.assertRaises(ValueError):
            buffer.add(
                {"obs": np.zeros((2, 31), np.float32)},
                np.zeros((2, 5), np.float32),
                np.zeros(2),
                np.zeros(2, bool),
                None,
                None,
                None,
                None,
                current_version=1,
                max_lag=4,
            )

    def test_ppo_update_runs(self):
        from maniple.algorithms.ppo import PPO

        for data in (V1, V2, DISCRETE):
            s = spec(dict(data, ppo={"rollout": 64, "minibatch": 32, "epochs": 1}))
            ppo = PPO(s, "cpu")
            buffer = TransitionBuffer(s)
            self.fill(s, buffer, agents=4, steps=16)
            stats = ppo.update(buffer.take())
            self.assertEqual(stats["samples"], 64)
            for key in ("loss_pi", "loss_v", "kl"):
                self.assertTrue(np.isfinite(stats[key]), key)
            if s.continuous_dim:
                self.assertIn("std", stats)
            # a second copy loads the checkpoint
            other = PPO(s, "cpu")
            other.load_state_dict(ppo.state_dict())


class ExportTest(unittest.TestCase):
    def test_onnx_export_and_layout_files(self):
        try:
            import onnxruntime as ort
        except ImportError:
            self.skipTest("onnxruntime not installed (docker compose build picks it up)")

        from maniple.export import export_actor, read_spec

        for data in (V1, V2, DISCRETE):
            s = spec(data)
            actor = Actor(s).eval()
            with tempfile.TemporaryDirectory() as repo:
                export_actor(actor, s, "t", 7, repo)
                model_dir = os.path.join(repo, "policy_t")
                self.assertTrue(os.path.exists(os.path.join(model_dir, "7", "model.onnx")))
                self.assertEqual(read_spec(model_dir).to_json(), s.to_json())
                with open(os.path.join(model_dir, "config.pbtxt")) as f:
                    config = f.read()
                for i in s.inputs:
                    self.assertIn(f'name: "{i.name}"', config)
                self.assertIn(f'name: "action", data_type: TYPE_FP32, dims: [ {s.action_dim} ]', config)
                self.assertEqual('name: "log_std"' in config, s.continuous_dim > 0)

                session = ort.InferenceSession(os.path.join(model_dir, "7", "model.onnx"))
                feeds = random_inputs(s, 3)
                outputs = {
                    o.name: v for o, v in zip(session.get_outputs(), session.run(None, feeds), strict=True)
                }
                self.assertEqual(outputs["action"].shape, (3, s.action_dim))
                self.assertEqual(int(outputs["policy_version"][0]), 7)
                if s.continuous_dim:
                    self.assertEqual(outputs["log_std"].shape, (3, s.continuous_dim))
                with torch.no_grad():
                    ref = actor(*[torch.as_tensor(feeds[n]) for n in s.input_names])
                ref = ref[0] if isinstance(ref, tuple) else ref
                self.assertTrue(np.allclose(ref.numpy(), outputs["action"], atol=1e-4))


if __name__ == "__main__":
    unittest.main()
