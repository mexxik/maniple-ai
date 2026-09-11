"""Recording writer / reader: round trip, header patching, crash recovery, episode grouping. numpy only."""

import json
import os
import struct
import tempfile
import unittest

import numpy as np

from maniple.recording import HEADER_SIZE, Recording, RecordingWriter, npy_header, read_npy


def rows(n, start=0):
    return {
        "obs": np.arange(start * 4, (start + n) * 4, dtype=np.float32).reshape(n, 4),
        "frame": np.full((n, 2, 2, 1), start, dtype=np.uint8),
        "action": np.ones((n, 2), dtype=np.float32),
        "reward": np.arange(n, dtype=np.float32),
        "done": np.zeros(n, dtype=bool),
        "agent_id": np.arange(n, dtype=np.int64) % 2,
        "episode_id": np.zeros(n, dtype=np.int64),
    }


class NpyTest(unittest.TestCase):
    def test_header_is_fixed_size_and_numpy_readable(self):
        for shape in ((), (32,), (84, 84, 4)):
            self.assertEqual(len(npy_header("<f4", 12345, shape)), HEADER_SIZE)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.npy")
            data = np.arange(6, dtype=np.float32).reshape(3, 2)
            with open(path, "wb") as f:
                f.write(npy_header("<f4", 3, (2,)) + data.tobytes())
            np.testing.assert_array_equal(np.load(path), data)  # numpy itself accepts the padded header
            np.testing.assert_array_equal(read_npy(path), data)

    def test_truncated_file_loads_whole_rows_only(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.npy")
            data = np.arange(8, dtype=np.float32).reshape(4, 2)
            with open(path, "wb") as f:
                f.write(npy_header("<f4", 4, (2,)) + data.tobytes()[:-6])  # last row cut mid-way
            np.testing.assert_array_equal(read_npy(path), data[:3])


class WriterReaderTest(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            w = RecordingWriter(d, meta={"game": "test", "kinds": {"0": "policy"}}, flush_rows=3)
            w.append(**rows(2, 0))
            w.append(**rows(2, 2))  # crosses flush_rows -> header patched to 4
            w.append(**rows(1, 4))
            w.close()

            with open(os.path.join(d, "meta.json")) as f:
                meta = json.load(f)
            self.assertEqual(meta["rows"], 5)
            self.assertTrue(meta["closed"])
            self.assertEqual(meta["columns"]["frame"], {"descr": "|u1", "shape": [2, 2, 1]})
            self.assertEqual(meta["game"], "test")

            r = Recording(d)
            self.assertEqual(len(r), 5)
            np.testing.assert_array_equal(r.column("obs")[4], np.arange(16, 20, dtype=np.float32))
            self.assertEqual(int(r.column("frame")[4, 0, 0, 0]), 4)
            self.assertEqual(r.column("done").dtype, np.bool_)
            with open(os.path.join(d, "reward.npy"), "rb") as f:
                head = f.read(HEADER_SIZE)
            (hlen,) = struct.unpack("<H", head[8:10])
            self.assertIn("'shape': (5,)", head[10 : 10 + hlen].decode("latin1"))

    def test_layout_is_fixed_by_the_first_append(self):
        with tempfile.TemporaryDirectory() as d:
            w = RecordingWriter(d)
            w.append(**rows(1))
            with self.assertRaises(ValueError):
                w.append(**{**rows(1), "obs": np.zeros((1, 5), np.float32)})
            with self.assertRaises(ValueError):
                w.append(**{**rows(1), "extra": np.zeros(1, np.float32)})
            with self.assertRaises(ValueError):
                w.append(**{k: v for k, v in rows(1).items() if k != "reward"})
            w.close()

    def test_unflushed_rows_survive_a_crash(self):
        with tempfile.TemporaryDirectory() as d:
            w = RecordingWriter(d, flush_rows=1000)
            w.append(**rows(3))
            w.flush()
            w.append(**rows(2, 3))  # pending, never flushed: the process "dies" here
            for col in w.columns.values():
                col["file"].close()
            r = Recording(d)
            self.assertEqual(len(r), 3)
            self.assertFalse(r.meta["closed"])

    def test_episodes_group_interleaved_agents(self):
        with tempfile.TemporaryDirectory() as d:
            w = RecordingWriter(d)
            for tick in range(6):
                batch = rows(2, tick * 2)  # agents 0 and 1 every tick
                batch["episode_id"] = np.array([tick // 3, tick // 4], dtype=np.int64)
                batch["done"] = np.array([tick == 2, tick == 3])
                w.append(**batch)
            w.close()
            r = Recording(d)
            eps = r.episodes()
            self.assertEqual(
                [(e["agent"], e["episode"], len(e["rows"])) for e in eps],
                [(0, 0, 3), (1, 0, 4), (0, 1, 3), (1, 1, 2)],
            )
            first = r.episode(eps[0]["rows"], ["obs", "done"])
            np.testing.assert_array_equal(first["obs"][:, 0], [0.0, 8.0, 16.0])
            self.assertEqual(first["done"].tolist(), [False, False, True])
            s = r.summary()
            self.assertEqual((s["rows"], s["episodes"], s["closed_episodes"]), (12, 4, 2))


if __name__ == "__main__":
    unittest.main()
