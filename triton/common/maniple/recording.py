"""Recordings of play: one directory per run, one .npy file per column, rows appended while the game runs.

    <dir>/meta.json          who produced it, the column layout, what the observation fields mean
    <dir>/<column>.npy       [rows, ...] plain numpy files, all columns row-aligned

Every recording has the policy inputs (obs, frame, ...) and action, reward, done, logp, policy_version,
agent_id, episode_id. A producer adds what it likes (Lyra: time, kind, pose). An episode is the set of rows
with one (agent_id, episode_id); rows of many agents interleave, one decision tick after the other.

The .npy header is a fixed 128 bytes whose row count is patched on every flush, so a run that crashes leaves
readable files: the reader trusts the file size, not the header. The C++ twin is ManipleRecorder.h.
"""

from __future__ import annotations

import json
import os
import struct
import time

import numpy as np

FORMAT = "maniple-recording"
FORMAT_VERSION = 1
META_FILE = "meta.json"
HEADER_SIZE = 128
MAGIC = b"\x93NUMPY\x01\x00"

# Triton datatype names <-> numpy descr, the same table as the C++ recorder
DESCR = {"FP32": "<f4", "INT64": "<i8", "INT32": "<i4", "UINT8": "|u1", "BOOL": "|b1"}


# ---------------------------------------------------------------- npy files


def npy_header(descr: str, rows: int, row_shape: tuple[int, ...]) -> bytes:
    """A version 1.0 header padded to HEADER_SIZE bytes, so the row count can be rewritten in place."""
    shape = "(" + ", ".join(str(int(d)) for d in (rows, *row_shape)) + ("," if not row_shape else "") + ")"
    text = f"{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}"
    body = text.encode("latin1")
    pad = HEADER_SIZE - len(MAGIC) - 2 - len(body) - 1
    if pad < 0:
        raise ValueError(f"npy header does not fit in {HEADER_SIZE} bytes: {text}")
    return MAGIC + struct.pack("<H", HEADER_SIZE - len(MAGIC) - 2) + body + b" " * pad + b"\n"


def read_npy(path: str) -> np.ndarray:
    """Memory-map one column. The row count comes from the file size, so a truncated file still loads."""
    with open(path, "rb") as f:
        head = f.read(len(MAGIC) + 2)
        if head[: len(MAGIC)] != MAGIC:
            raise ValueError(f"{path}: not a version 1.0 .npy file")
        (header_len,) = struct.unpack("<H", head[len(MAGIC) :])
        header = f.read(header_len).decode("latin1")
    fields = _parse_header(header)
    dtype = np.dtype(fields["descr"])
    row_shape = tuple(int(d) for d in fields["shape"][1:])
    row_bytes = dtype.itemsize * int(np.prod(row_shape, dtype=np.int64))
    offset = len(MAGIC) + 2 + header_len
    rows = (os.path.getsize(path) - offset) // row_bytes
    if rows == 0:
        return np.zeros((0, *row_shape), dtype=dtype)
    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(rows, *row_shape))


def _parse_header(header: str) -> dict:
    import ast

    return ast.literal_eval(header.strip())


# ---------------------------------------------------------------- writer


class RecordingWriter:
    """Appends row-aligned columns to a directory. Columns are fixed by the first append.

    w = RecordingWriter(dir, meta={"game": "pong", ...})
    w.append(obs=..., action=..., reward=..., ...)   # every array [n, ...], same n
    w.close()
    """

    def __init__(
        self, directory: str, meta: dict | None = None, flush_rows: int = 1024, flush_seconds: float = 5.0
    ):
        self.directory = directory
        self.meta = dict(meta or {})
        self.flush_rows = flush_rows
        self.flush_seconds = flush_seconds
        self.rows = 0
        self.columns: dict[str, dict] = {}  # name -> {"descr", "shape", "file", "pending", "written"}
        self._pending_rows = 0
        self._last_flush = time.time()
        os.makedirs(directory, exist_ok=True)
        self._write_meta()

    def append(self, **columns: np.ndarray) -> None:
        arrays = {name: np.ascontiguousarray(a) for name, a in columns.items()}
        n = None
        for name, array in arrays.items():
            if n is None:
                n = array.shape[0]
            elif array.shape[0] != n:
                raise ValueError(f"column '{name}' has {array.shape[0]} rows, expected {n}")
            if name not in self.columns:
                if self.rows > 0 or self._pending_rows > 0:
                    raise ValueError(f"column '{name}' appeared after the first append")
                self._open_column(name, array.dtype.str, array.shape[1:])
            col = self.columns[name]
            if array.dtype.str != col["descr"] or tuple(array.shape[1:]) != col["shape"]:
                raise ValueError(
                    f"column '{name}' is {array.dtype.str}{list(array.shape[1:])}, expected {col['descr']}{list(col['shape'])}"
                )
        missing = set(self.columns) - set(arrays)
        if missing:
            raise ValueError(f"append is missing columns {sorted(missing)}")
        if not n:
            return
        for name, array in arrays.items():
            self.columns[name]["pending"].append(array.tobytes())
        self._pending_rows += n
        if self._pending_rows >= self.flush_rows or time.time() - self._last_flush > self.flush_seconds:
            self.flush()

    def flush(self) -> None:
        if self._pending_rows:
            self.rows += self._pending_rows
            self._pending_rows = 0
            for col in self.columns.values():
                f = col["file"]
                f.seek(0, os.SEEK_END)
                for chunk in col["pending"]:
                    f.write(chunk)
                col["pending"] = []
                f.seek(0)
                f.write(npy_header(col["descr"], self.rows, col["shape"]))
                f.flush()
            self._write_meta()
        self._last_flush = time.time()

    def close(self) -> None:
        self.flush()
        for col in self.columns.values():
            col["file"].close()
        self._write_meta(closed=True)
        self.columns = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _open_column(self, name: str, descr: str, row_shape: tuple[int, ...]) -> None:
        f = open(os.path.join(self.directory, f"{name}.npy"), "w+b")  # noqa: SIM115  (stays open until close())
        f.write(npy_header(descr, 0, row_shape))
        self.columns[name] = {
            "descr": descr,
            "shape": tuple(int(d) for d in row_shape),
            "file": f,
            "pending": [],
        }

    def _write_meta(self, closed: bool = False) -> None:
        meta = dict(self.meta)
        meta.update(
            {
                "format": FORMAT,
                "format_version": FORMAT_VERSION,
                "columns": {
                    n: {"descr": c["descr"], "shape": list(c["shape"])} for n, c in self.columns.items()
                },
                "rows": self.rows,
                "closed": closed,
            }
        )
        path = os.path.join(self.directory, META_FILE)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(meta, f, indent=2)
        os.replace(tmp, path)


# ---------------------------------------------------------------- reader


class Recording:
    """One recording directory. Columns are memory-mapped on first access."""

    def __init__(self, directory: str):
        self.directory = directory
        with open(os.path.join(directory, META_FILE)) as f:
            self.meta = json.load(f)
        if self.meta.get("format") != FORMAT:
            raise ValueError(f"{directory}: not a {FORMAT} directory")
        self._columns: dict[str, np.ndarray] = {}
        self.names = sorted(os.path.splitext(n)[0] for n in os.listdir(directory) if n.endswith(".npy"))
        self.rows = min((len(self[n]) for n in self.names), default=0)  # a crash can leave columns uneven

    def __getitem__(self, name: str) -> np.ndarray:
        if name not in self._columns:
            self._columns[name] = read_npy(os.path.join(self.directory, f"{name}.npy"))
        return self._columns[name]

    def __contains__(self, name: str) -> bool:
        return name in self.names

    def __len__(self) -> int:
        return self.rows

    def column(self, name: str) -> np.ndarray:
        return self[name][: self.rows]

    def episodes(self) -> list[dict]:
        """Row indices per (agent_id, episode_id), in first-appearance order: [{"agent", "episode", "rows"}]."""
        if self.rows == 0:
            return []
        agent = np.asarray(self.column("agent_id"), dtype=np.int64)
        episode = np.asarray(self.column("episode_id"), dtype=np.int64)
        key = agent * (1 << 32) + episode
        order = np.argsort(key, kind="stable")
        keys, starts, counts = np.unique(key[order], return_index=True, return_counts=True)
        groups = [
            {"agent": int(k >> 32), "episode": int(k & 0xFFFFFFFF), "rows": order[s : s + c]}
            for k, s, c in zip(keys, starts, counts, strict=True)
        ]
        groups.sort(key=lambda g: g["rows"][0])
        return groups

    def episode(self, rows: np.ndarray, names: list[str] | None = None) -> dict[str, np.ndarray]:
        """The columns of one episode (rows from episodes()) as in-memory arrays."""
        return {n: np.asarray(self.column(n)[rows]) for n in (names or self.names)}

    def summary(self) -> dict:
        eps = self.episodes()
        done = np.asarray(self.column("done"), dtype=bool) if "done" in self else None
        out = {
            "rows": self.rows,
            "episodes": len(eps),
            "closed_episodes": int(done.sum()) if done is not None else None,
            "columns": {n: f"{self[n].dtype.str}{list(self[n].shape[1:])}" for n in self.names},
        }
        if "reward" in self:
            out["reward_sum"] = float(np.asarray(self.column("reward"), dtype=np.float64).sum())
        if "kind" in self:
            kinds = self.meta.get("kinds", {})
            values, counts = np.unique(np.asarray(self.column("kind")), return_counts=True)
            out["rows_per_kind"] = {
                kinds.get(str(int(v)), str(int(v))): int(c) for v, c in zip(values, counts, strict=True)
            }
        if "time" in self and self.rows:
            t = np.asarray(self.column("time"), dtype=np.float64)
            out["game_seconds"] = float(t.max() - t.min())
        return out


# ---------------------------------------------------------------- command line


def _cmd_info(args):
    rec = Recording(args.dir)
    s = rec.summary()
    print(f"{args.dir}: {s['rows']} rows, {s['episodes']} episodes ({s['closed_episodes']} closed)")
    for name, layout in s["columns"].items():
        print(f"  {name:16s} {layout}")
    for key in ("reward_sum", "game_seconds", "rows_per_kind"):
        if key in s:
            print(f"  {key}: {s[key]}")
    skip = {"format", "format_version", "columns", "rows", "closed"}
    for key, value in rec.meta.items():
        if key not in skip:
            text = json.dumps(value)
            print(f"  meta.{key}: {text[:100]}{'...' if len(text) > 100 else ''}")
    if args.episodes:
        print("  episodes (agent, episode, rows, return):")
        for e in rec.episodes()[: args.episodes]:
            ret = (
                float(np.asarray(rec.column("reward"))[e["rows"]].sum()) if "reward" in rec else float("nan")
            )
            print(f"    {e['agent']:4d} {e['episode']:6d} {len(e['rows']):6d} {ret:9.3f}")


def _cmd_dump(args):
    rec = Recording(args.dir)
    eps = rec.episodes()
    chosen = [e for e in eps if (args.agent is None or e["agent"] == args.agent)]
    if args.episode is not None:
        chosen = [e for e in chosen if e["episode"] == args.episode]
    if not chosen:
        print("no such episode")
        return
    e = chosen[0]
    data = rec.episode(e["rows"])
    layout = rec.meta.get("layout", {})
    obs_fields = layout.get("obs", {})
    act_fields = layout.get("action", {})
    print(f"agent {e['agent']} episode {e['episode']}: {len(e['rows'])} rows")
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    for i in range(min(len(e["rows"]), args.rows)):
        parts = []
        if "time" in data:
            parts.append(f"t={float(data['time'][i]):7.2f}")
        if "obs" in data:
            obs = data["obs"][i]
            if obs_fields:
                parts.append(
                    " ".join(
                        f"{k}={obs[v[0] : v[0] + v[1]]}" for k, v in obs_fields.items() if k in args.fields
                    )
                )
            else:
                parts.append(f"obs={obs}")
        if "action" in data:
            act = data["action"][i]
            parts.append(
                "act=" + " ".join(f"{k}={act[v]:+.2f}" for k, v in act_fields.items())
                if act_fields
                else f"act={act}"
            )
        parts.append(f"r={float(data['reward'][i]):+.3f}")
        if bool(data["done"][i]):
            parts.append("done")
        print("  " + "  ".join(parts))


def _cmd_compare(args):
    """Same-agent episodes of two recordings, step by step: how far apart are the observation fields?"""
    a, b = Recording(args.a), Recording(args.b)
    layout = a.meta.get("layout", {}).get("obs", {})
    ea = [e for e in a.episodes() if e["agent"] == args.agent_a]
    eb = [e for e in b.episodes() if e["agent"] == args.agent_b]
    n = min(len(ea), len(eb), args.episodes)
    if n == 0:
        print("nothing to compare")
        return
    print(f"{n} episode pairs, fields: mean |a - b| over the first {args.steps} steps")
    header = "  ep   steps  " + "  ".join(f"{k:>9s}" for k in layout)
    print(header)
    totals = {k: [] for k in layout}
    for i in range(n):
        oa = a.episode(ea[i]["rows"], ["obs"])["obs"]
        ob = b.episode(eb[i]["rows"], ["obs"])["obs"]
        steps = min(len(oa), len(ob), args.steps)
        cells = []
        for k, (start, size) in layout.items():
            d = float(np.abs(oa[:steps, start : start + size] - ob[:steps, start : start + size]).mean())
            totals[k].append(d)
            cells.append(f"{d:9.4f}")
        print(f"  {i:2d}  {steps:6d}  " + "  ".join(cells))
    print("  all         " + "  ".join(f"{np.mean(v):9.4f}" for v in totals.values()))


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Inspect recordings of play (maniple.recording).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info", help="rows, episodes, columns, meta")
    p.add_argument("dir")
    p.add_argument("--episodes", type=int, default=0, help="also list the first N episodes")
    p.set_defaults(fn=_cmd_info)

    p = sub.add_parser("dump", help="one episode as a table")
    p.add_argument("dir")
    p.add_argument("--agent", type=int, default=None)
    p.add_argument("--episode", type=int, default=None)
    p.add_argument("--rows", type=int, default=50)
    p.add_argument(
        "--fields", default="vel,health,pitch,rays", help="obs layout fields to print (comma list)"
    )
    p.set_defaults(fn=_cmd_dump)

    p = sub.add_parser(
        "compare", help="observation fields of two recordings, episode by episode (replay check)"
    )
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--agent-a", type=int, default=0)
    p.add_argument("--agent-b", type=int, default=0)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--steps", type=int, default=50)
    p.set_defaults(fn=_cmd_compare)

    args = parser.parse_args(argv)
    if hasattr(args, "fields") and isinstance(args.fields, str):
        args.fields = set(args.fields.split(","))
    args.fn(args)


if __name__ == "__main__":
    main()
