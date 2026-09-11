"""Reading request tensors in the Triton Python backend (shared by <algo>_train and <algo>_infer)."""

from __future__ import annotations

import numpy as np
import triton_python_backend_utils as pb_utils

from .spec import WIRE_INPUTS


def string_input(request, name):
    t = pb_utils.get_input_tensor_by_name(request, name)
    if t is None:
        return None
    v = t.as_numpy().reshape(-1)[0]
    return v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v)


def array_input(request, name):
    t = pb_utils.get_input_tensor_by_name(request, name)
    return None if t is None else t.as_numpy()


def scalar_input(request, name, default=None):
    a = array_input(request, name)
    return default if a is None else a.reshape(-1)[0]


def read_inputs(request) -> dict:
    """Every observation tensor present on the request (obs, frame, audio, text), cast to its wire dtype."""
    inputs = {}
    for name, (dtype, _rank, _encoder) in WIRE_INPUTS.items():
        a = array_input(request, name)
        if a is None:
            continue
        if dtype == "fp32":
            a = a.astype(np.float32)
        elif dtype == "uint8":
            a = a.astype(np.uint8)
        inputs[name] = a
    return inputs


def row_count(inputs: dict) -> int:
    for array in inputs.values():
        return int(array.shape[0])
    return 0
