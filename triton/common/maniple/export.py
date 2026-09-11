"""Export a policy's actor as a static Triton model 'policy_<name>/<version>/model.onnx' (+ config.pbtxt),
and optionally compile it to TensorRT as 'policy_<name>_trt'. Loaded on demand by <algo>_infer.

The exported model has one input per spec input (obs, frame, ...), and outputs 'action' [N, action_dim]
(means / logits, groups in spec order), 'log_std' [N, continuous_dim] when any group is continuous, and
'policy_version'. The spec is written next to the model as 'policy_<name>/spec.json' so <algo>_infer knows
the layout (which columns are which group) without asking the trainer."""

from __future__ import annotations

import copy
import os
import shutil
import tempfile

import torch

from .nets import Actor
from .spec import AgentSpec

TRITON_DTYPES = {"fp32": "TYPE_FP32", "uint8": "TYPE_UINT8", "string": "TYPE_STRING"}
TORCH_DTYPES = {"fp32": torch.float32, "uint8": torch.uint8}

SPEC_FILE = "spec.json"


class _Exported(torch.nn.Module):
    """Actor + constant 'policy_version' output, so whoever serves the model can report which version it is."""

    def __init__(self, actor: Actor, version: int):
        super().__init__()
        self.actor = actor
        self.register_buffer("version", torch.tensor([version], dtype=torch.int64))

    def forward(self, *inputs):
        out = self.actor(*inputs)
        outs = out if isinstance(out, tuple) else (out,)
        return (*outs, self.version.expand(inputs[0].shape[0]))


CONFIG = """name: "{model}"
platform: "{platform}"
max_batch_size: 1024
input  [ {inputs} ]
output [ {outputs} ]
dynamic_batching {{ preferred_batch_size: [ 64, 256, 1024 ] max_queue_delay_microseconds: 500 }}
instance_group [ {{ count: 1, kind: KIND_GPU }} ]
version_policy: {{ all: {{ }} }}
"""


def _inputs_config(spec: AgentSpec) -> str:
    return ", ".join(
        f'{{ name: "{i.name}", data_type: {TRITON_DTYPES[i.dtype]}, dims: [ {", ".join(str(d) for d in i.shape)} ] }}'
        for i in spec.inputs
    )


def _outputs_config(spec: AgentSpec) -> str:
    outputs = f'{{ name: "action", data_type: TYPE_FP32, dims: [ {spec.action_dim} ] }}'
    if spec.continuous_dim > 0:
        outputs += f', {{ name: "log_std", data_type: TYPE_FP32, dims: [ {spec.continuous_dim} ] }}'
    outputs += ', { name: "policy_version", data_type: TYPE_INT64, dims: [ 1 ], reshape: { shape: [ ] } }'
    return outputs


def output_names(spec: AgentSpec) -> list[str]:
    names = ["action"]
    if spec.continuous_dim > 0:
        names.append("log_std")
    return names + ["policy_version"]


def _pin_output_dims(onnx_path: str, spec: AgentSpec) -> None:
    """The tracer leaves the action dimension of 'log_std' symbolic ([-1, -1]); Triton then refuses the config
    ([-1, dim]). Pin every non-batch output dimension to its known size."""
    import onnx

    widths = {"action": spec.action_dim, "log_std": spec.continuous_dim}
    model = onnx.load(onnx_path)
    for output in model.graph.output:
        dims = output.type.tensor_type.shape.dim
        for i, dim in enumerate(dims):
            if i == 0:
                continue  # batch stays dynamic
            if dim.dim_param:
                dim.dim_param = ""
                dim.dim_value = widths.get(output.name, 1)
    onnx.save(model, onnx_path)


def policy_model_name(name: str) -> str:
    return f"policy_{name}"


def latest_exported_version(name: str, model_repository: str) -> int:
    """Highest numeric version dir of 'policy_<name>' in the repository, 0 if none."""
    d = os.path.join(model_repository, policy_model_name(name))
    if not os.path.isdir(d):
        return 0
    return max((int(v) for v in os.listdir(d) if v.isdigit()), default=0)


def _write_if_changed(path: str, content: str) -> None:
    """Triton reloads a model on a config change, so only touch the file when the content differs."""
    if os.path.exists(path):
        with open(path) as f:
            if f.read() == content:
                return
    with open(path, "w") as f:
        f.write(content)


def _write_config(model_dir: str, model: str, platform: str, spec: AgentSpec) -> None:
    content = CONFIG.format(
        model=model, platform=platform, inputs=_inputs_config(spec), outputs=_outputs_config(spec)
    )
    _write_if_changed(os.path.join(model_dir, "config.pbtxt"), content)


def write_spec(model_dir: str, spec: AgentSpec) -> None:
    _write_if_changed(os.path.join(model_dir, SPEC_FILE), spec.to_json())


def read_spec(model_dir: str) -> AgentSpec | None:
    """The layout of an exported policy; None for exports written before the file existed (v1 layout)."""
    path = os.path.join(model_dir, SPEC_FILE)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return AgentSpec.from_json(f.read())


def dummy_inputs(spec: AgentSpec, rows: int = 1) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.zeros(rows, *[int(d) for d in i.shape], dtype=TORCH_DTYPES[i.dtype]) for i in spec.inputs
    )


def export_actor(
    actor: Actor,
    spec: AgentSpec,
    name: str,
    version: int,
    model_repository: str,
    keep: set[int] | None = None,
) -> str:
    """Write 'policy_<name>/<version>/model.onnx'. `keep` = versions that must survive pruning (None = keep all)."""
    model_dir = os.path.join(model_repository, policy_model_name(name))
    version_dir = os.path.join(model_dir, str(version))
    os.makedirs(model_dir, exist_ok=True)
    _write_config(model_dir, policy_model_name(name), "onnxruntime_onnx", spec)
    write_spec(model_dir, spec)

    # write to a temp dir and rename: Triton must never see a half-written version directory
    tmp = tempfile.mkdtemp(prefix=f".{version}-", dir=model_dir)
    onnx_path = os.path.join(tmp, "model.onnx")
    module = _Exported(copy.deepcopy(actor).to("cpu").eval(), version)  # never move the live model
    outputs = output_names(spec)
    torch.onnx.export(
        module,
        dummy_inputs(spec),
        onnx_path,
        input_names=spec.input_names,
        output_names=outputs,
        dynamic_axes={**{i: {0: "batch"} for i in spec.input_names}, **{o: {0: "batch"} for o in outputs}},
        opset_version=17,
        dynamo=False,
    )
    if os.path.isdir(version_dir):  # never overwrite a version Triton may have loaded
        shutil.rmtree(tmp)
        raise FileExistsError(f"{version_dir} already exists")
    _pin_output_dims(onnx_path, spec)

    os.chmod(tmp, 0o755)
    os.replace(tmp, version_dir)
    if keep is not None:
        prune_versions(model_dir, keep | {version})
    return version_dir


def prune_versions(model_dir: str, keep: set[int]) -> None:
    for entry in os.listdir(model_dir):
        if entry.isdigit() and int(entry) not in keep:
            shutil.rmtree(os.path.join(model_dir, entry), ignore_errors=True)


# ---------------------------------------------------------------- TensorRT


def trt_model_name(name: str) -> str:
    return f"policy_{name}_trt"


def _trt_shapes(spec: AgentSpec, rows: int) -> str:
    """trtexec shape list: 'obs:64x32,frame:64x84x84x4'."""
    return ",".join(f"{i.name}:{'x'.join(str(d) for d in (rows, *i.shape))}" for i in spec.inputs)


def build_trt(
    name: str,
    version: int,
    spec: AgentSpec,
    model_repository: str,
    trtexec: str = "/usr/src/tensorrt/bin/trtexec",
) -> str:
    """Compile 'policy_<name>/<version>/model.onnx' into 'policy_<name>_trt/<version>/model.plan'. Blocking; run in a thread."""
    import subprocess

    onnx_path = os.path.join(model_repository, policy_model_name(name), str(version), "model.onnx")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"{onnx_path} is not exported")

    model_dir = os.path.join(model_repository, trt_model_name(name))
    os.makedirs(model_dir, exist_ok=True)
    _write_config(model_dir, trt_model_name(name), "tensorrt_plan", spec)
    write_spec(model_dir, spec)

    tmp = tempfile.mkdtemp(prefix=f".{version}-", dir=model_dir)
    plan_path = os.path.join(tmp, "model.plan")
    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={plan_path}",
        f"--minShapes={_trt_shapes(spec, 1)}",
        f"--optShapes={_trt_shapes(spec, 64)}",
        f"--maxShapes={_trt_shapes(spec, 1024)}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(plan_path):
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError(f"trtexec failed: {result.stderr[-800:] or result.stdout[-800:]}")

    version_dir = os.path.join(model_dir, str(version))
    if os.path.isdir(version_dir):
        shutil.rmtree(version_dir)
    os.chmod(tmp, 0o755)
    os.replace(tmp, version_dir)
    return version_dir
