"""Export a policy's actor as a static Triton model 'policy_<name>/<version>/model.onnx' (+ config.pbtxt),
and optionally compile it to TensorRT as 'policy_<name>_trt'. Loaded on demand by <algo>_infer."""

from __future__ import annotations

import copy
import os
import shutil
import tempfile

import torch

from .nets import Actor
from .spec import AgentSpec


class _Exported(torch.nn.Module):
    """Actor + constant 'policy_version' output, so whoever serves the model can report which version it is."""

    def __init__(self, actor: Actor, version: int):
        super().__init__()
        self.actor = actor
        self.register_buffer("version", torch.tensor([version], dtype=torch.int64))

    def forward(self, obs):
        out = self.actor(obs)
        outs = out if isinstance(out, tuple) else (out,)
        return (*outs, self.version.expand(obs.shape[0]))


CONFIG = """name: "{model}"
platform: "onnxruntime_onnx"
max_batch_size: 1024
input  [ {{ name: "obs",    data_type: TYPE_FP32, dims: [ {obs} ] }} ]
output [ {outputs} ]
dynamic_batching {{ preferred_batch_size: [ 64, 256, 1024 ] max_queue_delay_microseconds: 500 }}
instance_group [ {{ count: 1, kind: KIND_GPU }} ]
version_policy: {{ all: {{ }} }}
"""


def _pin_output_dims(onnx_path: str, act_dim: int) -> None:
    """The tracer leaves the action dimension of 'log_std' symbolic ([-1, -1]); Triton then refuses the config
    ([-1, act_dim]). Pin every non-batch output dimension to its known size."""
    import onnx

    model = onnx.load(onnx_path)
    for output in model.graph.output:
        dims = output.type.tensor_type.shape.dim
        for i, dim in enumerate(dims):
            if i == 0:
                continue  # batch stays dynamic
            if dim.dim_param:
                dim.dim_param = ""
                dim.dim_value = act_dim if output.name in ("action", "log_std") else 1
    onnx.save(model, onnx_path)


def policy_model_name(name: str) -> str:
    return f"policy_{name}"


def latest_exported_version(name: str, model_repository: str) -> int:
    """Highest numeric version dir of 'policy_<name>' in the repository, 0 if none."""
    d = os.path.join(model_repository, policy_model_name(name))
    if not os.path.isdir(d):
        return 0
    return max((int(v) for v in os.listdir(d) if v.isdigit()), default=0)


def _write_config(model_dir: str, name: str, spec: AgentSpec) -> None:
    """Write config.pbtxt if missing or different (Triton reloads the model on a config change)."""
    outputs = f'{{ name: "action", data_type: TYPE_FP32, dims: [ {spec.action.out_dim} ] }}'
    if spec.action.type == "continuous":
        outputs += f', {{ name: "log_std", data_type: TYPE_FP32, dims: [ {spec.action.out_dim} ] }}'
    outputs += ', { name: "policy_version", data_type: TYPE_INT64, dims: [ 1 ], reshape: { shape: [ ] } }'
    content = CONFIG.format(model=policy_model_name(name), obs=spec.obs_dim, outputs=outputs)

    cfg_path = os.path.join(model_dir, "config.pbtxt")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            if f.read() == content:
                return
    with open(cfg_path, "w") as f:
        f.write(content)


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
    _write_config(model_dir, name, spec)

    # write to a temp dir and rename: Triton must never see a half-written version directory
    tmp = tempfile.mkdtemp(prefix=f".{version}-", dir=model_dir)
    onnx_path = os.path.join(tmp, "model.onnx")
    module = _Exported(copy.deepcopy(actor).to("cpu").eval(), version)  # never move the live model
    outputs = (["action", "log_std"] if spec.action.type == "continuous" else ["action"]) + ["policy_version"]
    torch.onnx.export(
        module,
        torch.zeros(1, spec.obs_dim),
        onnx_path,
        input_names=["obs"],
        output_names=outputs,
        dynamic_axes={"obs": {0: "batch"}, **{o: {0: "batch"} for o in outputs}},
        opset_version=17,
        dynamo=False,
    )
    if os.path.isdir(version_dir):  # never overwrite a version Triton may have loaded
        shutil.rmtree(tmp)
        raise FileExistsError(f"{version_dir} already exists")
    _pin_output_dims(onnx_path, spec.action.out_dim)

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

TRT_CONFIG = """name: "{model}"
platform: "tensorrt_plan"
max_batch_size: 1024
input  [ {{ name: "obs",    data_type: TYPE_FP32, dims: [ {obs} ] }} ]
output [ {outputs} ]
dynamic_batching {{ preferred_batch_size: [ 64, 256, 1024 ] max_queue_delay_microseconds: 500 }}
instance_group [ {{ count: 1, kind: KIND_GPU }} ]
version_policy: {{ all: {{ }} }}
"""


def trt_model_name(name: str) -> str:
    return f"policy_{name}_trt"


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
    outputs = f'{{ name: "action", data_type: TYPE_FP32, dims: [ {spec.action.out_dim} ] }}'
    if spec.action.type == "continuous":
        outputs += f', {{ name: "log_std", data_type: TYPE_FP32, dims: [ {spec.action.out_dim} ] }}'
    outputs += ', { name: "policy_version", data_type: TYPE_INT64, dims: [ 1 ], reshape: { shape: [ ] } }'
    with open(os.path.join(model_dir, "config.pbtxt"), "w") as f:
        f.write(TRT_CONFIG.format(model=trt_model_name(name), obs=spec.obs_dim, outputs=outputs))

    tmp = tempfile.mkdtemp(prefix=f".{version}-", dir=model_dir)
    plan_path = os.path.join(tmp, "model.plan")
    obs = spec.obs_dim
    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={plan_path}",
        f"--minShapes=obs:1x{obs}",
        f"--optShapes=obs:64x{obs}",
        f"--maxShapes=obs:1024x{obs}",
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
