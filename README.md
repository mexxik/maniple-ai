# maniple-ai

A unified ML inference layer for Unreal Engine 5.8 with two interchangeable backends.
## Build

Requires Unreal Engine 5.8 (source or launcher build), CMake ≥ 3.20, Ninja, and Docker with the NVIDIA runtime.

```bash
git clone --recurse-submodules https://github.com/mexxik/maniple-ai.git
cd maniple-ai
UE_ROOT=~/UnrealEngine scripts/build_grpc_linux.sh   # gRPC + protobuf with the UE toolchain (one-off, ~10 min)
scripts/gen_triton_protos.sh                          # Triton stubs (already committed; re-run after a Triton bump)
cd triton && docker compose up -d                     # model server on :8000 (HTTP) / :8001 (gRPC)
```

Then add `ue/Plugins/ManipleInference` (core: `FManipleTritonClient`, `FManipleBatchInferer`) to your project's
`Plugins/` and enable it. `ue/Plugins/ManipleLyra` is the reference integration for the Lyra Starter Game
(`-ManipleBrain=triton` and friends, see the plugin header comments).

Model artifacts (`*.onnx`, TensorRT plans) are not in git: fill `triton/model_repository/<model>/<version>/` yourself.

## Code style

    uvx pre-commit install   # once per clone: the same checks run on staged files at every commit
    scripts/lint.sh          # check the whole repo: ruff (Python, pyproject.toml) + clang-format (C++, Unreal style)
    scripts/lint.sh --fix    # rewrite in place

Runs the tools through `uvx`, nothing to install beyond [uv](https://docs.astral.sh/uv/).
