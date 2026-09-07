#!/usr/bin/env bash
#
# Generate the Triton gRPC client stubs from the protos in the triton-common submodule.
#
#   input : third_party/triton-common/protobuf/{grpc_service,model_config}.proto
#   tools : protoc + grpc_cpp_plugin produced by scripts/build_grpc_linux.sh
#   output: ue/Plugins/ManipleInference/Source/ThirdParty/ManipleGrpc/gen/*.pb.{h,cc}   (committed)
#           ue/Plugins/ManipleInference/Source/ThirdParty/ManipleGrpc/lib/Linux/libmaniple_triton.a   (not committed)
#
# The generated .cc files are compiled here with the UE toolchain instead of by UBT, so protobuf-generated
# code never has to pass UBT's warning settings.
#
# Usage: UE_ROOT=~/UnrealEngine scripts/gen_triton_protos.sh

set -euo pipefail

# ---------------------------------------------------------------- paths

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UE_ROOT="${UE_ROOT:-$HOME/UnrealEngine}"
TC="$(ls -d "$UE_ROOT"/Engine/Extras/ThirdPartyNotUE/SDKs/HostLinux/Linux_x64/v*_clang-*/x86_64-unknown-linux-gnu | tail -1)"

TP="$REPO/ue/Plugins/ManipleInference/Source/ThirdParty/ManipleGrpc"
PROTOS="$REPO/third_party/triton-common/protobuf"
GEN="$TP/gen"
OBJ="$REPO/third_party/_build/triton-gen"

if [ ! -x "$TP/tools/protoc" ]; then
  echo "protoc not found: run scripts/build_grpc_linux.sh first"
  exit 1
fi

mkdir -p "$GEN" "$TP/lib/Linux"

# ---------------------------------------------------------------- generate

"$TP/tools/protoc" \
  -I "$PROTOS" \
  --cpp_out="$GEN" \
  --grpc_out="$GEN" \
  --plugin=protoc-gen-grpc="$TP/tools/grpc_cpp_plugin" \
  "$PROTOS/grpc_service.proto" "$PROTOS/model_config.proto"

# ---------------------------------------------------------------- compile with the UE toolchain

rm -rf "$OBJ"
mkdir -p "$OBJ"

for cc in "$GEN"/*.cc; do
  "$TC/bin/clang++" \
    --sysroot="$TC" -stdlib=libc++ -std=c++17 \
    -O2 -fPIC -fvisibility=hidden -fvisibility-inlines-hidden \
    -I "$TP/include" -I "$GEN" \
    -c "$cc" -o "$OBJ/$(basename "$cc" .cc).o"
done

# ---------------------------------------------------------------- archive

rm -f "$TP/lib/Linux/libmaniple_triton.a"
"$TC/bin/llvm-ar" rcs "$TP/lib/Linux/libmaniple_triton.a" "$OBJ"/*.o

echo "generated:"
ls -la "$GEN"
echo "archive:"
ls -la "$TP/lib/Linux/libmaniple_triton.a"
