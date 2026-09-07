#!/usr/bin/env bash
# Generate the Triton gRPC client stubs (submodule third_party/triton-common) with the protoc/grpc_cpp_plugin
# produced by build_grpc_linux.sh, into ue/Plugins/ManipleInference/Source/ThirdParty/ManipleGrpc/gen (committed), and
# compile them with the UE toolchain into lib/Linux/libmaniple_triton.a (not committed) so UBT never has to
# compile protobuf-generated code under its own warning settings.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UE_ROOT="${UE_ROOT:-$HOME/UnrealEngine}"
TC="$(ls -d "$UE_ROOT"/Engine/Extras/ThirdPartyNotUE/SDKs/HostLinux/Linux_x64/v*_clang-*/x86_64-unknown-linux-gnu | tail -1)"
TP="$REPO/ue/Plugins/ManipleInference/Source/ThirdParty/ManipleGrpc"
PROTOS="$REPO/third_party/triton-common/protobuf"
GEN="$TP/gen"
[ -x "$TP/tools/protoc" ] || { echo "run scripts/build_grpc_linux.sh first"; exit 1; }
mkdir -p "$GEN" "$TP/lib/Linux"
"$TP/tools/protoc" -I "$PROTOS" --cpp_out="$GEN" --grpc_out="$GEN" --plugin=protoc-gen-grpc="$TP/tools/grpc_cpp_plugin" \
  "$PROTOS/grpc_service.proto" "$PROTOS/model_config.proto"
OBJ="$REPO/third_party/_build/triton-gen"; rm -rf "$OBJ"; mkdir -p "$OBJ"
for cc in "$GEN"/*.cc; do
  "$TC/bin/clang++" --sysroot="$TC" -stdlib=libc++ -std=c++17 -O2 -fPIC -fvisibility=hidden -fvisibility-inlines-hidden \
    -I "$TP/include" -I "$GEN" -c "$cc" -o "$OBJ/$(basename "$cc" .cc).o"
done
rm -f "$TP/lib/Linux/libmaniple_triton.a"
"$TC/bin/llvm-ar" rcs "$TP/lib/Linux/libmaniple_triton.a" "$OBJ"/*.o
ls -la "$GEN" "$TP/lib/Linux"
