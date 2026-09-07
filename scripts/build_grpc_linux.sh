#!/usr/bin/env bash
#
# Build gRPC (submodule third_party/grpc, with its bundled protobuf / abseil / boringssl / re2 / c-ares / zlib)
# as static libraries with the Unreal Engine Linux toolchain (clang, libc++, rockylinux sysroot), and bundle
# everything into ONE archive that the ManipleGrpc third-party module links.
#
#   output: ue/Plugins/ManipleInference/Source/ThirdParty/ManipleGrpc/lib/Linux/libmaniple_grpc.a
#           ue/Plugins/ManipleInference/Source/ThirdParty/ManipleGrpc/include/{grpc,grpcpp,google,absl}
#           ue/Plugins/ManipleInference/Source/ThirdParty/ManipleGrpc/tools/{protoc,grpc_cpp_plugin}
#
# All outputs are gitignored. Takes ~10 minutes on 24 cores.
#
# Usage: UE_ROOT=~/UnrealEngine scripts/build_grpc_linux.sh

set -euo pipefail

# ---------------------------------------------------------------- paths

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UE_ROOT="${UE_ROOT:-$HOME/UnrealEngine}"
TC="$(ls -d "$UE_ROOT"/Engine/Extras/ThirdPartyNotUE/SDKs/HostLinux/Linux_x64/v*_clang-*/x86_64-unknown-linux-gnu | tail -1)"

SRC="$REPO/third_party/grpc"
BUILD="$REPO/third_party/_build/grpc-linux"
OUT="$REPO/ue/Plugins/ManipleInference/Source/ThirdParty/ManipleGrpc"

if [ ! -f "$SRC/CMakeLists.txt" ]; then
  echo "submodule missing: git submodule update --init --recursive third_party/grpc"
  exit 1
fi

if [ ! -x "$TC/bin/clang++" ]; then
  echo "UE toolchain not found under $UE_ROOT"
  exit 1
fi

echo "toolchain: $TC"

# ---------------------------------------------------------------- configure

# Same flags UE uses for its own third-party code: hidden symbols, PIC, libc++ from the sysroot.
FLAGS="--sysroot=$TC -fPIC -fvisibility=hidden -fvisibility-inlines-hidden -O2"

cmake -G Ninja -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_C_COMPILER="$TC/bin/clang" \
  -DCMAKE_CXX_COMPILER="$TC/bin/clang++" \
  -DCMAKE_AR="$TC/bin/llvm-ar" \
  -DCMAKE_RANLIB="$TC/bin/llvm-ranlib" \
  -DCMAKE_C_FLAGS="$FLAGS" \
  -DCMAKE_CXX_FLAGS="$FLAGS -stdlib=libc++" \
  -DCMAKE_EXE_LINKER_FLAGS="--sysroot=$TC -stdlib=libc++ -fuse-ld=lld -lc++abi -lpthread" \
  -DCMAKE_CXX_STANDARD=17 \
  -DABSL_PROPAGATE_CXX_STD=ON \
  \
  -DgRPC_BUILD_TESTS=OFF \
  -DgRPC_BUILD_CODEGEN=ON \
  -DgRPC_BUILD_GRPC_CPP_PLUGIN=ON \
  -DgRPC_BUILD_CSHARP_EXT=OFF \
  -DgRPC_BUILD_GRPC_CSHARP_PLUGIN=OFF \
  -DgRPC_BUILD_GRPC_NODE_PLUGIN=OFF \
  -DgRPC_BUILD_GRPC_OBJECTIVE_C_PLUGIN=OFF \
  -DgRPC_BUILD_GRPC_PHP_PLUGIN=OFF \
  -DgRPC_BUILD_GRPC_PYTHON_PLUGIN=OFF \
  -DgRPC_BUILD_GRPC_RUBY_PLUGIN=OFF \
  \
  -DgRPC_ABSL_PROVIDER=module \
  -DgRPC_PROTOBUF_PROVIDER=module \
  -DgRPC_SSL_PROVIDER=module \
  -DgRPC_ZLIB_PROVIDER=module \
  -DgRPC_CARES_PROVIDER=module \
  -DgRPC_RE2_PROVIDER=module \
  \
  -DgRPC_INSTALL=OFF \
  -Dprotobuf_INSTALL=OFF \
  -Dprotobuf_BUILD_TESTS=OFF \
  -Dutf8_range_ENABLE_INSTALL=OFF \
  -DABSL_ENABLE_INSTALL=OFF

# ---------------------------------------------------------------- build

cmake --build "$BUILD" -j"$(nproc)" --target grpc++ protoc grpc_cpp_plugin

# ---------------------------------------------------------------- collect outputs

rm -rf "$OUT/lib/Linux" "$OUT/include" "$OUT/tools"
mkdir -p "$OUT/lib/Linux" "$OUT/include" "$OUT/tools"

# host tools used by gen_triton_protos.sh
cp "$BUILD/third_party/protobuf/protoc" "$BUILD/grpc_cpp_plugin" "$OUT/tools/"

# one archive from all static libs: no link-order problems for UBT
LIBS=$(find "$BUILD" -name "*.a" | sort)
{
  echo "CREATE $OUT/lib/Linux/libmaniple_grpc.a"
  for l in $LIBS; do
    echo "ADDLIB $l"
  done
  echo "SAVE"
  echo "END"
} | "$TC/bin/llvm-ar" -M
"$TC/bin/llvm-ranlib" "$OUT/lib/Linux/libmaniple_grpc.a"

# public headers only
cp -r "$SRC/include/grpc" "$SRC/include/grpcpp" "$OUT/include/"
cp -r "$SRC/third_party/protobuf/src/google" "$OUT/include/"
cp -r "$SRC/third_party/abseil-cpp/absl" "$OUT/include/"
find "$OUT/include" -type f ! \( -name "*.h" -o -name "*.inc" -o -name "*.hpp" \) -delete
find "$OUT/include" -type d -empty -delete

echo "done: $(du -sh "$OUT/lib/Linux/libmaniple_grpc.a" | cut -f1) archive from $(echo "$LIBS" | wc -l) libs; tools in $OUT/tools"
