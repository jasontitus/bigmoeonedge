#!/usr/bin/env bash
# Build and run the Swift smoke gate for the C ABI against the tiny gate model. Needs a Swift
# toolchain (swiftc on PATH) and a completed host build — the tiny model comes from the
# byte-identity gates' generator, so run ctest (or just its make_tiny_model tests) first:
#   cd build && ctest -R moe_make_tiny_model_qwen3moe
#   tests/swift-abi/run.sh
# Env overrides: BUILD_DIR (default build). Argument 1 overrides the model path.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD="${BUILD_DIR:-$ROOT/build}"
MODEL="${1:-$BUILD/tests/tiny-moe-qwen3moe.gguf}"

if [ ! -f "$MODEL" ]; then
    echo "tiny model missing at $MODEL — run: cd $BUILD && ctest -R moe_make_tiny_model_qwen3moe"
    exit 1
fi

# The engine's C++ runtime differs by platform; everything else about the link is the same.
CXXRT="-lstdc++"
[ "$(uname -s)" = "Darwin" ] && CXXRT="-lc++"

OUT="${TMPDIR:-/tmp}/bmoe-swift-abi"
mkdir -p "$OUT"
swiftc "$ROOT/tests/swift-abi/main.swift" \
    -I "$ROOT/tests/swift-abi" \
    -L "$BUILD/core" -L "$BUILD/bin" \
    -lbmoe_core -lllama-common -lllama -lggml -lggml-base -lggml-cpu $CXXRT -lm \
    -Xlinker -rpath -Xlinker "$BUILD/bin" \
    -o "$OUT/swift-abi"

"$OUT/swift-abi" "$MODEL"
