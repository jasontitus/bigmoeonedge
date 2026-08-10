#!/usr/bin/env bash
# Build the engine as a static xcframework for iOS. Runs on macOS with Xcode; produces
# dist/ios/bmoe.xcframework, which examples/ios links. Env overrides:
#   BUILD_TYPE (Release), JOBS (hw.ncpu), IOS_DEPLOYMENT_TARGET (16.4),
#   SIMULATOR=1 to add an arm64 simulator slice (device-only by default).
#
# The engine is built CPU-only on purpose: GGML_METAL stays OFF because the streaming seam
# rebinds expert tensor data in host memory and computes experts on the CPU backend — Metal
# would try to own the weights it streams. Offloading the dense side to Metal is future work;
# it is not a flag flip here. See docs/architecture.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
TARGET="${IOS_DEPLOYMENT_TARGET:-16.4}"
DIST="$ROOT/dist/ios"

if [ "$(uname -s)" != "Darwin" ] || ! xcode-select -p >/dev/null 2>&1; then
    echo "build-ios.sh needs macOS with Xcode (xcode-select -p failed)"
    exit 1
fi
cd "$ROOT"
if [ ! -f third_party/llama.cpp/CMakeLists.txt ]; then
    echo "llama.cpp submodule missing — run: git submodule update --init --recursive"
    exit 1
fi

# One slice: configure + build static libs for the given sysroot, then merge every produced
# archive into a single libbmoe.a (an xcframework slice is one library, not a lib set).
build_slice() {
    local sysroot="$1" dir="$2"
    cmake -S . -B "$dir" \
        -DCMAKE_SYSTEM_NAME=iOS \
        -DCMAKE_OSX_SYSROOT="$sysroot" \
        -DCMAKE_OSX_ARCHITECTURES=arm64 \
        -DCMAKE_OSX_DEPLOYMENT_TARGET="$TARGET" \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DBUILD_SHARED_LIBS=OFF \
        -DBMOE_BUILD_CLI=OFF -DBMOE_BUILD_TESTS=OFF \
        -DGGML_METAL=OFF -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF \
        -DLLAMA_CURL=OFF
    cmake --build "$dir" -j "$JOBS"
    find "$dir" -name '*.a' -print0 | xargs -0 libtool -static -o "$dir/libbmoe.a"
}

build_slice iphoneos build-ios-device

# The public surface of the framework is the C ABI alone: bmoe_c.h is what Swift imports
# (via a bridging header); the C++ headers stay engine-internal.
HDRS="$ROOT/build-ios-device/bmoe-headers"
rm -rf "$HDRS" && mkdir -p "$HDRS"
cp core/include/bmoe/bmoe_c.h "$HDRS/"

ARGS=(-library build-ios-device/libbmoe.a -headers "$HDRS")
if [ "${SIMULATOR:-0}" = "1" ]; then
    build_slice iphonesimulator build-ios-sim
    ARGS+=(-library build-ios-sim/libbmoe.a -headers "$HDRS")
fi

rm -rf "$DIST/bmoe.xcframework" && mkdir -p "$DIST"
xcodebuild -create-xcframework "${ARGS[@]}" -output "$DIST/bmoe.xcframework"
echo "built: $DIST/bmoe.xcframework"
