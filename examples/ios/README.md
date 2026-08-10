# iOS example app

A minimal SwiftUI chat app over the same engine as everything else in this repository, linked
statically through the C ABI ([`bmoe_c.h`](../../core/include/bmoe/bmoe_c.h)). iOS has no
equivalent of the Android app's ProcessBuilder-around-`bmoe-cli` arrangement — an app cannot
spawn processes — so here the engine lives inside the app process, which is the arrangement the
`Session` API was designed for anyway.

**Status: scaffold.** The engine's Darwin code paths (F_NOCACHE direct reads, Mach memory
telemetry, jetsam-aware cache eviction) build and pass the byte-identity gates on macOS in CI;
this app compiles only on a Mac and has **not yet been validated on a device**. The open
question a first device run must answer is the jetsam budget: whether the dense weights + expert
cache + KV of the model you pick fit inside what iOS grants the app (see Memory, below).

## Build

Prerequisites: a Mac with Xcode 16+, [CMake](https://cmake.org), and
[XcodeGen](https://github.com/yonaskolb/XcodeGen) (`brew install cmake xcodegen`). Running on a
device needs an Apple Developer account: the increased-memory-limit entitlement this app carries
requires a paid team, not a free personal one.

```bash
git submodule update --init --recursive
scripts/build-ios.sh          # → dist/ios/bmoe.xcframework (CPU engine, static)
cd examples/ios
xcodegen generate
open BMoEExample.xcodeproj    # set your Team under Signing & Capabilities, then run on a device
```

`build-ios.sh` builds the engine CPU-only (`GGML_METAL=OFF`) on purpose: the streaming seam
rebinds expert tensors in host memory and computes them on the CPU backend. Decode on a
far-past-RAM model is flash-I/O-bound, so the GPU is not what is missing; offloading the dense
side to Metal is future work, not a flag.

## Getting a model onto the phone

The app lists `*.gguf` files in its Documents directory. `UIFileSharingEnabled` is on, so with
the phone plugged in they can be copied in Finder (phone → Files → BMoE Example), or through the
Files app / AirDrop. For a multi-shard model copy **all** shards and pick the first
(`-00001-of-...`), exactly as with the CLI. Mind free space first: DeepSeek V4 Flash UD-IQ2_M is
~91 GB on disk.

Start smaller than the headline model: a Qwen3-30B-A3B Q4_K_M (~18.6 GB) proves the whole path
on a phone-sized download before you commit an evening to copying 91 GB over USB.

## Memory

The app's entitlements request the increased memory limit and extended virtual addressing —
without the latter, mmapping a model tens of GB large does not even map. What iOS actually
grants is device-dependent and only measurable on the device; the engine sizes its cache from
`os_proc_available_memory`, i.e. from the app's remaining jetsam headroom, when the cache is set
to auto. If the app dies mid-load or mid-generation without an error, that is jetsam — lower the
cache budget, or pick the `mmap` dense policy, and watch the telemetry row for what the process
footprint was when it died.

Generation only runs in the foreground: iOS suspends backgrounded apps, so the app keeps the
screen awake while generating instead of pretending it could continue in the background.
