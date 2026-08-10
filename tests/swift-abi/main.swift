// Swift smoke gate for the C ABI (bmoe_c.h): drives the real engine through the exact bridging
// the iOS app relies on — module import, struct field bridging, the capture-free callback
// trampoline with an Unmanaged user pointer, withCString parameter lifetimes — against the tiny
// MoE model the byte-identity gates generate. What it proves is that the ABI is consumable from
// Swift and behaves (tokens arrive, text accumulates, stats add up); byte-identity itself is the
// C++ gates' job. Runs on any host with a Swift toolchain; see run.sh.
import bmoe_c

#if canImport(Glibc)
    import Glibc
#else
    import Darwin
#endif

func fail(_ msg: String) -> Never {
    print("swift-abi FAIL: \(msg)")
    exit(1)
}

func check(_ cond: Bool, _ msg: String) {
    if !cond { fail(msg) }
}

guard CommandLine.arguments.count == 2 else {
    fail("usage: swift-abi <tiny-moe.gguf>")
}
let modelPath = CommandLine.arguments[1]

// 1. Defaults come through the bridge and match the C++ side's documented ones.
var params = bmoe_session_params_default()
check(params.o_direct, "default o_direct should be true")
check(params.io_threads == 4, "default io_threads should be 4")
check(params.dense_weights.rawValue == BMOE_DENSE_ANON.rawValue, "default dense mode should be anon")
check(!params.moe_stream, "streaming should be opt-in")

// 2. The error path fills the caller's buffer instead of crashing.
var err = [CChar](repeating: 0, count: 256)
check(bmoe_open(nil, &err, numericCast(err.count)) == nil, "open(nil) must fail")
check(String(cString: err).contains("model_path"), "open(nil) should say what is missing")

// 3. Open the tiny model with streaming on (cache off: the tiny expert set is below the cache
// floor, and 0 is the documented no-cache mode).
params.n_threads = 2
params.n_ctx = 512
params.moe_stream = true
params.cache_mb = 0
let session = modelPath.withCString { cs -> OpaquePointer? in
    params.model_path = cs
    return bmoe_open(&params, &err, numericCast(err.count))
}
guard let session else { fail("open failed: \(String(cString: err))") }
check(String(cString: bmoe_arch(session)) == "qwen3moe", "arch should be qwen3moe")
check(bmoe_n_ctx(session) == 512, "n_ctx should round-trip")
check(bmoe_load_seconds(session) > 0, "load_seconds should be measured")

// 4. Generate through the callback trampoline — the pattern Engine.swift uses on iOS.
final class Collector {
    var steps: [Int] = []
    var pieces = ""
}

func trampoline(_ tok: UnsafePointer<bmoe_token>?, _ user: UnsafeMutableRawPointer?) {
    guard let t = tok?.pointee, let user else { return }
    let c = Unmanaged<Collector>.fromOpaque(user).takeUnretainedValue()
    c.steps.append(Int(t.step))
    if let piece = t.piece { c.pieces += String(cString: piece) }
    if t.read_mib < 0 { fail("negative read_mib") }
}

func generate(_ prompt: String, nPredict: Int32) -> (ok: Bool, text: String, stats: bmoe_stats) {
    let collector = Collector()
    let user = Unmanaged.passRetained(collector).toOpaque()
    defer { Unmanaged<Collector>.fromOpaque(user).release() }
    var g = bmoe_generate_params_default()
    g.n_predict = nPredict
    let ok = prompt.withCString { cs -> Bool in
        g.prompt = cs
        return bmoe_generate(session, &g, trampoline, user)
    }
    var stats = bmoe_stats()
    bmoe_last_stats(session, &stats)
    check(collector.steps == Array(1 ... Int(stats.n_generated)), "callback steps must be 1..n in order")
    check(collector.pieces == String(cString: bmoe_last_text(session)),
          "concatenated pieces must equal the final text")
    return (ok, String(cString: bmoe_last_text(session)), stats)
}

let first = generate("The capital of Japan is", nPredict: 8)
check(first.ok, "generate failed: \(String(cString: bmoe_last_error(session)))")
check(first.stats.ok && !first.stats.cancelled, "stats should report a clean run")
check(first.stats.n_generated == 8, "expected 8 tokens, got \(first.stats.n_generated)")
check(first.stats.tokens_per_second > 0, "tok/s should be positive")
check(first.stats.moe_read_mib > 0, "a streamed run must have read expert bytes")
check(!first.text.isEmpty, "generated text should not be empty")

// 5. The session is reusable (the whole point of Session over run()); greedy decode makes the
// repeat deterministic, so the same prompt must give the same text.
let second = generate("The capital of Japan is", nPredict: 8)
check(second.ok, "second generate failed")
check(second.text == first.text, "greedy decode must repeat identically on a reused session")

bmoe_close(session)
bmoe_close(nil) // documented NULL-safe

let tps = (first.stats.tokens_per_second * 10).rounded() / 10
let mib = (first.stats.moe_read_mib * 100).rounded() / 100
print("swift-abi ok: \(first.stats.n_generated) tokens, \(tps) tok/s, \(mib) MiB streamed")
