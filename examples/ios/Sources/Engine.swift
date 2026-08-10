import Foundation
import UIKit

/// The telemetry a token row carries into the UI. A subset of bmoe_token, copied out immediately:
/// the C strings are only valid inside the callback.
struct TokenEvent {
    let step: Int
    let wallMs: Double
    let readMiB: Double
    let cacheHitPct: Double
    let text: String
    let reasoning: String
}

/// Streaming knobs the UI exposes. Defaults are the far-past-RAM shape from the README benchmarks
/// (dense weights anonymous, a fixed cache, overlap on) — the regime an iPhone-sized RAM puts most
/// interesting models in.
struct EngineSettings {
    var moeStream = true
    var cacheMB = 2000
    var ioThreads = 4
    var nThreads = 4
    var nCtx = 2048
    var nPredict = 256
    var denseAnon = true
    var overlap = true
}

private final class TokenSink {
    let onToken: (TokenEvent) -> Void
    init(_ onToken: @escaping (TokenEvent) -> Void) { self.onToken = onToken }
}

/// C callback trampoline: a global function (capture-free, so it converts to a C function pointer)
/// that unwraps the TokenSink passed through the user pointer.
private func tokenTrampoline(_ tok: UnsafePointer<bmoe_token>?, _ user: UnsafeMutableRawPointer?) {
    guard let t = tok?.pointee, let user else { return }
    let sink = Unmanaged<TokenSink>.fromOpaque(user).takeUnretainedValue()
    sink.onToken(TokenEvent(
        step: Int(t.step),
        wallMs: t.wall_ms,
        readMiB: t.read_mib,
        cacheHitPct: t.cache_hit_pct,
        text: t.text.map { String(cString: $0) } ?? "",
        reasoning: t.reasoning.map { String(cString: $0) } ?? ""
    ))
}

@MainActor
final class Engine: ObservableObject {
    enum State: Equatable {
        case idle
        case loading
        case ready(model: String)
        case generating(model: String)
        case failed(String)
    }

    @Published var state: State = .idle
    @Published var answer = ""
    @Published var reasoning = ""
    @Published var stats = ""
    @Published var liveTelemetry = ""
    @Published var settings = EngineSettings()

    private var session: OpaquePointer? // bmoe_session*
    /// All engine calls run here: bmoe_generate blocks for the whole generation, and open takes
    /// tens of seconds for a big model. Serial, so calls can never overlap on one session.
    private let queue = DispatchQueue(label: "bmoe.engine", qos: .userInitiated)
    private var pressure: DispatchSourceMemoryPressure?
    private var shrinkAfterGeneration = false

    func open(modelPath: String) {
        closeSession()
        state = .loading
        let name = (modelPath as NSString).lastPathComponent
        let s = settings
        queue.async { [weak self] in
            var p = bmoe_session_params_default()
            p.n_threads = Int32(s.nThreads)
            p.n_ctx = Int32(s.nCtx)
            p.chat = true
            p.moe_stream = s.moeStream
            p.cache_mb = Int32(s.cacheMB)
            p.io_threads = Int32(s.ioThreads)
            p.overlap = s.overlap
            p.dense_weights = s.denseAnon ? BMOE_DENSE_ANON : BMOE_DENSE_MMAP
            var err = [CChar](repeating: 0, count: 512)
            let session = modelPath.withCString { cs -> OpaquePointer? in
                p.model_path = cs
                return bmoe_open(&p, &err, numericCast(err.count))
            }
            DispatchQueue.main.async {
                guard let self else { return }
                if let session {
                    self.session = session
                    self.state = .ready(model: name)
                    self.stats = String(format: "loaded in %.1fs — %@", bmoe_load_seconds(session),
                                        String(cString: bmoe_arch(session)))
                    self.watchMemoryPressure()
                } else {
                    self.state = .failed(String(cString: err))
                }
            }
        }
    }

    func generate(prompt: String) {
        guard case .ready(let model) = state, let session else { return }
        state = .generating(model: model)
        answer = ""
        reasoning = ""
        // A backgrounded app is suspended mid-generation; keeping the screen awake is the honest
        // alternative to pretending the run could continue behind the lock screen.
        UIApplication.shared.isIdleTimerDisabled = true

        let sink = TokenSink { [weak self] event in
            DispatchQueue.main.async {
                guard let self else { return }
                self.answer = event.text
                self.reasoning = event.reasoning
                let hit = event.cacheHitPct >= 0 ? String(format: ", cache %.0f%%", event.cacheHitPct) : ""
                self.liveTelemetry = String(format: "tok %d — %.0f ms, %.0f MiB flash%@",
                                            event.step, event.wallMs, event.readMiB, hit)
            }
        }
        let user = Unmanaged.passRetained(sink).toOpaque()
        let s = settings
        queue.async { [weak self] in
            var p = bmoe_generate_params_default()
            p.n_predict = Int32(s.nPredict)
            let ok = prompt.withCString { cs -> Bool in
                p.prompt = cs
                return bmoe_generate(session, &p, tokenTrampoline, user)
            }
            Unmanaged<TokenSink>.fromOpaque(user).release()
            let text = String(cString: bmoe_last_text(session))
            let thinking = String(cString: bmoe_last_reasoning(session))
            let error = String(cString: bmoe_last_error(session))
            var st = bmoe_stats()
            bmoe_last_stats(session, &st)
            DispatchQueue.main.async {
                guard let self else { return }
                UIApplication.shared.isIdleTimerDisabled = false
                if ok {
                    self.answer = text
                    self.reasoning = thinking
                    self.stats = String(format: "%d tokens at %.2f tok/s, %.0f MiB streamed%@",
                                        st.n_generated, st.tokens_per_second, st.moe_read_mib,
                                        st.cancelled ? " (stopped)" : "")
                    self.state = .ready(model: model)
                } else {
                    self.state = .failed(error)
                }
                if self.shrinkAfterGeneration {
                    self.shrinkAfterGeneration = false
                    self.shrinkCache()
                }
            }
        }
    }

    func cancel() {
        if let session { bmoe_cancel(session) } // thread-safe; generation stops at the next token
    }

    func closeSession() {
        pressure?.cancel()
        pressure = nil
        if let session {
            self.session = nil
            queue.async { bmoe_close(session) } // after any in-flight generate on the serial queue
        }
        state = .idle
    }

    /// Jetsam does not warn, but the memory-pressure source often fires first. Shrinking the
    /// expert cache is the one big lever the app holds at runtime — legal only between
    /// generations (session.h), so mid-generation the shrink is deferred to the end of the run.
    private func watchMemoryPressure() {
        pressure?.cancel()
        let src = DispatchSource.makeMemoryPressureSource(eventMask: [.warning, .critical], queue: .main)
        src.setEventHandler { [weak self] in
            guard let self else { return }
            if case .generating = self.state {
                self.shrinkAfterGeneration = true
            } else {
                self.shrinkCache()
            }
        }
        src.resume()
        pressure = src
    }

    private func shrinkCache() {
        guard let session else { return }
        // Halve, but never into the engine's pathological band (a budget below its floor only
        // churns — config.h): below 2× the floor, go straight to cache-off.
        let halved = settings.cacheMB / 2
        settings.cacheMB = halved >= 1500 ? halved : 0
        bmoe_set_cache_budget_mb(session, Int32(settings.cacheMB))
        stats = "memory pressure — cache lowered to \(settings.cacheMB) MiB"
    }
}
