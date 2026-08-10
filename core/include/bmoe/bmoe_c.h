// C ABI over the engine's Session, for embedders that cannot consume C++.
//
// The first such embedder is the iOS example app: Swift links the engine statically and talks to
// it through this header (via a bridging header), because iOS has no equivalent of the Android
// app's ProcessBuilder-around-bmoe-cli arrangement — the engine must live inside the app process.
// The surface is deliberately a subset of session.h: what a chat UI needs, nothing more. Anything
// beyond it (traces, CSV sinks, speculation) stays C++-side; an embedder that wants those links
// session.h directly.
//
// Conventions:
//   * Structs are obtained from their _default() function and then overridden field by field —
//     never zero-initialized by hand — so a field added later keeps its default in older callers.
//   * bmoe_generate() BLOCKS until the generation finishes (or bmoe_cancel() stops it): run it off
//     the UI thread. The token callback fires on that same blocked thread, once per token.
//   * String pointers handed to the token callback are valid only during that call; the strings
//     returned by bmoe_last_text()/bmoe_last_reasoning()/bmoe_last_error() stay valid until the
//     next bmoe_generate()/bmoe_close() on the same session. Copy out, don't keep.
//   * One generation at a time per session; bmoe_cancel() is the only call that is safe from
//     another thread while bmoe_generate() runs.
#pragma once

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bmoe_session bmoe_session;

// Dense (non-expert) weight policy. Mirrors bmoe::DenseWeightsMode — see bmoe/config.h for what
// each mode costs and when it wins. The Android-only pinned mode is not exposed here.
typedef enum bmoe_dense_mode {
    BMOE_DENSE_MMAP = 0, // leave mmap'd; the kernel serves and reclaims them (the A/B baseline)
    BMOE_DENSE_WARM = 1, // mmap'd but page-cached once at load
    BMOE_DENSE_ANON = 2, // copied into the engine's own anonymous buffers and rebound (default)
} bmoe_dense_mode;

// Fixed for the session's lifetime (bmoe_open). Defaults match the CLI's.
typedef struct bmoe_session_params {
    const char * model_path; // required; a split model is opened by its first shard
    int n_threads;
    int n_ctx;
    bool chat;         // wrap prompts in the model's own chat template
    int n_expert_used; // top-k override; 0 = the model's own count. Lossy but reproducible.
    bool moe_stream;   // stream routed experts from flash (the engine's reason to exist)
    int cache_mb;      // fixed expert-cache budget in MiB; 0 = no cache; ignored under cache_auto
    bool cache_auto;   // size the cache once at load from free memory instead of cache_mb
    int cache_ceil_mb; // cap on the auto budget (0 = uncapped)
    int io_threads;    // parallel expert read lanes
    bool o_direct;     // bypass the OS page cache (O_DIRECT; F_NOCACHE on Apple platforms)
    bool overlap;      // overlap expert reads with FFN compute (needs the fork's ready hook)
    bmoe_dense_mode dense_weights;
    float drop_cold_frac; // cache-aware expert dropping; 0 = off (lossless). See config.h.
} bmoe_session_params;

bmoe_session_params bmoe_session_params_default(void);

// Per-prompt request. Defaults match GenerateRequest's.
typedef struct bmoe_generate_params {
    const char * prompt;
    int n_predict;
    bool think;       // render the chat template with reasoning enabled
    bool clear_kv;    // true: independent prompt; false: continue the conversation's KV
    bool render_text; // build bmoe_token.text/.reasoning each token — O(n²) over a turn; a UI
                      // streaming the running answer needs it, a consumer of `piece` does not
} bmoe_generate_params;

bmoe_generate_params bmoe_generate_params_default(void);

// One generated token, delivered to the callback before the next decode starts. The telemetry
// subset a live UI panel shows; the full picture stays with the C++ metrics types.
typedef struct bmoe_token {
    int step;               // 1-based
    double wall_ms;         // this token's wall time
    double io_ms;           // flash read time inside it
    double read_mib;        // expert bytes pulled from flash for it
    double cache_hit_pct;   // cumulative; -1 without a cache
    const char * piece;     // this token's text alone (UTF-8 fragment; may split a code point)
    const char * text;      // full answer so far, reasoning stripped ("" unless render_text)
    const char * reasoning; // reasoning-so-far ("" unless the model is thinking and render_text)
} bmoe_token;

typedef void (*bmoe_token_cb)(const bmoe_token * tok, void * user);

// End-of-generation figures, copied out by bmoe_last_stats().
typedef struct bmoe_stats {
    bool ok;
    bool cancelled; // stopped by bmoe_cancel(); ok stays true and the partial text stands
    int n_generated;
    double tokens_per_second; // decode only, as everywhere in this project
    double load_seconds;      // measured once at open; repeated here for convenience
    double prefill_seconds;
    double moe_read_mib; // flash bytes the whole generation streamed
} bmoe_stats;

// Load the model and initialise streaming. Returns NULL on failure with a message in err
// (truncated to err_len, always NUL-terminated; pass NULL/0 to skip it). Opening is the expensive
// step — tens of seconds for a large model — and is done once; generations reuse the session.
bmoe_session * bmoe_open(const bmoe_session_params * params, char * err, size_t err_len);

// Generate one response (blocking; see header comment). cb may be NULL. Returns false on error —
// bmoe_last_error() says why. A cancelled generation returns true with stats.cancelled set.
bool bmoe_generate(bmoe_session * s, const bmoe_generate_params * params, bmoe_token_cb cb, void * user);

// Ask the in-flight generation to stop at the next decode boundary. Thread-safe.
void bmoe_cancel(bmoe_session * s);

// Set the expert-cache budget in MiB and evict down to it now. Between generations only — the
// natural caller is an app's memory-pressure handler. No-op when the cache is off.
void bmoe_set_cache_budget_mb(bmoe_session * s, int mib);

const char * bmoe_last_text(const bmoe_session * s);      // final answer of the last generation
const char * bmoe_last_reasoning(const bmoe_session * s); // its reasoning span ("" if none)
const char * bmoe_last_error(const bmoe_session * s);     // "" when the last call succeeded
void bmoe_last_stats(const bmoe_session * s, bmoe_stats * out);

const char * bmoe_arch(const bmoe_session * s); // "qwen3moe", "deepseek4", …
int bmoe_n_ctx(const bmoe_session * s);
double bmoe_load_seconds(const bmoe_session * s);

void bmoe_close(bmoe_session * s); // NULL-safe

#ifdef __cplusplus
} // extern "C"
#endif
