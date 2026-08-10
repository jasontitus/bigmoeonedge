// The C ABI (bmoe/bmoe_c.h) over Session. A translation layer and nothing else: every default and
// every mapping decision lives in config.h / session.h, reached through the same RunConfig →
// SessionConfig path the CLI takes, so the C surface can never drift from the C++ one.
#include "bmoe/bmoe_c.h"

#include "bmoe/config.h"
#include "bmoe/session.h"

#include <cstring>
#include <memory>
#include <string>

using namespace bmoe;

struct bmoe_session {
    std::unique_ptr<Session> s;
    RunResult last;    // strings handed out by bmoe_last_* live here until the next generate/close
    std::string error; // last failure, "" after a success
};

extern "C" {

bmoe_session_params bmoe_session_params_default(void) {
    // Defaults come FROM the C++ config structs rather than being restated, so a default changed
    // there changes here in the same commit.
    const RunConfig rc;
    bmoe_session_params p;
    std::memset(&p, 0, sizeof(p));
    p.model_path = nullptr;
    p.n_threads = rc.n_threads;
    p.n_ctx = rc.n_ctx;
    p.chat = rc.chatml;
    p.n_expert_used = rc.n_expert_used;
    p.moe_stream = rc.moe.enabled;
    p.cache_mb = rc.moe.cache_mb;
    p.cache_auto = rc.moe.cache_auto;
    p.cache_ceil_mb = rc.moe.cache_ceil_mb;
    p.io_threads = rc.moe.io_threads;
    p.o_direct = rc.moe.o_direct;
    p.overlap = rc.moe.overlap;
    p.dense_weights = (bmoe_dense_mode) rc.moe.dense_weights;
    p.drop_cold_frac = rc.moe.drop_cold_frac;
    return p;
}

bmoe_generate_params bmoe_generate_params_default(void) {
    const GenerateRequest gr;
    bmoe_generate_params p;
    std::memset(&p, 0, sizeof(p));
    p.prompt = nullptr;
    p.n_predict = gr.n_predict;
    p.think = gr.think;
    p.clear_kv = gr.clear_kv;
    p.render_text = gr.render_text;
    return p;
}

static void copy_err(char * err, size_t err_len, const std::string & msg) {
    if (!err || err_len == 0) return;
    const size_t n = msg.size() < err_len - 1 ? msg.size() : err_len - 1;
    std::memcpy(err, msg.c_str(), n);
    err[n] = '\0';
}

bmoe_session * bmoe_open(const bmoe_session_params * params, char * err, size_t err_len) {
    if (!params || !params->model_path || !params->model_path[0]) {
        copy_err(err, err_len, "model_path is required");
        return nullptr;
    }
    RunConfig cfg;
    cfg.model_path = params->model_path;
    cfg.n_threads = params->n_threads;
    cfg.n_ctx = params->n_ctx;
    cfg.chatml = params->chat;
    cfg.n_expert_used = params->n_expert_used;
    cfg.moe.enabled = params->moe_stream;
    cfg.moe.cache_mb = params->cache_mb;
    cfg.moe.cache_auto = params->cache_auto;
    cfg.moe.cache_ceil_mb = params->cache_ceil_mb;
    cfg.moe.io_threads = params->io_threads;
    cfg.moe.o_direct = params->o_direct;
    cfg.moe.overlap = params->overlap;
    cfg.moe.dense_weights = (DenseWeightsMode) params->dense_weights;
    cfg.moe.drop_cold_frac = params->drop_cold_frac;

    // The same validation the CLI runs, so a bad combination fails here with the same message
    // instead of surfacing later as an engine error.
    if (ValidationResult v = validate(cfg); !v) {
        copy_err(err, err_len, v.error);
        return nullptr;
    }

    std::string error;
    auto s = Session::open(session_config_from(cfg), error);
    if (!s) {
        copy_err(err, err_len, error);
        return nullptr;
    }
    auto * out = new bmoe_session();
    out->s = std::move(s);
    return out;
}

bool bmoe_generate(bmoe_session * s, const bmoe_generate_params * params, bmoe_token_cb cb, void * user) {
    if (!s || !s->s || !params || !params->prompt) return false;
    GenerateRequest req;
    req.prompt = params->prompt;
    req.n_predict = params->n_predict;
    req.think = params->think;
    req.clear_kv = params->clear_kv;
    req.render_text = params->render_text;

    std::function<void(const TokenMetrics &)> on_token;
    if (cb) {
        on_token = [cb, user](const TokenMetrics & m) {
            bmoe_token t;
            t.step = m.step;
            t.wall_ms = m.wall_ms;
            t.io_ms = m.io_ms;
            t.read_mib = (double) m.read_bytes / (1024.0 * 1024.0);
            t.cache_hit_pct = m.cache_hit_pct;
            t.piece = m.piece.c_str();
            t.text = m.text.c_str();
            t.reasoning = m.reasoning.c_str();
            cb(&t, user);
        };
    }
    s->last = s->s->generate(req, on_token);
    s->error = s->last.ok ? "" : s->last.error;
    return s->last.ok;
}

void bmoe_cancel(bmoe_session * s) {
    if (s && s->s) s->s->cancel();
}

void bmoe_set_cache_budget_mb(bmoe_session * s, int mib) {
    if (s && s->s) s->s->set_cache_budget_mb(mib);
}

const char * bmoe_last_text(const bmoe_session * s) {
    return s ? s->last.generated_text.c_str() : "";
}

const char * bmoe_last_reasoning(const bmoe_session * s) {
    return s ? s->last.reasoning_text.c_str() : "";
}

const char * bmoe_last_error(const bmoe_session * s) {
    return s ? s->error.c_str() : "invalid session";
}

void bmoe_last_stats(const bmoe_session * s, bmoe_stats * out) {
    if (!out) return;
    std::memset(out, 0, sizeof(*out));
    if (!s) return;
    out->ok = s->last.ok;
    out->cancelled = s->last.cancelled;
    out->n_generated = s->last.summary.n_generated;
    out->tokens_per_second = s->last.summary.tokens_per_second;
    out->load_seconds = s->s ? s->s->load_seconds() : 0.0;
    out->prefill_seconds = s->last.summary.prefill_seconds;
    out->moe_read_mib = s->last.summary.moe_read_mib;
}

const char * bmoe_arch(const bmoe_session * s) {
    return s && s->s ? s->s->arch().c_str() : "";
}

int bmoe_n_ctx(const bmoe_session * s) {
    return s && s->s ? s->s->n_ctx() : 0;
}

double bmoe_load_seconds(const bmoe_session * s) {
    return s && s->s ? s->s->load_seconds() : 0.0;
}

void bmoe_close(bmoe_session * s) {
    delete s;
}

} // extern "C"
