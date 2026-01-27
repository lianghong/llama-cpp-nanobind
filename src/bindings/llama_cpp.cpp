#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/pair.h>

#include "llama.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

struct ModelParams {
    llama_model_params raw;
    ModelParams() : raw(llama_model_default_params()) {}
};

struct ContextParams {
    llama_context_params raw;
    ContextParams() : raw(llama_context_default_params()) {}
};

// Thread-safe backend initialization with reference counting
static std::once_flag g_backend_init_flag;
static std::atomic<int> g_model_count{0};
static std::mutex g_init_mutex;

static void init_backend() {
    llama_backend_init();
}

class Model {
public:
    explicit Model(const std::string &path, const ModelParams &params) {
        std::lock_guard<std::mutex> lock(g_init_mutex);
        std::call_once(g_backend_init_flag, init_backend);
        model_ = llama_model_load_from_file(path.c_str(), params.raw);
        if (!model_) {
            throw std::runtime_error("failed to load model: " + path);
        }
        ++g_model_count;
    }

    ~Model() {
        close();
    }

    void close() {
        if (model_) {
            llama_model_free(model_);
            model_ = nullptr;
            --g_model_count;
        }
    }

    Model(const Model &) = delete;
    Model & operator=(const Model &) = delete;

    const llama_model * get() const {
        if (!model_) {
            throw std::runtime_error("model is null (already freed or failed to load)");
        }
        return model_;
    }
    llama_model * get() {
        if (!model_) {
            throw std::runtime_error("model is null (already freed or failed to load)");
        }
        return model_;
    }

    const llama_vocab * vocab() const {
        if (!model_) {
            throw std::runtime_error("model is null (already freed or failed to load)");
        }
        return llama_model_get_vocab(model_);
    }

    int32_t n_vocab() const {
        return llama_vocab_n_tokens(vocab());
    }

    int32_t n_ctx_train() const {
        if (!model_) {
            throw std::runtime_error("model is null (already freed or failed to load)");
        }
        return llama_model_n_ctx_train(model_);
    }

    uint64_t model_size() const {
        if (!model_) {
            throw std::runtime_error("model is null (already freed or failed to load)");
        }
        return llama_model_size(model_);
    }

    uint64_t n_params() const {
        if (!model_) {
            throw std::runtime_error("model is null (already freed or failed to load)");
        }
        return llama_model_n_params(model_);
    }

    int32_t n_layer() const {
        if (!model_) {
            throw std::runtime_error("model is null (already freed or failed to load)");
        }
        return llama_model_n_layer(model_);
    }

    int32_t n_head() const {
        if (!model_) {
            throw std::runtime_error("model is null (already freed or failed to load)");
        }
        return llama_model_n_head(model_);
    }

    bool has_encoder() const {
        if (!model_) {
            throw std::runtime_error("model is null (already freed or failed to load)");
        }
        return llama_model_has_encoder(model_);
    }

    bool has_decoder() const {
        if (!model_) {
            throw std::runtime_error("model is null (already freed or failed to load)");
        }
        return llama_model_has_decoder(model_);
    }

    bool is_recurrent() const {
        if (!model_) {
            throw std::runtime_error("model is null (already freed or failed to load)");
        }
        return llama_model_is_recurrent(model_);
    }

    std::string chat_template(const std::string &name = "") const {
        if (!model_) {
            throw std::runtime_error("model is null (already freed or failed to load)");
        }
        const char *tmpl = llama_model_chat_template(model_, name.empty() ? nullptr : name.c_str());
        return tmpl ? std::string(tmpl) : "";
    }

    std::string desc() const {
        if (!model_) {
            throw std::runtime_error("model is null (already freed or failed to load)");
        }
        // Query required size first to avoid buffer overflow
        int32_t needed = llama_model_desc(model_, nullptr, 0);
        if (needed <= 0) {
            return "";  // Empty description
        }
        std::string buf(static_cast<size_t>(needed) + 1, '\0');
        llama_model_desc(model_, buf.data(), static_cast<int32_t>(buf.size()));
        // Remove null terminator if present
        if (!buf.empty() && buf.back() == '\0') {
            buf.pop_back();
        }
        return buf;
    }

    llama_token bos() const { return llama_vocab_bos(vocab()); }
    llama_token eos() const { return llama_vocab_eos(vocab()); }
    llama_token eot() const { return llama_vocab_eot(vocab()); }

    std::string token_to_piece(llama_token token) const {
        const char *text = llama_vocab_get_text(vocab(), token);
        return text ? std::string(text) : "";
    }

    int32_t n_embd() const {
        if (!model_) {
            throw std::runtime_error("model is null (already freed or failed to load)");
        }
        return llama_model_n_embd(model_);
    }

    // Metadata access
    int32_t meta_count() const {
        if (!model_) {
            throw std::runtime_error("model is null (already freed or failed to load)");
        }
        return llama_model_meta_count(model_);
    }

    std::string meta_val_str(const std::string &key) const {
        if (!model_) {
            throw std::runtime_error("model is null (already freed or failed to load)");
        }
        // First call to get required length
        int32_t len = llama_model_meta_val_str(model_, key.c_str(), nullptr, 0);
        if (len < 0) return "";
        std::string buf(static_cast<size_t>(len) + 1, '\0');
        llama_model_meta_val_str(model_, key.c_str(), buf.data(), static_cast<int32_t>(buf.size()));
        buf.resize(static_cast<size_t>(len));
        return buf;
    }

    std::string meta_key_by_index(int32_t i) const {
        if (!model_) {
            throw std::runtime_error("model is null (already freed or failed to load)");
        }
        int32_t len = llama_model_meta_key_by_index(model_, i, nullptr, 0);
        if (len < 0) return "";
        std::string buf(static_cast<size_t>(len) + 1, '\0');
        llama_model_meta_key_by_index(model_, i, buf.data(), static_cast<int32_t>(buf.size()));
        buf.resize(static_cast<size_t>(len));
        return buf;
    }

    std::string meta_val_by_index(int32_t i) const {
        if (!model_) {
            throw std::runtime_error("model is null (already freed or failed to load)");
        }
        int32_t len = llama_model_meta_val_str_by_index(model_, i, nullptr, 0);
        if (len < 0) return "";
        std::string buf(static_cast<size_t>(len) + 1, '\0');
        llama_model_meta_val_str_by_index(model_, i, buf.data(), static_cast<int32_t>(buf.size()));
        buf.resize(static_cast<size_t>(len));
        return buf;
    }

    std::vector<llama_token> tokenize(const std::string &text, bool add_special, bool parse_special) const {
        // Clamp to prevent integer overflow on very large inputs
        constexpr size_t MAX_TEXT_SIZE = static_cast<size_t>(INT32_MAX);
        if (text.size() > MAX_TEXT_SIZE) {
            throw std::runtime_error("input text too large for tokenization (exceeds INT32_MAX)");
        }
        constexpr size_t MAX_TOKENS = 1 << 24;  // 16M tokens max
        size_t estimated = text.size() + 8;
        if (estimated > MAX_TOKENS) {
            throw std::runtime_error("input text too large for tokenization");
        }
        int32_t max_tokens = static_cast<int32_t>(std::max(estimated, size_t{32}));
        std::vector<llama_token> tokens(static_cast<size_t>(max_tokens));
        int32_t n = llama_tokenize(vocab(), text.c_str(), static_cast<int32_t>(text.size()), tokens.data(), max_tokens, add_special, parse_special);
        if (n < 0) {
            max_tokens = -n;
            tokens.assign(static_cast<size_t>(max_tokens), 0);
            n = llama_tokenize(vocab(), text.c_str(), static_cast<int32_t>(text.size()), tokens.data(), max_tokens, add_special, parse_special);
        }
        if (n < 0) {
            throw std::runtime_error("tokenization failed");
        }
        tokens.resize(static_cast<size_t>(n));
        return tokens;
    }

    std::string detokenize(const std::vector<llama_token> &tokens, bool remove_special, bool unparse_special) const {
        if (tokens.size() > static_cast<size_t>(INT32_MAX)) {
            throw std::runtime_error("too many tokens for detokenization");
        }
        int32_t n_tokens = static_cast<int32_t>(tokens.size());
        int32_t needed = llama_detokenize(vocab(), tokens.data(), n_tokens, nullptr, 0, remove_special, unparse_special);
        if (needed < 0) {
            needed = -needed;
        }
        std::string out;
        out.resize(static_cast<size_t>(needed));
        int32_t written = llama_detokenize(vocab(), tokens.data(), n_tokens, out.data(), needed, remove_special, unparse_special);
        if (written < 0) {
            throw std::runtime_error("detokenize failed");
        }
        out.resize(static_cast<size_t>(written));
        return out;
    }

private:
    llama_model * model_ = nullptr;
};

class Context;
class LoraAdapter;

class SamplerChain {
public:
    struct Params {
        int32_t top_k = 40;
        float   top_p = 0.95f;
        float   min_p = 0.0f;
        size_t  min_keep = 1;
        float   temp = 0.8f;
        int32_t penalty_last_n = 64;
        float   repeat_penalty = 1.1f;
        float   freq_penalty = 0.0f;
        float   presence_penalty = 0.0f;
        int32_t seed = -1;
    };

    SamplerChain(const Model &model, const Params &params) {
        auto chain_params = llama_sampler_chain_default_params();
        sampler_ = llama_sampler_chain_init(chain_params);
        if (!sampler_) {
            throw std::runtime_error("failed to create sampler chain");
        }

        if (params.penalty_last_n != 0 || params.repeat_penalty != 1.0f ||
            params.freq_penalty != 0.0f || params.presence_penalty != 0.0f) {
            llama_sampler * penalties = llama_sampler_init_penalties(
                params.penalty_last_n,
                params.repeat_penalty,
                params.freq_penalty,
                params.presence_penalty);
            llama_sampler_chain_add(sampler_, penalties);
        }

        if (params.top_k > 0) {
            llama_sampler_chain_add(sampler_, llama_sampler_init_top_k(params.top_k));
        }
        if (params.top_p < 1.0f) {
            llama_sampler_chain_add(sampler_, llama_sampler_init_top_p(params.top_p, params.min_keep));
        }
        if (params.min_p > 0.0f) {
            llama_sampler_chain_add(sampler_, llama_sampler_init_min_p(params.min_p, params.min_keep));
        }
        if (params.temp != 1.0f) {
            llama_sampler_chain_add(sampler_, llama_sampler_init_temp(params.temp));
        }

        uint32_t rng_seed = params.seed >= 0
            ? static_cast<uint32_t>(params.seed)
            : static_cast<uint32_t>(llama_time_us() & 0xFFFFFFFF);
        llama_sampler_chain_add(sampler_, llama_sampler_init_dist(rng_seed));

        llama_token bos = llama_vocab_bos(model.vocab());
        if (bos != LLAMA_TOKEN_NULL) {
            llama_sampler_accept(sampler_, bos);
        }
    }

    ~SamplerChain() {
        if (sampler_) {
            llama_sampler_free(sampler_);
            sampler_ = nullptr;
        }
    }

    SamplerChain(const SamplerChain &) = delete;
    SamplerChain & operator=(const SamplerChain &) = delete;

    void reset() {
        if (sampler_) {
            llama_sampler_reset(sampler_);
        }
    }

    llama_sampler * get() const {
        if (!sampler_) {
            throw std::runtime_error("sampler is null (not initialized)");
        }
        return sampler_;
    }

    llama_token sample(Context &ctx, int32_t idx);

private:
    llama_sampler * sampler_ = nullptr;
};

class Context {
public:
    Context(Model &model_ref, const ContextParams &params)
        : model_(&model_ref), params_(params) {
        ctx_ = llama_init_from_model(model_->get(), params_.raw);
        if (!ctx_) {
            throw std::runtime_error("failed to create llama context");
        }
        cur_pos_ = 0;
        // Pre-allocate single-token batch for decode_one to avoid per-token allocations
        single_batch_ = llama_batch_init(1, 0, 1);
        llama_set_n_threads(ctx_, params_.raw.n_threads, params_.raw.n_threads_batch);
    }

    ~Context() {
        close();
    }

    void close() {
        if (single_batch_.token) {
            llama_batch_free(single_batch_);
            single_batch_ = {};
        }
        if (ctx_) {
            llama_free(ctx_);
            ctx_ = nullptr;
        }
        model_ = nullptr;
    }

    Context(const Context &) = delete;
    Context & operator=(const Context &) = delete;

    int32_t n_ctx() const {
        if (!ctx_) return 0;
        return llama_n_ctx(ctx_);
    }

    void set_thread_count(int32_t n_threads, int32_t n_threads_batch) {
        if (!ctx_) return;
        llama_set_n_threads(ctx_, n_threads, n_threads_batch);
        params_.raw.n_threads = n_threads;
        params_.raw.n_threads_batch = n_threads_batch;
    }

    void reset() {
        if (!model_) {
            throw std::runtime_error("context has been closed");
        }
        if (ctx_) {
            llama_free(ctx_);
        }
        ctx_ = llama_init_from_model(model_->get(), params_.raw);
        if (!ctx_) {
            throw std::runtime_error("failed to recreate llama context");
        }
        llama_set_n_threads(ctx_, params_.raw.n_threads, params_.raw.n_threads_batch);
        cur_pos_ = 0;
        // Reinitialize single-token batch if needed
        if (!single_batch_.token) {
            single_batch_ = llama_batch_init(1, 0, 1);
        }
    }

    void decode(const std::vector<llama_token> &tokens, bool return_logits = true) {
        if (!ctx_) throw std::runtime_error("context has been closed");
        if (tokens.empty()) return;
        llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
        batch.n_tokens = (int32_t) tokens.size();
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            batch.token[i]    = tokens[static_cast<size_t>(i)];
            batch.pos[i]      = cur_pos_ + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0]= 0;
            batch.logits[i]   = (return_logits && i == batch.n_tokens - 1) ? 1 : 0;
        }
        int32_t rc = llama_decode(ctx_, batch);
        llama_batch_free(batch);
        if (rc < 0) {
            throw std::runtime_error("llama_decode failed with code " + std::to_string(rc));
        }
        cur_pos_ += (int32_t) tokens.size();
    }

    void decode_one(llama_token token, bool request_logits = true) {
        if (!ctx_) throw std::runtime_error("context has been closed");
        // Reuse pre-allocated single-token batch to avoid per-token allocations
        single_batch_.n_tokens = 1;
        single_batch_.token[0]    = token;
        single_batch_.pos[0]      = cur_pos_;
        single_batch_.n_seq_id[0] = 1;
        single_batch_.seq_id[0][0]= 0;
        single_batch_.logits[0]   = request_logits ? 1 : 0;
        int32_t rc = llama_decode(ctx_, single_batch_);
        if (rc < 0) {
            throw std::runtime_error("llama_decode (single) failed with code " + std::to_string(rc));
        }
        ++cur_pos_;
    }

    std::vector<float> logits() const {
        if (!ctx_ || !model_) {
            throw std::runtime_error("context has been closed");
        }
        const int32_t n_vocab = model_->n_vocab();
        float * ptr = llama_get_logits(const_cast<llama_context *>(ctx_));
        if (!ptr) {
            throw std::runtime_error("logits unavailable; ensure decode was called with logits enabled");
        }
        return std::vector<float>(ptr, ptr + n_vocab);
    }

    std::vector<float> embeddings() const {
        if (!ctx_ || !model_) {
            throw std::runtime_error("context has been closed");
        }
        float * ptr = llama_get_embeddings(const_cast<llama_context *>(ctx_));
        if (!ptr) {
            throw std::runtime_error("embeddings unavailable; ensure pooling_type is set and decode was called");
        }
        const int32_t n_embd = llama_model_n_embd(model_->get());
        return std::vector<float>(ptr, ptr + n_embd);
    }

    llama_token generate_next(SamplerChain &sampler, int32_t idx = -1) {
        return sampler.sample(*this, idx);
    }

    Model & model() const {
        if (!model_) {
            throw std::runtime_error("context has been closed");
        }
        return *model_;
    }
    
    llama_context * raw() const {
        if (!ctx_) {
            throw std::runtime_error("context is null (already freed or failed to initialize)");
        }
        return ctx_;
    }

    // State save/load
    bool save_state(const std::string &path) {
        if (!ctx_) {
            throw std::runtime_error("context has been closed");
        }
        return llama_state_save_file(ctx_, path.c_str(), nullptr, 0);
    }

    size_t load_state(const std::string &path) {
        if (!ctx_) throw std::runtime_error("context has been closed");
        size_t n_token_count = 0;
        bool ok = llama_state_load_file(ctx_, path.c_str(), nullptr, 0, &n_token_count);
        if (!ok) {
            throw std::runtime_error("failed to load state from: " + path);
        }
        // Update cur_pos_ from KV cache to maintain correct position bookkeeping
        cur_pos_ = kv_cache_seq_pos_max(0) + 1;
        if (cur_pos_ < 0) cur_pos_ = 0;
        return n_token_count;
    }

    std::vector<uint8_t> get_state_data() {
        if (!ctx_) {
            throw std::runtime_error("context has been closed");
        }
        size_t size = llama_state_get_size(ctx_);
        std::vector<uint8_t> data(size);
        size_t written = llama_state_get_data(ctx_, data.data(), size);
        data.resize(written);
        return data;
    }

    size_t set_state_data(const std::vector<uint8_t> &data) {
        if (!ctx_) {
            throw std::runtime_error("context has been closed");
        }
        size_t result = llama_state_set_data(ctx_, data.data(), data.size());
        // Update cur_pos_ from KV cache to maintain correct position bookkeeping
        cur_pos_ = kv_cache_seq_pos_max(0) + 1;
        if (cur_pos_ < 0) cur_pos_ = 0;
        return result;
    }

    // LoRA adapter management - defined after LoraAdapter class
    int32_t set_lora(LoraAdapter &adapter, float scale = 1.0f);
    int32_t remove_lora(LoraAdapter &adapter);

    void clear_lora() {
        if (!ctx_) return;
        llama_clear_adapter_lora(ctx_);
    }

    // Performance metrics
    nb::dict perf() const {
        nb::dict d;
        if (!ctx_) return d;
        auto data = llama_perf_context(ctx_);
        d["t_start_ms"] = data.t_start_ms;
        d["t_load_ms"] = data.t_load_ms;
        d["t_p_eval_ms"] = data.t_p_eval_ms;
        d["t_eval_ms"] = data.t_eval_ms;
        d["n_p_eval"] = data.n_p_eval;
        d["n_eval"] = data.n_eval;
        return d;
    }

    void perf_reset() {
        if (!ctx_) return;
        llama_perf_context_reset(ctx_);
    }

    // KV cache / memory sequence management
    void kv_cache_clear() {
        if (!ctx_) return;
        llama_memory_t mem = llama_get_memory(ctx_);
        llama_memory_seq_rm(mem, -1, -1, -1);
        cur_pos_ = 0;
    }

    bool kv_cache_seq_rm(int32_t seq_id, int32_t p0 = -1, int32_t p1 = -1) {
        if (!ctx_) return false;
        llama_memory_t mem = llama_get_memory(ctx_);
        bool result = llama_memory_seq_rm(mem, seq_id, p0, p1);
        // Update cur_pos_ if we modified sequence 0 (the default sequence)
        if (result && (seq_id == 0 || seq_id == -1)) {
            int32_t new_pos = llama_memory_seq_pos_max(mem, 0) + 1;
            cur_pos_ = new_pos < 0 ? 0 : new_pos;
        }
        return result;
    }

    void kv_cache_seq_cp(int32_t seq_id_src, int32_t seq_id_dst, int32_t p0 = -1, int32_t p1 = -1) {
        if (!ctx_) return;
        llama_memory_t mem = llama_get_memory(ctx_);
        llama_memory_seq_cp(mem, seq_id_src, seq_id_dst, p0, p1);
    }

    void kv_cache_seq_keep(int32_t seq_id) {
        if (!ctx_) return;
        llama_memory_t mem = llama_get_memory(ctx_);
        llama_memory_seq_keep(mem, seq_id);
        // Update cur_pos_ based on what remains in sequence 0
        int32_t new_pos = llama_memory_seq_pos_max(mem, 0) + 1;
        cur_pos_ = new_pos < 0 ? 0 : new_pos;
    }

    void kv_cache_seq_add(int32_t seq_id, int32_t p0, int32_t p1, int32_t delta) {
        if (!ctx_) return;
        llama_memory_t mem = llama_get_memory(ctx_);
        llama_memory_seq_add(mem, seq_id, p0, p1, delta);
        // Update cur_pos_ if we modified sequence 0
        if (seq_id == 0) {
            int32_t new_pos = llama_memory_seq_pos_max(mem, 0) + 1;
            cur_pos_ = new_pos < 0 ? 0 : new_pos;
        }
    }

    int32_t kv_cache_seq_pos_max(int32_t seq_id = 0) {
        if (!ctx_) return -1;
        llama_memory_t mem = llama_get_memory(ctx_);
        return llama_memory_seq_pos_max(mem, seq_id);
    }

private:
    Model * model_ = nullptr;
    llama_context * ctx_ = nullptr;
    ContextParams params_;
    int32_t cur_pos_ = 0;
    llama_batch single_batch_ = {};  // Reusable single-token batch for decode_one
};

inline llama_token SamplerChain::sample(Context &ctx, int32_t idx) {
    if (!sampler_) {
        throw std::runtime_error("sampler not initialized");
    }
    return llama_sampler_sample(sampler_, ctx.raw(), idx);
}

std::vector<llama_token> generate_tokens(Context &ctx,
                                         SamplerChain &sampler,
                                         const std::vector<llama_token> &prompt,
                                         int32_t max_new_tokens,
                                         bool add_bos,
                                         llama_token eos_token,
                                         const std::vector<llama_token> &stop_tokens) {
    std::vector<llama_token> output;
    output.reserve(static_cast<size_t>(max_new_tokens));

    std::vector<llama_token> priming = prompt;
    if (add_bos && (priming.empty() || priming.front() != ctx.model().bos())) {
        priming.insert(priming.begin(), ctx.model().bos());
    }

    // Accept prompt tokens into sampler for penalty tracking
    for (llama_token t : priming) {
        llama_sampler_accept(sampler.get(), t);
    }

    if (!priming.empty()) {
        ctx.decode(priming, /*return_logits=*/true);
    }

    for (int i = 0; i < max_new_tokens; ++i) {
        llama_token token = ctx.generate_next(sampler, -1);
        llama_sampler_accept(sampler.get(), token);
        if (token == eos_token || token == LLAMA_TOKEN_NULL) {
            break;
        }
        if (!stop_tokens.empty() &&
            std::ranges::find(stop_tokens, token) != stop_tokens.end()) {
            break;
        }
        output.push_back(token);
        ctx.decode_one(token, /*request_logits=*/true);
    }
    return output;
}

// Logging control -----------------------------------------------------------
static ggml_log_level g_min_log_level = GGML_LOG_LEVEL_INFO;

static void log_filter_bridge(ggml_log_level level, const char * text, void * /*user*/) {
    if (level < g_min_log_level) return;
    std::fputs(text, stderr);
    std::fflush(stderr);
}

void set_log_level(int min_level) {
    g_min_log_level = static_cast<ggml_log_level>(min_level);
    llama_log_set(log_filter_bridge, nullptr);
}

void disable_logging() {
    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
}

void reset_logging() {
    llama_log_set(nullptr, nullptr);
}

// Chat template helper
std::string chat_apply_template(
    [[maybe_unused]] const Model &model,
    const std::vector<std::pair<std::string, std::string>> &messages,
    const std::string &tmpl,
    bool add_generation_prompt) {

    std::vector<llama_chat_message> chat_msgs;
    chat_msgs.reserve(messages.size());
    for (const auto &m : messages) {
        chat_msgs.push_back({m.first.c_str(), m.second.c_str()});
    }

    const char *tmpl_ptr = tmpl.empty() ? nullptr : tmpl.c_str();

    // First call to get required size
    int32_t needed = llama_chat_apply_template(
        tmpl_ptr,
        chat_msgs.data(),
        chat_msgs.size(),
        add_generation_prompt,
        nullptr,
        0);

    if (needed < 0) {
        throw std::runtime_error("llama_chat_apply_template failed");
    }

    std::string result(static_cast<size_t>(needed) + 1, '\0');
    int32_t written = llama_chat_apply_template(
        tmpl_ptr,
        chat_msgs.data(),
        chat_msgs.size(),
        add_generation_prompt,
        result.data(),
        static_cast<int32_t>(result.size()));

    if (written < 0) {
        throw std::runtime_error("llama_chat_apply_template failed on second call");
    }
    result.resize(static_cast<size_t>(written));
    return result;
}

// Grammar sampler wrapper
class GrammarSampler {
public:
    GrammarSampler(const Model &model, const std::string &grammar_str, const std::string &grammar_root) {
        sampler_ = llama_sampler_init_grammar(model.vocab(), grammar_str.c_str(), grammar_root.c_str());
        if (!sampler_) {
            throw std::runtime_error("failed to create grammar sampler - check grammar syntax");
        }
    }

    ~GrammarSampler() {
        if (sampler_) {
            llama_sampler_free(sampler_);
            sampler_ = nullptr;
        }
    }

    GrammarSampler(const GrammarSampler &) = delete;
    GrammarSampler &operator=(const GrammarSampler &) = delete;

    llama_sampler *get() const { return sampler_; }

    void accept(llama_token token) {
        if (sampler_) {
            llama_sampler_accept(sampler_, token);
        }
    }

    void reset() {
        if (sampler_) {
            llama_sampler_reset(sampler_);
        }
    }

private:
    llama_sampler *sampler_ = nullptr;
};

// LoRA adapter wrapper
class LoraAdapter {
public:
    LoraAdapter(Model &model, const std::string &path) {
        adapter_ = llama_adapter_lora_init(model.get(), path.c_str());
        if (!adapter_) {
            throw std::runtime_error("failed to load LoRA adapter: " + path);
        }
    }

    // Note: adapters are freed automatically with the associated model (llama.cpp API change)
    ~LoraAdapter() = default;

    LoraAdapter(const LoraAdapter &) = delete;
    LoraAdapter &operator=(const LoraAdapter &) = delete;

    llama_adapter_lora *get() const { return adapter_; }

private:
    llama_adapter_lora *adapter_ = nullptr;
};

// Context LoRA methods (defined after LoraAdapter)
inline int32_t Context::set_lora(LoraAdapter &adapter, float scale) {
    if (!ctx_) return -1;
    return llama_set_adapter_lora(ctx_, adapter.get(), scale);
}

inline int32_t Context::remove_lora(LoraAdapter &adapter) {
    if (!ctx_) return -1;
    return llama_rm_adapter_lora(ctx_, adapter.get());
}

struct TokenProb {
    llama_token token;
    float logprob;
    std::vector<std::pair<llama_token, float>> top_logprobs;
};

// softmax helpers
inline double logsumexp(const float *logits, int32_t n_vocab) {
    float max_l = -std::numeric_limits<float>::infinity();
    for (int32_t i = 0; i < n_vocab; ++i) {
        max_l = std::max(max_l, logits[i]);
    }
    double sum = 0.0;
    for (int32_t i = 0; i < n_vocab; ++i) {
        sum += std::exp(double(logits[i] - max_l));
    }
    return std::log(sum) + double(max_l);
}

inline std::vector<std::pair<llama_token, float>> compute_top_logprobs(const float *logits, int32_t n_vocab, int32_t top_n, double lse) {
    if (top_n <= 0) return {};
    std::vector<llama_token> idx(static_cast<size_t>(n_vocab));
    std::iota(idx.begin(), idx.end(), 0);
    if (top_n < n_vocab) {
        std::partial_sort(idx.begin(), idx.begin() + top_n, idx.end(),
                          [&](llama_token a, llama_token b) { return logits[a] > logits[b]; });
        idx.resize(static_cast<size_t>(top_n));
    }
    std::vector<std::pair<llama_token, float>> out;
    out.reserve(idx.size());
    for (auto t : idx) {
        float lp = static_cast<float>(double(logits[t]) - lse);
        out.emplace_back(t, lp);
    }
    return out;
}

std::vector<TokenProb> generate_tokens_with_details(
    Context &ctx,
    SamplerChain &sampler,
    const std::vector<llama_token> &prompt,
    int32_t max_new_tokens,
    bool add_bos,
    llama_token eos_token,
    const std::vector<std::vector<llama_token>> &stop_sequences,
    int32_t top_logprobs,
    bool echo_prompt) {

    std::vector<TokenProb> results;
    std::vector<llama_token> priming = prompt;
    if (add_bos && (priming.empty() || priming.front() != ctx.model().bos())) {
        priming.insert(priming.begin(), ctx.model().bos());
    }

    // Accept prompt tokens into sampler for penalty tracking
    for (llama_token t : priming) {
        llama_sampler_accept(sampler.get(), t);
    }

    const size_t prompt_out_start = results.size();

    // process prompt
    if (!priming.empty()) {
        ctx.decode(priming, /*return_logits=*/true);
        if (echo_prompt) {
            for (size_t i = 0; i < priming.size(); ++i) {
                TokenProb tp;
                tp.token = priming[i];
                tp.logprob = std::numeric_limits<float>::quiet_NaN();
                results.push_back(std::move(tp));
            }
        }
    }

    std::vector<llama_token> generated;
    generated.reserve(static_cast<size_t>(max_new_tokens));

    for (int i = 0; i < max_new_tokens; ++i) {
        const float *logits = llama_get_logits(ctx.raw());
        if (!logits) {
            throw std::runtime_error("logits unavailable before sampling");
        }
        const int32_t n_vocab = ctx.model().n_vocab();

        // Build candidates and apply sampler to get adjusted probabilities
        std::vector<llama_token_data> candidates(static_cast<size_t>(n_vocab));
        for (int32_t j = 0; j < n_vocab; ++j) {
            candidates[static_cast<size_t>(j)] = {j, logits[j], 0.0f};
        }
        llama_token_data_array cur_p = {candidates.data(), static_cast<size_t>(n_vocab), -1, false};
        llama_sampler_apply(sampler.get(), &cur_p);

        // Compute logprobs from sampler-adjusted logits
        double lse = 0.0;
        {
            float max_l = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < cur_p.size; ++j) {
                max_l = std::max(max_l, cur_p.data[j].logit);
            }
            double sum = 0.0;
            for (size_t j = 0; j < cur_p.size; ++j) {
                sum += std::exp(double(cur_p.data[j].logit - max_l));
            }
            lse = std::log(sum) + double(max_l);
        }

        llama_token token = ctx.generate_next(sampler, -1);
        llama_sampler_accept(sampler.get(), token);

        // Find the adjusted logit for the sampled token
        float token_logit = logits[token];
        for (size_t j = 0; j < cur_p.size; ++j) {
            if (cur_p.data[j].id == token) {
                token_logit = cur_p.data[j].logit;
                break;
            }
        }

        TokenProb tp;
        tp.token = token;
        tp.logprob = static_cast<float>(double(token_logit) - lse);
        if (top_logprobs > 0) {
            std::vector<std::pair<llama_token, float>> top_lp;
            std::vector<size_t> idx(cur_p.size);
            std::iota(idx.begin(), idx.end(), 0);
            size_t n = std::min(static_cast<size_t>(top_logprobs), cur_p.size);
            std::partial_sort(idx.begin(), idx.begin() + static_cast<long>(n), idx.end(),
                              [&](size_t a, size_t b) { return cur_p.data[a].logit > cur_p.data[b].logit; });
            for (size_t j = 0; j < n; ++j) {
                float lp = static_cast<float>(double(cur_p.data[idx[j]].logit) - lse);
                top_lp.emplace_back(cur_p.data[idx[j]].id, lp);
            }
            tp.top_logprobs = std::move(top_lp);
        }
        results.push_back(std::move(tp));

        generated.push_back(token);

        // stop on EOS / NULL (do not emit the token)
        if ((token == eos_token || token == LLAMA_TOKEN_NULL) && generated.size() >= 1) {
            results.pop_back();
            generated.pop_back();
            break;
        }

        // stop sequence check on generated tokens
        bool matched_stop = false;
        size_t remove_n = 0;
        for (const auto &seq : stop_sequences) {
            if (seq.empty() || seq.size() > generated.size()) continue;
            if (std::equal(seq.rbegin(), seq.rend(), generated.rbegin())) {
                matched_stop = true;
                remove_n = seq.size();
                break;
            }
        }
        if (matched_stop) {
            // remove stop tokens from output (but never remove echoed prompt)
            for (size_t j = 0; j < remove_n && !generated.empty(); ++j) {
                generated.pop_back();
                if (results.size() > prompt_out_start) {
                    results.pop_back();
                }
            }
            break;
        }

        ctx.decode_one(token, /*request_logits=*/true);
    }

    return results;
}

// Generation with grammar constraint
std::vector<llama_token> generate_tokens_with_grammar(
    Context &ctx,
    SamplerChain &sampler,
    GrammarSampler &grammar,
    const std::vector<llama_token> &prompt,
    int32_t max_new_tokens,
    bool add_bos,
    llama_token eos_token,
    const std::vector<llama_token> &stop_tokens) {

    std::vector<llama_token> output;
    output.reserve(static_cast<size_t>(max_new_tokens));

    std::vector<llama_token> priming = prompt;
    if (add_bos && (priming.empty() || priming.front() != ctx.model().bos())) {
        priming.insert(priming.begin(), ctx.model().bos());
    }

    // Accept prompt tokens into sampler for penalty tracking
    for (llama_token t : priming) {
        llama_sampler_accept(sampler.get(), t);
    }

    if (!priming.empty()) {
        ctx.decode(priming, /*return_logits=*/true);
    }

    const int32_t n_vocab = ctx.model().n_vocab();

    for (int i = 0; i < max_new_tokens; ++i) {
        float *logits = llama_get_logits(ctx.raw());
        if (!logits) {
            throw std::runtime_error("logits unavailable");
        }

        // Build token data array for grammar sampling
        std::vector<llama_token_data> candidates(static_cast<size_t>(n_vocab));
        for (int32_t j = 0; j < n_vocab; ++j) {
            candidates[static_cast<size_t>(j)] = {j, logits[j], 0.0f};
        }
        llama_token_data_array cur_p = {candidates.data(), static_cast<size_t>(n_vocab), -1, false};

        // Apply grammar constraint first (masks invalid tokens)
        llama_sampler_apply(grammar.get(), &cur_p);

        // Apply sampler chain (temperature, top_k, top_p, etc.) to grammar-filtered candidates
        llama_sampler_apply(sampler.get(), &cur_p);

        // Select token from the sampled distribution
        llama_token token = LLAMA_TOKEN_NULL;
        if (cur_p.size > 0 && cur_p.selected >= 0 && static_cast<size_t>(cur_p.selected) < cur_p.size) {
            token = cur_p.data[cur_p.selected].id;
        } else if (cur_p.size > 0) {
            // Fallback: pick highest probability after sampling
            float best_logit = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < cur_p.size; ++j) {
                if (cur_p.data[j].logit > best_logit) {
                    best_logit = cur_p.data[j].logit;
                    token = cur_p.data[j].id;
                }
            }
        }

        if (token == eos_token || token == LLAMA_TOKEN_NULL) {
            break;
        }
        if (!stop_tokens.empty()) {
            bool should_stop = false;
            for (auto st : stop_tokens) {
                if (token == st) { should_stop = true; break; }
            }
            if (should_stop) break;
        }

        // Accept token in grammar and sampler
        llama_sampler_accept(grammar.get(), token);
        llama_sampler_accept(sampler.get(), token);

        output.push_back(token);
        ctx.decode_one(token, /*request_logits=*/true);
    }
    return output;
}

// Generation with multi-token stop sequences (no grammar)
std::vector<llama_token> generate_tokens_multi_stop(
    Context &ctx,
    SamplerChain &sampler,
    const std::vector<llama_token> &prompt,
    int32_t max_new_tokens,
    bool add_bos,
    llama_token eos_token,
    const std::vector<std::vector<llama_token>> &stop_sequences) {

    std::vector<llama_token> output;
    output.reserve(static_cast<size_t>(max_new_tokens));

    std::vector<llama_token> priming = prompt;
    if (add_bos && (priming.empty() || priming.front() != ctx.model().bos())) {
        priming.insert(priming.begin(), ctx.model().bos());
    }

    // Accept prompt tokens into sampler for penalty tracking
    for (llama_token t : priming) {
        llama_sampler_accept(sampler.get(), t);
    }

    if (!priming.empty()) {
        ctx.decode(priming, /*return_logits=*/true);
    }

    for (int i = 0; i < max_new_tokens; ++i) {
        llama_token token = ctx.generate_next(sampler, -1);
        llama_sampler_accept(sampler.get(), token);
        if (token == eos_token || token == LLAMA_TOKEN_NULL) {
            break;
        }
        output.push_back(token);
        
        // Check multi-token stop sequences
        bool matched = false;
        for (const auto &seq : stop_sequences) {
            if (seq.empty() || seq.size() > output.size()) continue;
            if (std::equal(seq.rbegin(), seq.rend(), output.rbegin())) {
                matched = true;
                output.erase(output.end() - static_cast<long>(seq.size()), output.end());
                break;
            }
        }
        if (matched) break;
        
        ctx.decode_one(token, /*request_logits=*/true);
    }
    return output;
}

// Generation with grammar and multi-token stop sequences
std::vector<llama_token> generate_tokens_grammar_multi_stop(
    Context &ctx,
    SamplerChain &sampler,
    GrammarSampler &grammar,
    const std::vector<llama_token> &prompt,
    int32_t max_new_tokens,
    bool add_bos,
    llama_token eos_token,
    const std::vector<std::vector<llama_token>> &stop_sequences) {

    std::vector<llama_token> output;
    output.reserve(static_cast<size_t>(max_new_tokens));

    std::vector<llama_token> priming = prompt;
    if (add_bos && (priming.empty() || priming.front() != ctx.model().bos())) {
        priming.insert(priming.begin(), ctx.model().bos());
    }

    // Accept prompt tokens into sampler for penalty tracking
    for (llama_token t : priming) {
        llama_sampler_accept(sampler.get(), t);
    }

    if (!priming.empty()) {
        ctx.decode(priming, /*return_logits=*/true);
    }

    const int32_t n_vocab = ctx.model().n_vocab();

    for (int i = 0; i < max_new_tokens; ++i) {
        float *logits = llama_get_logits(ctx.raw());
        if (!logits) {
            throw std::runtime_error("logits unavailable");
        }

        std::vector<llama_token_data> candidates(static_cast<size_t>(n_vocab));
        for (int32_t j = 0; j < n_vocab; ++j) {
            candidates[static_cast<size_t>(j)] = {j, logits[j], 0.0f};
        }
        llama_token_data_array cur_p = {candidates.data(), static_cast<size_t>(n_vocab), -1, false};

        // Apply grammar constraint first (masks invalid tokens)
        llama_sampler_apply(grammar.get(), &cur_p);

        // Apply sampler chain (temperature, top_k, top_p, etc.) to grammar-filtered candidates
        llama_sampler_apply(sampler.get(), &cur_p);

        // Select token from the sampled distribution
        llama_token token = LLAMA_TOKEN_NULL;
        if (cur_p.size > 0 && cur_p.selected >= 0 && static_cast<size_t>(cur_p.selected) < cur_p.size) {
            token = cur_p.data[cur_p.selected].id;
        } else if (cur_p.size > 0) {
            // Fallback: pick highest probability after sampling
            float best_logit = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < cur_p.size; ++j) {
                if (cur_p.data[j].logit > best_logit) {
                    best_logit = cur_p.data[j].logit;
                    token = cur_p.data[j].id;
                }
            }
        }

        if (token == eos_token || token == LLAMA_TOKEN_NULL) {
            break;
        }

        llama_sampler_accept(grammar.get(), token);
        llama_sampler_accept(sampler.get(), token);
        output.push_back(token);

        // Check multi-token stop sequences
        bool matched = false;
        for (const auto &seq : stop_sequences) {
            if (seq.empty() || seq.size() > output.size()) continue;
            if (std::equal(seq.rbegin(), seq.rend(), output.rbegin())) {
                matched = true;
                output.erase(output.end() - static_cast<long>(seq.size()), output.end());
                break;
            }
        }
        if (matched) break;

        ctx.decode_one(token, /*request_logits=*/true);
    }
    return output;
}

// Streaming generation with callback - yields tokens as they're generated
// Returns total number of tokens generated
int32_t generate_tokens_streaming(
    Context &ctx,
    SamplerChain &sampler,
    const std::vector<llama_token> &prompt,
    int32_t max_new_tokens,
    bool add_bos,
    llama_token eos_token,
    const std::vector<std::vector<llama_token>> &stop_sequences,
    const std::function<bool(llama_token)> &callback) {

    std::vector<llama_token> output;
    output.reserve(static_cast<size_t>(max_new_tokens));

    std::vector<llama_token> priming = prompt;
    if (add_bos && (priming.empty() || priming.front() != ctx.model().bos())) {
        priming.insert(priming.begin(), ctx.model().bos());
    }

    for (llama_token t : priming) {
        llama_sampler_accept(sampler.get(), t);
    }

    if (!priming.empty()) {
        ctx.decode(priming, /*return_logits=*/true);
    }

    for (int i = 0; i < max_new_tokens; ++i) {
        llama_token token = ctx.generate_next(sampler, -1);
        llama_sampler_accept(sampler.get(), token);

        if (token == eos_token || token == LLAMA_TOKEN_NULL) {
            break;
        }

        output.push_back(token);

        // Check multi-token stop sequences
        bool matched = false;
        size_t remove_n = 0;
        for (const auto &seq : stop_sequences) {
            if (seq.empty() || seq.size() > output.size()) continue;
            if (std::equal(seq.rbegin(), seq.rend(), output.rbegin())) {
                matched = true;
                remove_n = seq.size();
                break;
            }
        }

        if (matched) {
            output.erase(output.end() - static_cast<long>(remove_n), output.end());
            break;
        }

        // Call Python callback with GIL acquired
        {
            nb::gil_scoped_acquire gil;
            if (!callback(token)) {
                break;  // Callback returned False, stop generation
            }
        }

        ctx.decode_one(token, /*request_logits=*/true);
    }
    return static_cast<int32_t>(output.size());
}

} // namespace

NB_MODULE(_llama, m) {
    m.doc() = "High-performance nanobind bindings for llama.cpp";

    nb::class_<ModelParams>(m, "ModelParams", "Parameters for loading a model")
        .def(nb::init<>())
        .def_prop_rw("n_gpu_layers", [](ModelParams &p){ return p.raw.n_gpu_layers; }, [](ModelParams &p, int32_t v){ p.raw.n_gpu_layers = v; }, "Number of layers to offload to GPU (-1 = all)")
        .def_prop_rw("main_gpu", [](ModelParams &p){ return p.raw.main_gpu; }, [](ModelParams &p, int32_t v){ p.raw.main_gpu = v; }, "Main GPU index for multi-GPU setups")
        .def_prop_rw("split_mode", [](ModelParams &p){ return p.raw.split_mode; }, [](ModelParams &p, int32_t v){ p.raw.split_mode = static_cast<llama_split_mode>(v); }, "How to split model across GPUs")
        .def_prop_rw("vocab_only", [](ModelParams &p){ return p.raw.vocab_only; }, [](ModelParams &p, bool v){ p.raw.vocab_only = v; }, "Load only vocabulary, no weights")
        .def_prop_rw("use_mmap", [](ModelParams &p){ return p.raw.use_mmap; }, [](ModelParams &p, bool v){ p.raw.use_mmap = v; }, "Use memory-mapped file for model")
        .def_prop_rw("use_mlock", [](ModelParams &p){ return p.raw.use_mlock; }, [](ModelParams &p, bool v){ p.raw.use_mlock = v; }, "Lock model in RAM")
        .def_prop_rw("check_tensors", [](ModelParams &p){ return p.raw.check_tensors; }, [](ModelParams &p, bool v){ p.raw.check_tensors = v; }, "Validate tensor data on load")
        .def_prop_rw("no_host", [](ModelParams &p){ return p.raw.no_host; }, [](ModelParams &p, bool v){ p.raw.no_host = v; }, "Don't allocate host memory for tensors")
        .def("as_dict", [](const ModelParams &p) {
            nb::dict d;
            d["n_gpu_layers"] = p.raw.n_gpu_layers;
            d["main_gpu"] = p.raw.main_gpu;
            d["split_mode"] = p.raw.split_mode;
            d["vocab_only"] = p.raw.vocab_only;
            d["use_mmap"] = p.raw.use_mmap;
            d["use_mlock"] = p.raw.use_mlock;
            d["check_tensors"] = p.raw.check_tensors;
            d["no_host"] = p.raw.no_host;
            return d;
        }, "Convert parameters to dictionary");

    nb::class_<ContextParams>(m, "ContextParams", "Parameters for creating an inference context")
        .def(nb::init<>())
        .def_prop_rw("n_ctx", [](ContextParams &p){ return p.raw.n_ctx; }, [](ContextParams &p, uint32_t v){ p.raw.n_ctx = v; }, "Context size (max tokens)")
        .def_prop_rw("n_batch", [](ContextParams &p){ return p.raw.n_batch; }, [](ContextParams &p, uint32_t v){ p.raw.n_batch = v; }, "Batch size for prompt processing")
        .def_prop_rw("n_ubatch", [](ContextParams &p){ return p.raw.n_ubatch; }, [](ContextParams &p, uint32_t v){ p.raw.n_ubatch = v; }, "Micro-batch size")
        .def_prop_rw("n_seq_max", [](ContextParams &p){ return p.raw.n_seq_max; }, [](ContextParams &p, uint32_t v){ p.raw.n_seq_max = v; }, "Max number of sequences")
        .def_prop_rw("n_threads", [](ContextParams &p){ return p.raw.n_threads; }, [](ContextParams &p, int32_t v){ p.raw.n_threads = v; }, "Threads for generation")
        .def_prop_rw("n_threads_batch", [](ContextParams &p){ return p.raw.n_threads_batch; }, [](ContextParams &p, int32_t v){ p.raw.n_threads_batch = v; }, "Threads for batch processing")
        .def_prop_rw("rope_freq_base", [](ContextParams &p){ return p.raw.rope_freq_base; }, [](ContextParams &p, float v){ p.raw.rope_freq_base = v; }, "RoPE base frequency")
        .def_prop_rw("rope_freq_scale", [](ContextParams &p){ return p.raw.rope_freq_scale; }, [](ContextParams &p, float v){ p.raw.rope_freq_scale = v; }, "RoPE frequency scale")
        .def_prop_rw("embeddings", [](ContextParams &p){ return p.raw.embeddings; }, [](ContextParams &p, bool v){ p.raw.embeddings = v; }, "Enable embedding extraction")
        .def_prop_rw("offload_kqv", [](ContextParams &p){ return p.raw.offload_kqv; }, [](ContextParams &p, bool v){ p.raw.offload_kqv = v; }, "Offload KQV to GPU")
        .def_prop_rw("flash_attn_type", [](ContextParams &p){ return p.raw.flash_attn_type; }, [](ContextParams &p, int v){ p.raw.flash_attn_type = static_cast<llama_flash_attn_type>(v); }, "Flash attention type (0=disabled)")
        .def("as_dict", [](const ContextParams &p) {
            nb::dict d;
            d["n_ctx"] = p.raw.n_ctx;
            d["n_batch"] = p.raw.n_batch;
            d["n_ubatch"] = p.raw.n_ubatch;
            d["n_seq_max"] = p.raw.n_seq_max;
            d["n_threads"] = p.raw.n_threads;
            d["n_threads_batch"] = p.raw.n_threads_batch;
            d["rope_freq_base"] = p.raw.rope_freq_base;
            d["rope_freq_scale"] = p.raw.rope_freq_scale;
            d["embeddings"] = p.raw.embeddings;
            d["offload_kqv"] = p.raw.offload_kqv;
            d["flash_attn_type"] = p.raw.flash_attn_type;
            return d;
        }, "Convert parameters to dictionary");

    nb::class_<Model>(m, "Model", "Loaded LLM model")
        .def(nb::init<const std::string &, const ModelParams &>(), "path"_a, "params"_a, 
             nb::call_guard<nb::gil_scoped_release>(), "Load model from GGUF file")
        .def("close", &Model::close, "Explicitly free model resources")
        .def("n_vocab", &Model::n_vocab, "Vocabulary size")
        .def("n_ctx_train", &Model::n_ctx_train, "Training context size")
        .def("desc", &Model::desc, "Model description string")
        .def("tokenize", &Model::tokenize, "text"_a, nb::kw_only(), "add_special"_a=true, "parse_special"_a=false,
             nb::call_guard<nb::gil_scoped_release>(), "Convert text to tokens")
        .def("detokenize", &Model::detokenize, "tokens"_a, nb::kw_only(), "remove_special"_a=true, "unparse_special"_a=false,
             nb::call_guard<nb::gil_scoped_release>(), "Convert tokens to text")
        .def("bos", &Model::bos, "Beginning-of-sequence token ID")
        .def("eos", &Model::eos, "End-of-sequence token ID")
        .def("eot", &Model::eot, "End-of-turn token ID")
        .def("n_embd", &Model::n_embd, "Embedding dimension")
        .def("meta_count", &Model::meta_count, "Number of metadata entries")
        .def("meta_val_str", &Model::meta_val_str, "key"_a, "Get metadata value by key")
        .def("meta_key_by_index", &Model::meta_key_by_index, "index"_a, "Get metadata key by index")
        .def("meta_val_by_index", &Model::meta_val_by_index, "index"_a, "Get metadata value by index")
        .def("model_size", &Model::model_size, "Model size in bytes")
        .def("n_params", &Model::n_params, "Number of parameters")
        .def("n_layer", &Model::n_layer, "Number of layers")
        .def("chat_template", &Model::chat_template, "name"_a = "", "Get chat template string")
        .def("token_to_piece", &Model::token_to_piece, "token"_a, "Convert single token to text");

    nb::class_<SamplerChain::Params>(m, "SamplerParams", "Sampling parameters for text generation")
        .def(nb::init<>())
        .def_rw("top_k", &SamplerChain::Params::top_k, "Top-K sampling (0 = disabled)")
        .def_rw("top_p", &SamplerChain::Params::top_p, "Top-P (nucleus) sampling")
        .def_rw("min_p", &SamplerChain::Params::min_p, "Min-P sampling threshold")
        .def_rw("min_keep", &SamplerChain::Params::min_keep, "Minimum tokens to keep")
        .def_rw("temp", &SamplerChain::Params::temp, "Temperature (1.0 = neutral)")
        .def_rw("penalty_last_n", &SamplerChain::Params::penalty_last_n, "Tokens to consider for penalties")
        .def_rw("repeat_penalty", &SamplerChain::Params::repeat_penalty, "Repetition penalty (1.0 = disabled)")
        .def_rw("freq_penalty", &SamplerChain::Params::freq_penalty, "Frequency penalty")
        .def_rw("presence_penalty", &SamplerChain::Params::presence_penalty, "Presence penalty")
        .def_rw("seed", &SamplerChain::Params::seed, "RNG seed (-1 = random)");

    nb::class_<SamplerChain>(m, "SamplerChain", "Sampler chain for token selection")
        .def(nb::init<const Model &, const SamplerChain::Params &>(), "model"_a, "params"_a,
             nb::keep_alive<1, 2>(), "Create sampler chain")
        .def("reset", &SamplerChain::reset, "Reset sampler state")
        .def("sample", &SamplerChain::sample, "ctx"_a, nb::arg("idx") = -1, "Sample next token from logits");

    nb::class_<Context>(m, "Context", "Inference context with KV cache")
        .def(nb::init<Model &, const ContextParams &>(),
            "model"_a,
            "params"_a,
            nb::keep_alive<1, 2>(),
            "Create inference context")
        .def("close", &Context::close, "Explicitly free context resources")
        .def("n_ctx", &Context::n_ctx, "Current context size")
        .def("set_thread_count", &Context::set_thread_count, "n_threads"_a, "n_threads_batch"_a, "Set thread counts")
        .def("reset", &Context::reset, "Reset context (recreates KV cache)")
        .def("decode", &Context::decode, "tokens"_a, nb::arg("return_logits") = true, 
             nb::call_guard<nb::gil_scoped_release>(), "Process tokens through model")
        .def("decode_one", &Context::decode_one, "token"_a, nb::arg("request_logits") = true,
             nb::call_guard<nb::gil_scoped_release>())
        .def("logits", &Context::logits, "Get logits from last decode")
        .def("embeddings", &Context::embeddings, "Get embeddings from last decode")
        .def("generate_next", &Context::generate_next, "sampler"_a, nb::arg("idx") = -1, 
             nb::call_guard<nb::gil_scoped_release>(), "Sample and return next token")
        .def("model", &Context::model, nb::rv_policy::reference, "Get associated model")
        .def("save_state", &Context::save_state, "path"_a, 
             nb::call_guard<nb::gil_scoped_release>(), "Save context state to file")
        .def("load_state", &Context::load_state, "path"_a, 
             nb::call_guard<nb::gil_scoped_release>(), "Load context state from file")
        .def("get_state_data", &Context::get_state_data, 
             nb::call_guard<nb::gil_scoped_release>(), "Get state as bytes")
        .def("set_state_data", &Context::set_state_data, "data"_a, 
             nb::call_guard<nb::gil_scoped_release>(), "Set state from bytes")
        .def("set_lora", &Context::set_lora, "adapter"_a, "scale"_a = 1.0f, "Apply LoRA adapter with scale")
        .def("remove_lora", &Context::remove_lora, "adapter"_a, "Remove specific LoRA adapter")
        .def("clear_lora", &Context::clear_lora, "Remove all LoRA adapters")
        .def("perf", &Context::perf, "Get performance metrics dict")
        .def("perf_reset", &Context::perf_reset, "Reset performance counters")
        .def("kv_cache_clear", &Context::kv_cache_clear, "Clear entire KV cache")
        .def("kv_cache_seq_rm", &Context::kv_cache_seq_rm, "seq_id"_a, "p0"_a = -1, "p1"_a = -1, "Remove KV cache for sequence")
        .def("kv_cache_seq_cp", &Context::kv_cache_seq_cp, "seq_id_src"_a, "seq_id_dst"_a, "p0"_a = -1, "p1"_a = -1, "Copy KV cache between sequences")
        .def("kv_cache_seq_keep", &Context::kv_cache_seq_keep, "seq_id"_a, "Keep only specified sequence")
        .def("kv_cache_seq_add", &Context::kv_cache_seq_add, "seq_id"_a, "p0"_a, "p1"_a, "delta"_a, "Add position delta to sequence")
        .def("kv_cache_seq_pos_max", &Context::kv_cache_seq_pos_max, "seq_id"_a = 0, "Get max position in sequence");

    nb::class_<LoraAdapter>(m, "LoraAdapter", "LoRA adapter for model fine-tuning")
        .def(nb::init<Model &, const std::string &>(), "model"_a, "path"_a,
             nb::keep_alive<1, 2>(), "Load LoRA adapter from file");

    m.def("generate_tokens", &generate_tokens,
          "ctx"_a,
          "sampler"_a,
          "prompt"_a,
          "max_new_tokens"_a,
          "add_bos"_a,
          "eos_token"_a,
          "stop_tokens"_a = std::vector<llama_token>{},
          nb::call_guard<nb::gil_scoped_release>(),
          "Generate tokens using sampler chain. Returns list of token IDs.");

    nb::class_<TokenProb>(m, "TokenProb", "Token with probability information")
        .def_ro("token", &TokenProb::token, "Token ID")
        .def_ro("logprob", &TokenProb::logprob, "Log probability")
        .def_ro("top_logprobs", &TokenProb::top_logprobs, "Top alternative tokens with logprobs");

    m.def("generate_tokens_with_details", &generate_tokens_with_details,
          "ctx"_a,
          "sampler"_a,
          "prompt"_a,
          "max_new_tokens"_a,
          "add_bos"_a,
          "eos_token"_a,
          "stop_sequences"_a = std::vector<std::vector<llama_token>>{},
          "top_logprobs"_a = 0,
          "echo_prompt"_a = false,
          nb::call_guard<nb::gil_scoped_release>(),
          "Generate tokens with per-token logprobs. Returns list of TokenProb.");

    // logging controls
    m.def("set_log_level", &set_log_level, "min_level"_a,
          "Set minimum log level (0=none, 1=debug, 2=info, 3=warn, 4=error)");
    m.def("disable_logging", &disable_logging, "Disable all llama.cpp logging");
    m.def("reset_logging", &reset_logging, "Restore default llama.cpp logging");
    m.def("print_system_info", []() { return std::string(llama_print_system_info()); },
          "Return llama.cpp system info string (CPU features, build info, etc.).");

    // Chat template
    m.def("chat_apply_template", &chat_apply_template,
          "model"_a,
          "messages"_a,
          "tmpl"_a = "",
          "add_generation_prompt"_a = true,
          "Apply chat template to messages. Returns formatted prompt string.");

    // Grammar sampler
    nb::class_<GrammarSampler>(m, "GrammarSampler")
        .def(nb::init<const Model &, const std::string &, const std::string &>(),
             "model"_a, "grammar_str"_a, "grammar_root"_a = "root")
        .def("accept", &GrammarSampler::accept, "token"_a)
        .def("reset", &GrammarSampler::reset);

    m.def("generate_tokens_with_grammar", &generate_tokens_with_grammar,
          "ctx"_a,
          "sampler"_a,
          "grammar"_a,
          "prompt"_a,
          "max_new_tokens"_a,
          "add_bos"_a,
          "eos_token"_a,
          "stop_tokens"_a = std::vector<llama_token>{},
          nb::call_guard<nb::gil_scoped_release>(),
          "Generation with grammar constraint");

    m.def("generate_tokens_multi_stop", &generate_tokens_multi_stop,
          "ctx"_a,
          "sampler"_a,
          "prompt"_a,
          "max_new_tokens"_a,
          "add_bos"_a,
          "eos_token"_a,
          "stop_sequences"_a = std::vector<std::vector<llama_token>>{},
          nb::call_guard<nb::gil_scoped_release>(),
          "Generation with multi-token stop sequences");

    m.def("generate_tokens_grammar_multi_stop", &generate_tokens_grammar_multi_stop,
          "ctx"_a,
          "sampler"_a,
          "grammar"_a,
          "prompt"_a,
          "max_new_tokens"_a,
          "add_bos"_a,
          "eos_token"_a,
          "stop_sequences"_a = std::vector<std::vector<llama_token>>{},
          nb::call_guard<nb::gil_scoped_release>(),
          "Generation with grammar and multi-token stop sequences");

    m.def("generate_tokens_streaming", &generate_tokens_streaming,
          "ctx"_a,
          "sampler"_a,
          "prompt"_a,
          "max_new_tokens"_a,
          "add_bos"_a,
          "eos_token"_a,
          "stop_sequences"_a = std::vector<std::vector<llama_token>>{},
          "callback"_a,
          "Streaming generation with callback. Callback receives token, returns False to stop.");

    // Backend cleanup - call before interpreter shutdown to prevent segfault
    m.def("backend_free", []() {
        if (g_model_count.load() == 0) {
            llama_backend_free();
        }
    }, "Free llama.cpp backend resources. Only frees if no models are loaded.");

    m.def("backend_can_free", []() -> bool {
        return g_model_count.load() == 0;
    }, "Check if backend can be safely freed (no models loaded).");

    m.def("model_count", []() -> int {
        return g_model_count.load();
    }, "Return number of currently loaded models.");
}
