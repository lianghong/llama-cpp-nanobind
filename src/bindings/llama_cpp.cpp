#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

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
std::once_flag g_backend_init_flag;
std::atomic<int> g_model_count{0};
std::mutex g_init_mutex;

void init_backend() {
  llama_backend_init();
}

class Model {
 public:
  explicit Model(const std::string& path, const ModelParams& params)
      : model_(llama_model_load_from_file(path.c_str(), params.raw)) {
    std::lock_guard<std::mutex> const lock(g_init_mutex);
    std::call_once(g_backend_init_flag, init_backend);

    if (!model_) {
      throw std::runtime_error("failed to load model: " + path);
    }
    ++g_model_count;
  }

  ~Model() { close(); }

  void close() {
    std::lock_guard<std::mutex> const lock(g_init_mutex);
    if (model_) {
      llama_model_free(model_);
      model_ = nullptr;
      --g_model_count;
    }
  }

  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;
  Model(Model&&) = delete;
  Model& operator=(Model&&) = delete;

  const llama_model* get() const {
    check_model();
    return model_;
  }
  llama_model* get() {
    check_model();
    return model_;
  }

  const llama_vocab* vocab() const {
    check_model();
    return llama_model_get_vocab(model_);
  }

  int32_t n_vocab() const { return llama_vocab_n_tokens(vocab()); }

  int32_t n_ctx_train() const {
    check_model();
    return llama_model_n_ctx_train(model_);
  }

  uint64_t model_size() const {
    check_model();
    return llama_model_size(model_);
  }

  uint64_t n_params() const {
    check_model();
    return llama_model_n_params(model_);
  }

  int32_t n_layer() const {
    check_model();
    return llama_model_n_layer(model_);
  }

  int32_t n_head() const {
    check_model();
    return llama_model_n_head(model_);
  }

  bool has_encoder() const {
    check_model();
    return llama_model_has_encoder(model_);
  }

  bool has_decoder() const {
    check_model();
    return llama_model_has_decoder(model_);
  }

  bool is_recurrent() const {
    check_model();
    return llama_model_is_recurrent(model_);
  }

  bool is_hybrid() const {
    check_model();
    return llama_model_is_hybrid(model_);
  }

  std::string chat_template(const std::string& name = "") const {
    check_model();
    const char* tmpl = llama_model_chat_template(model_, name.empty() ? nullptr : name.c_str());
    return tmpl ? std::string(tmpl) : "";
  }

  std::string desc() const {
    check_model();
    // Query required size first to avoid buffer overflow
    int32_t const needed = llama_model_desc(model_, nullptr, 0);
    if (needed <= 0) {
      return "";  // Empty description
    }
    std::string buf(static_cast<size_t>(needed) + 1, '\0');
    llama_model_desc(model_, buf.data(), static_cast<int32_t>(buf.size()));
    buf.resize(static_cast<size_t>(needed));
    return buf;
  }

  llama_token bos() const { return llama_vocab_bos(vocab()); }
  llama_token eos() const { return llama_vocab_eos(vocab()); }
  llama_token eot() const { return llama_vocab_eot(vocab()); }
  llama_token sep() const { return llama_vocab_sep(vocab()); }
  llama_token nl() const { return llama_vocab_nl(vocab()); }
  llama_token pad() const { return llama_vocab_pad(vocab()); }

  bool get_add_bos() const { return llama_vocab_get_add_bos(vocab()); }

  std::string token_to_piece(llama_token token) const {
    const char* text = llama_vocab_get_text(vocab(), token);
    return text ? std::string(text) : "";
  }

  int32_t n_embd() const {
    check_model();
    return llama_model_n_embd(model_);
  }

  // Metadata access
  int32_t meta_count() const {
    check_model();
    return llama_model_meta_count(model_);
  }

  std::string meta_val_str(const std::string& key) const {
    check_model();
    // First call to get required length
    int32_t const len = llama_model_meta_val_str(model_, key.c_str(), nullptr, 0);
    if (len < 0) return "";
    std::string buf(static_cast<size_t>(len) + 1, '\0');
    llama_model_meta_val_str(model_, key.c_str(), buf.data(), static_cast<int32_t>(buf.size()));
    buf.resize(static_cast<size_t>(len));
    return buf;
  }

  std::string meta_key_by_index(int32_t i) const {
    check_model();
    int32_t const len = llama_model_meta_key_by_index(model_, i, nullptr, 0);
    if (len < 0) return "";
    std::string buf(static_cast<size_t>(len) + 1, '\0');
    llama_model_meta_key_by_index(model_, i, buf.data(), static_cast<int32_t>(buf.size()));
    buf.resize(static_cast<size_t>(len));
    return buf;
  }

  std::string meta_val_by_index(int32_t i) const {
    check_model();
    int32_t const len = llama_model_meta_val_str_by_index(model_, i, nullptr, 0);
    if (len < 0) return "";
    std::string buf(static_cast<size_t>(len) + 1, '\0');
    llama_model_meta_val_str_by_index(model_, i, buf.data(), static_cast<int32_t>(buf.size()));
    buf.resize(static_cast<size_t>(len));
    return buf;
  }

  std::vector<llama_token> tokenize(const std::string& text, bool add_special,
                                    bool parse_special) const {
    // Clamp to prevent integer overflow on very large inputs
    constexpr size_t MAX_TEXT_SIZE = static_cast<size_t>(INT32_MAX);
    if (text.size() > MAX_TEXT_SIZE) {
      throw std::runtime_error("input text too large for tokenization (exceeds INT32_MAX)");
    }
    constexpr size_t MAX_TOKENS = 1 << 24;  // 16M tokens max
    size_t const estimated = text.size() + 8;
    if (estimated > MAX_TOKENS) {
      throw std::runtime_error("input text too large for tokenization");
    }
    int32_t max_tokens = static_cast<int32_t>(std::max(estimated, size_t{32}));
    std::vector<llama_token> tokens(static_cast<size_t>(max_tokens));
    int32_t n = llama_tokenize(vocab(), text.c_str(), static_cast<int32_t>(text.size()),
                               tokens.data(), max_tokens, add_special, parse_special);
    if (n < 0) {
      max_tokens = -n;
      tokens.assign(static_cast<size_t>(max_tokens), 0);
      n = llama_tokenize(vocab(), text.c_str(), static_cast<int32_t>(text.size()), tokens.data(),
                         max_tokens, add_special, parse_special);
    }
    if (n < 0) {
      throw std::runtime_error("tokenization failed");
    }
    tokens.resize(static_cast<size_t>(n));
    return tokens;
  }

  std::string detokenize(const std::vector<llama_token>& tokens, bool remove_special,
                         bool unparse_special) const {
    if (tokens.size() > static_cast<size_t>(INT32_MAX)) {
      throw std::runtime_error("too many tokens for detokenization");
    }
    int32_t const n_tokens = static_cast<int32_t>(tokens.size());
    int32_t needed = llama_detokenize(vocab(), tokens.data(), n_tokens, nullptr, 0, remove_special,
                                      unparse_special);
    if (needed < 0) {
      needed = -needed;
    }
    std::string out;
    out.resize(static_cast<size_t>(needed));
    int32_t const written = llama_detokenize(vocab(), tokens.data(), n_tokens, out.data(), needed,
                                             remove_special, unparse_special);
    if (written < 0) {
      throw std::runtime_error("detokenize failed");
    }
    out.resize(static_cast<size_t>(written));
    return out;
  }

  nb::bytes detokenize_bytes(const std::vector<llama_token>& tokens, bool remove_special,
                             bool unparse_special) const {
    if (tokens.size() > static_cast<size_t>(INT32_MAX)) {
      throw std::runtime_error("too many tokens for detokenization");
    }
    std::string out;
    bool detok_failed = false;
    {
      // Release GIL for the heavy detokenization work, but re-acquire
      // before constructing the Python bytes object below.
      nb::gil_scoped_release const release;
      int32_t const n_tokens = static_cast<int32_t>(tokens.size());
      int32_t needed = llama_detokenize(vocab(), tokens.data(), n_tokens, nullptr, 0,
                                        remove_special, unparse_special);
      if (needed < 0) {
        needed = -needed;
      }
      out.resize(static_cast<size_t>(needed));
      int32_t const written = llama_detokenize(vocab(), tokens.data(), n_tokens, out.data(), needed,
                                               remove_special, unparse_special);
      if (written < 0) {
        detok_failed = true;
      } else {
        out.resize(static_cast<size_t>(written));
      }
    }
    // GIL re-acquired — safe to throw
    if (detok_failed) {
      throw std::runtime_error("detokenize failed");
    }
    // GIL re-acquired — safe to create Python bytes object
    return nb::bytes(out.data(), out.size());
  }

 private:
  llama_model* model_ = nullptr;

  void check_model() const {
    if (!model_) {
      throw std::runtime_error("model is null (already freed or failed to load)");
    }
  }
};

class Context;
class LoraAdapter;

class SamplerChain {
 public:
  struct Params {
    int32_t top_k = 40;
    float top_p = 0.95F;
    float min_p = 0.0F;
    size_t min_keep = 1;
    float temp = 0.8F;
    int32_t penalty_last_n = 64;
    float repeat_penalty = 1.1F;
    float freq_penalty = 0.0F;
    float presence_penalty = 0.0F;
    int32_t seed = -1;
    // Dynamic temperature
    float temp_delta = 0.0F;
    float temp_exponent = 1.0F;
    // XTC sampler
    float xtc_probability = 0.0F;
    float xtc_threshold = 0.1F;
    // Top-n-sigma (negative = disabled)
    float top_n_sigma = -1.0F;
    // DRY (Don't Repeat Yourself) anti-repetition
    float dry_multiplier = 0.0F;
    float dry_base = 1.75F;
    int32_t dry_allowed_length = 2;
    int32_t dry_penalty_last_n = -1;
    std::vector<std::string> dry_seq_breakers = {"\n", ":", "\"", "*"};
  };

  SamplerChain(const Model& model, const Params& params) {
    auto chain_params = llama_sampler_chain_default_params();
    sampler_ = llama_sampler_chain_init(chain_params);
    if (!sampler_) {
      throw std::runtime_error("failed to create sampler chain");
    }

    // Canonical sampler ordering:
    // 1. DRY (anti-repetition on raw logits)
    if (params.dry_multiplier > 0.0F) {
      // Convert breaker strings to C-style array
      std::vector<const char*> breaker_ptrs;
      breaker_ptrs.reserve(params.dry_seq_breakers.size());
      for (const auto& s : params.dry_seq_breakers) {
        breaker_ptrs.push_back(s.c_str());
      }
      llama_sampler_chain_add(
          sampler_, llama_sampler_init_dry(model.vocab(), model.n_ctx_train(),
                                           params.dry_multiplier, params.dry_base,
                                           params.dry_allowed_length, params.dry_penalty_last_n,
                                           breaker_ptrs.data(), breaker_ptrs.size()));
    }

    // 2. Penalties (repeat/freq/presence)
    if (params.penalty_last_n != 0 || params.repeat_penalty != 1.0F ||
        params.freq_penalty != 0.0F || params.presence_penalty != 0.0F) {
      llama_sampler* penalties =
          llama_sampler_init_penalties(params.penalty_last_n, params.repeat_penalty,
                                       params.freq_penalty, params.presence_penalty);
      llama_sampler_chain_add(sampler_, penalties);
    }

    // 3. Top-n-sigma (truncate by standard deviations)
    if (params.top_n_sigma >= 0.0F) {
      llama_sampler_chain_add(sampler_, llama_sampler_init_top_n_sigma(params.top_n_sigma));
    }

    // 4. Top-K
    if (params.top_k > 0) {
      llama_sampler_chain_add(sampler_, llama_sampler_init_top_k(params.top_k));
    }

    // 5. Top-P
    if (params.top_p < 1.0F) {
      llama_sampler_chain_add(sampler_, llama_sampler_init_top_p(params.top_p, params.min_keep));
    }

    // 6. Min-P
    if (params.min_p > 0.0F) {
      llama_sampler_chain_add(sampler_, llama_sampler_init_min_p(params.min_p, params.min_keep));
    }

    // 7. XTC (on filtered candidates)
    if (params.xtc_probability > 0.0F) {
      uint32_t const xtc_seed = params.seed >= 0
                                    ? static_cast<uint32_t>(params.seed)
                                    : static_cast<uint32_t>(llama_time_us() & 0xFFFFFFFF);
      llama_sampler_chain_add(sampler_,
                              llama_sampler_init_xtc(params.xtc_probability, params.xtc_threshold,
                                                     params.min_keep, xtc_seed));
    }

    // 8. Temperature: use dynamic temp if delta > 0, otherwise static temp
    if (params.temp_delta > 0.0F) {
      llama_sampler_chain_add(sampler_, llama_sampler_init_temp_ext(params.temp, params.temp_delta,
                                                                    params.temp_exponent));
    } else if (params.temp != 1.0F) {
      llama_sampler_chain_add(sampler_, llama_sampler_init_temp(params.temp));
    }

    // 9. Dist (final sampling)
    uint32_t const rng_seed = params.seed >= 0
                                  ? static_cast<uint32_t>(params.seed)
                                  : static_cast<uint32_t>(llama_time_us() & 0xFFFFFFFF);
    llama_sampler_chain_add(sampler_, llama_sampler_init_dist(rng_seed));

    // Note: BOS is NOT pre-accepted here. All generation functions accept
    // prompt tokens (including BOS) into the sampler before generating,
    // so pre-accepting BOS here would double-count it in penalty tracking.
  }

  ~SamplerChain() {
    if (sampler_) {
      llama_sampler_free(sampler_);
      sampler_ = nullptr;
    }
  }

  SamplerChain(const SamplerChain&) = delete;
  SamplerChain& operator=(const SamplerChain&) = delete;
  SamplerChain(SamplerChain&&) = delete;
  SamplerChain& operator=(SamplerChain&&) = delete;

  void reset() {
    if (sampler_) {
      llama_sampler_reset(sampler_);
    }
  }

  llama_sampler* get() const {
    if (!sampler_) {
      throw std::runtime_error("sampler is null (not initialized)");
    }
    return sampler_;
  }

  llama_token sample(Context& ctx, int32_t idx);

 private:
  llama_sampler* sampler_ = nullptr;
};

class Context {
 public:
  Context(Model& model_ref, const ContextParams& params) : model_(&model_ref), params_(params) {
    ctx_ = llama_init_from_model(model_->get(), params_.raw);
    if (!ctx_) {
      throw std::runtime_error("failed to create llama context");
    }
    cur_pos_ = 0;
    // Pre-allocate single-token batch for decode_one to avoid per-token
    // allocations
    single_batch_ = llama_batch_init(1, 0, 1);
    llama_set_n_threads(ctx_, params_.raw.n_threads, params_.raw.n_threads_batch);
  }

  ~Context() { close(); }

  void close() {
    std::lock_guard<std::mutex> const lock(g_init_mutex);
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

  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
  Context(Context&&) = delete;
  Context& operator=(Context&&) = delete;

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
    // single_batch_ persists across resets (allocated in constructor,
    // freed only in close()). Reinitialize only if it was freed.
    if (!single_batch_.token) {
      single_batch_ = llama_batch_init(1, 0, 1);
    }
  }

  void decode(const std::vector<llama_token>& tokens, bool return_logits = true) {
    check_ctx();
    if (tokens.empty()) return;
    llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
    // RAII guard ensures batch is freed regardless of how scope exits
    struct BatchGuard {
      llama_batch& b;
      explicit BatchGuard(llama_batch& batch) : b(batch) {}
      ~BatchGuard() { llama_batch_free(b); }
      BatchGuard(const BatchGuard&) = delete;
      BatchGuard& operator=(const BatchGuard&) = delete;
      BatchGuard(BatchGuard&&) = delete;
      BatchGuard& operator=(BatchGuard&&) = delete;
    } const guard(batch);

    batch.n_tokens = static_cast<int32_t>(tokens.size());
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
      batch.token[i] = tokens[static_cast<size_t>(i)];
      batch.pos[i] = cur_pos_ + i;
      batch.n_seq_id[i] = 1;
      batch.seq_id[i][0] = 0;
      batch.logits[i] = (return_logits && i == batch.n_tokens - 1) ? 1 : 0;
    }
    int32_t const rc = llama_decode(ctx_, batch);
    if (rc < 0) {
      throw std::runtime_error("llama_decode failed with code " + std::to_string(rc));
    }
    cur_pos_ += static_cast<int32_t>(tokens.size());
  }

  void decode_one(llama_token token, bool request_logits = true) {
    check_ctx();
    // Reuse pre-allocated single-token batch to avoid per-token allocations
    single_batch_.n_tokens = 1;
    single_batch_.token[0] = token;
    single_batch_.pos[0] = cur_pos_;
    single_batch_.n_seq_id[0] = 1;
    single_batch_.seq_id[0][0] = 0;
    single_batch_.logits[0] = request_logits ? 1 : 0;
    int32_t const rc = llama_decode(ctx_, single_batch_);
    if (rc < 0) {
      throw std::runtime_error("llama_decode (single) failed with code " + std::to_string(rc));
    }
    ++cur_pos_;
  }

  std::vector<float> logits() const {
    check_ctx();
    if (!model_) {
      throw std::runtime_error("context has been closed");
    }
    const int32_t n_vocab = model_->n_vocab();
    float* ptr = llama_get_logits(const_cast<llama_context*>(ctx_));
    if (!ptr) {
      throw std::runtime_error("logits unavailable; ensure decode was called with logits enabled");
    }
    return std::vector<float>(ptr, ptr + n_vocab);
  }

  std::vector<float> embeddings() const {
    check_ctx();
    if (!model_) {
      throw std::runtime_error("context has been closed");
    }
    float* ptr = llama_get_embeddings(const_cast<llama_context*>(ctx_));
    if (!ptr) {
      throw std::runtime_error(
          "embeddings unavailable; ensure pooling_type is "
          "set and decode was called");
    }
    const int32_t n_embd = llama_model_n_embd(model_->get());
    return std::vector<float>(ptr, ptr + n_embd);
  }

  llama_token generate_next(SamplerChain& sampler, int32_t idx = -1) {
    return sampler.sample(*this, idx);
  }

  Model& model() const {
    if (!model_) {
      throw std::runtime_error("context has been closed");
    }
    return *model_;
  }

  llama_context* raw() const {
    check_ctx();
    return ctx_;
  }

  // State save/load
  bool save_state(const std::string& path) {
    check_ctx();
    return llama_state_save_file(ctx_, path.c_str(), nullptr, 0);
  }

  size_t load_state(const std::string& path) {
    check_ctx();
    size_t n_token_count = 0;
    bool const ok = llama_state_load_file(ctx_, path.c_str(), nullptr, 0, &n_token_count);
    if (!ok) {
      throw std::runtime_error("failed to load state from: " + path);
    }
    // Update cur_pos_ from KV cache to maintain correct position bookkeeping
    cur_pos_ = kv_cache_seq_pos_max(0) + 1;
    if (cur_pos_ < 0) cur_pos_ = 0;
    return n_token_count;
  }

  // Returns state as Python bytes via zero-copy path (no intermediate list).
  // GIL is managed manually: released during heavy C++ work, held for Python
  // object creation.
  nb::bytes get_state_data() {
    check_ctx();
    size_t size = 0;
    size_t written = 0;
    std::vector<uint8_t> buf;
    {
      nb::gil_scoped_release const release;
      size = llama_state_get_size(ctx_);
      buf.resize(size);
      written = llama_state_get_data(ctx_, buf.data(), size);
    }
    // GIL held here — safe to construct Python bytes object
    return nb::bytes(buf.data(), written);
  }

  // Accepts Python bytes directly (pointer access, no per-element conversion).
  // GIL is managed manually: pointer extracted while held, released for heavy
  // C++ work.
  size_t set_state_data(const nb::bytes& data) {
    check_ctx();
    const auto* ptr = reinterpret_cast<const uint8_t*>(data.data());
    size_t const len = data.size();
    size_t result = 0;
    {
      nb::gil_scoped_release const release;
      result = llama_state_set_data(ctx_, ptr, len);
      // Update cur_pos_ from KV cache to maintain correct position bookkeeping
      cur_pos_ = kv_cache_seq_pos_max(0) + 1;
      if (cur_pos_ < 0) cur_pos_ = 0;
    }
    // GIL re-acquired — `data` (nb::bytes) guaranteed alive until function returns
    return result;
  }

  // LoRA adapter management - defined after LoraAdapter class
  int32_t set_adapters_lora(const nb::list& py_adapters, const nb::list& py_scales);

  void clear_lora() {
    if (!ctx_) return;
    llama_set_adapters_lora(ctx_, nullptr, 0, nullptr);
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
    // Use llama_memory_clear for full reset (handles both attention KV cache
    // and recurrent state in hybrid architectures like Qwen3.5)
    llama_memory_clear(mem, false);
    cur_pos_ = 0;
  }

  bool kv_cache_seq_rm(int32_t seq_id, int32_t p0 = -1, int32_t p1 = -1) {
    if (!ctx_) return false;
    llama_memory_t mem = llama_get_memory(ctx_);
    bool const result = llama_memory_seq_rm(mem, seq_id, p0, p1);
    // Update cur_pos_ if we modified sequence 0 (the default sequence)
    if (result && (seq_id == 0 || seq_id == -1)) {
      int32_t const new_pos = llama_memory_seq_pos_max(mem, 0) + 1;
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
    int32_t const new_pos = llama_memory_seq_pos_max(mem, 0) + 1;
    cur_pos_ = new_pos < 0 ? 0 : new_pos;
  }

  void kv_cache_seq_add(int32_t seq_id, int32_t p0, int32_t p1, int32_t delta) {
    if (!ctx_) return;
    llama_memory_t mem = llama_get_memory(ctx_);
    llama_memory_seq_add(mem, seq_id, p0, p1, delta);
    // Update cur_pos_ if we modified sequence 0
    if (seq_id == 0) {
      int32_t const new_pos = llama_memory_seq_pos_max(mem, 0) + 1;
      cur_pos_ = new_pos < 0 ? 0 : new_pos;
    }
  }

  int32_t kv_cache_seq_pos_max(int32_t seq_id = 0) {
    if (!ctx_) return -1;
    llama_memory_t mem = llama_get_memory(ctx_);
    return llama_memory_seq_pos_max(mem, seq_id);
  }

  int32_t kv_cache_seq_pos_min(int32_t seq_id = 0) {
    if (!ctx_) return -1;
    llama_memory_t mem = llama_get_memory(ctx_);
    return llama_memory_seq_pos_min(mem, seq_id);
  }

  bool memory_can_shift() {
    if (!ctx_) return false;
    llama_memory_t mem = llama_get_memory(ctx_);
    return llama_memory_can_shift(mem);
  }

  void set_embeddings(bool enabled) {
    check_ctx();
    llama_set_embeddings(ctx_, enabled);
  }

  void set_causal_attn(bool enabled) {
    check_ctx();
    llama_set_causal_attn(ctx_, enabled);
  }

 private:
  Model* model_ = nullptr;
  llama_context* ctx_ = nullptr;
  ContextParams params_;
  int32_t cur_pos_ = 0;
  llama_batch single_batch_ = {};  // Reusable single-token batch for decode_one

  void check_ctx() const {
    if (!ctx_) {
      throw std::runtime_error("context has been closed");
    }
  }
};

inline llama_token SamplerChain::sample(Context& ctx, int32_t idx) {
  if (!sampler_) {
    throw std::runtime_error("sampler not initialized");
  }
  return llama_sampler_sample(sampler_, ctx.raw(), idx);
}

std::vector<llama_token> generate_tokens(Context& ctx, SamplerChain& sampler,
                                         const std::vector<llama_token>& prompt,
                                         int32_t max_new_tokens, bool add_bos,
                                         llama_token eos_token,
                                         const std::vector<llama_token>& stop_tokens) {
  std::vector<llama_token> output;
  output.reserve(static_cast<size_t>(max_new_tokens));

  std::vector<llama_token> priming = prompt;
  if (add_bos && (priming.empty() || priming.front() != ctx.model().bos())) {
    priming.insert(priming.begin(), ctx.model().bos());
  }

  // Accept prompt tokens into sampler for penalty tracking
  for (llama_token const t : priming) {
    llama_sampler_accept(sampler.get(), t);
  }

  if (!priming.empty()) {
    ctx.decode(priming, /*return_logits=*/true);
  }

  for (int i = 0; i < max_new_tokens; ++i) {
    // llama_sampler_sample (called by generate_next) already accepts the token
    llama_token const token = ctx.generate_next(sampler, -1);
    if (token == eos_token || token == LLAMA_TOKEN_NULL) {
      break;
    }
    if (!stop_tokens.empty() && std::ranges::find(stop_tokens, token) != stop_tokens.end()) {
      break;
    }
    output.push_back(token);
    ctx.decode_one(token, /*request_logits=*/true);
  }
  return output;
}

// Logging control -----------------------------------------------------------
std::atomic<ggml_log_level> g_min_log_level{GGML_LOG_LEVEL_INFO};

void log_filter_bridge(ggml_log_level level, const char* text, void* /*user*/) {
  if (level < g_min_log_level.load(std::memory_order_relaxed)) return;
  std::fputs(text, stderr);
  std::fflush(stderr);
}

void set_log_level(int min_level) {
  g_min_log_level.store(static_cast<ggml_log_level>(min_level), std::memory_order_relaxed);
  llama_log_set(log_filter_bridge, nullptr);
}

void disable_logging() {
  llama_log_set([](ggml_log_level, const char*, void*) {}, nullptr);
}

void reset_logging() {
  llama_log_set(nullptr, nullptr);
}

// Chat template helper
std::string chat_apply_template([[maybe_unused]] const Model& model,
                                const std::vector<std::pair<std::string, std::string>>& messages,
                                const std::string& tmpl, bool add_generation_prompt) {
  std::vector<llama_chat_message> chat_msgs;
  chat_msgs.reserve(messages.size());
  for (const auto& m : messages) {
    chat_msgs.push_back({m.first.c_str(), m.second.c_str()});
  }

  const char* tmpl_ptr = tmpl.empty() ? nullptr : tmpl.c_str();

  // First call to get required size
  int32_t const needed = llama_chat_apply_template(tmpl_ptr, chat_msgs.data(), chat_msgs.size(),
                                                   add_generation_prompt, nullptr, 0);

  if (needed < 0) {
    throw std::runtime_error("llama_chat_apply_template failed");
  }

  std::string result(static_cast<size_t>(needed) + 1, '\0');
  int32_t const written =
      llama_chat_apply_template(tmpl_ptr, chat_msgs.data(), chat_msgs.size(), add_generation_prompt,
                                result.data(), static_cast<int32_t>(result.size()));

  if (written < 0) {
    throw std::runtime_error("llama_chat_apply_template failed on second call");
  }
  result.resize(static_cast<size_t>(written));
  return result;
}

// Grammar sampler wrapper
class GrammarSampler {
 public:
  GrammarSampler(const Model& model, const std::string& grammar_str,
                 const std::string& grammar_root)
      : sampler_(
            llama_sampler_init_grammar(model.vocab(), grammar_str.c_str(), grammar_root.c_str())) {
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

  GrammarSampler(const GrammarSampler&) = delete;
  GrammarSampler& operator=(const GrammarSampler&) = delete;
  GrammarSampler(GrammarSampler&&) = delete;
  GrammarSampler& operator=(GrammarSampler&&) = delete;

  llama_sampler* get() const { return sampler_; }

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
  llama_sampler* sampler_ = nullptr;
};

// LoRA adapter wrapper
class LoraAdapter {
 public:
  LoraAdapter(Model& model, const std::string& path)
      : adapter_(llama_adapter_lora_init(model.get(), path.c_str())) {
    if (!adapter_) {
      throw std::runtime_error("failed to load LoRA adapter: " + path);
    }
  }

  // Note: adapters are freed automatically with the associated model (llama.cpp
  // API change)
  ~LoraAdapter() = default;

  LoraAdapter(const LoraAdapter&) = delete;
  LoraAdapter& operator=(const LoraAdapter&) = delete;
  LoraAdapter(LoraAdapter&&) = delete;
  LoraAdapter& operator=(LoraAdapter&&) = delete;

  llama_adapter_lora* get() const { return adapter_; }

 private:
  llama_adapter_lora* adapter_ = nullptr;
};

// Context LoRA methods (defined after LoraAdapter)
inline int32_t Context::set_adapters_lora(const nb::list& py_adapters, const nb::list& py_scales) {
  if (!ctx_) return -1;
  size_t const n = nb::len(py_adapters);
  if (n != nb::len(py_scales)) {
    throw std::invalid_argument("adapters and scales must have same length");
  }
  if (n == 0) {
    return llama_set_adapters_lora(ctx_, nullptr, 0, nullptr);
  }
  std::vector<llama_adapter_lora*> adapters(n);
  std::vector<float> scales(n);
  for (size_t i = 0; i < n; i++) {
    adapters[i] = nb::cast<LoraAdapter&>(py_adapters[i]).get();
    scales[i] = nb::cast<float>(py_scales[i]);
  }
  return llama_set_adapters_lora(ctx_, adapters.data(), n, scales.data());
}

struct TokenProb {
  llama_token token{};
  float logprob{};
  std::vector<std::pair<llama_token, float>> top_logprobs;
};

// softmax helpers
inline double logsumexp(const float* logits, int32_t n_vocab) {
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

inline std::vector<std::pair<llama_token, float>> compute_top_logprobs(const float* logits,
                                                                       int32_t n_vocab,
                                                                       int32_t top_n, double lse) {
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
    float const lp = static_cast<float>(double(logits[t]) - lse);
    out.emplace_back(t, lp);
  }
  return out;
}

std::vector<TokenProb> generate_tokens_with_details(
    Context& ctx, SamplerChain& sampler, const std::vector<llama_token>& prompt,
    int32_t max_new_tokens, bool add_bos, llama_token eos_token,
    const std::vector<std::vector<llama_token>>& stop_sequences, int32_t top_logprobs,
    bool echo_prompt) {
  std::vector<TokenProb> results;
  std::vector<llama_token> priming = prompt;
  if (add_bos && (priming.empty() || priming.front() != ctx.model().bos())) {
    priming.insert(priming.begin(), ctx.model().bos());
  }

  // Accept prompt tokens into sampler for penalty tracking
  for (llama_token const t : priming) {
    llama_sampler_accept(sampler.get(), t);
  }

  // process prompt
  if (!priming.empty()) {
    ctx.decode(priming, /*return_logits=*/true);
    if (echo_prompt) {
      for (const int i : priming) {
        TokenProb tp;
        tp.token = i;
        tp.logprob = std::numeric_limits<float>::quiet_NaN();
        results.push_back(std::move(tp));
      }
    }
  }

  // Index where generated (non-prompt) tokens begin in results.
  // Set AFTER echo block so stop-sequence removal never removes prompt tokens.
  const size_t generated_start = results.size();

  std::vector<llama_token> generated;
  generated.reserve(static_cast<size_t>(max_new_tokens));

  const int32_t n_vocab = ctx.model().n_vocab();
  // Allocate once outside the loop to avoid per-token heap allocation
  std::vector<llama_token_data> candidates(static_cast<size_t>(n_vocab));

  for (int i = 0; i < max_new_tokens; ++i) {
    const float* logits = llama_get_logits(ctx.raw());
    if (!logits) {
      throw std::runtime_error("logits unavailable before sampling");
    }

    // Build candidates and apply sampler to get adjusted probabilities
    for (int32_t j = 0; j < n_vocab; ++j) {
      candidates[static_cast<size_t>(j)] = {j, logits[j], 0.0F};
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

    // Use token selected by the apply above — do NOT call generate_next
    // (llama_sampler_sample) which would re-apply the sampler chain,
    // advancing the dist sampler's RNG and potentially selecting a
    // different token than what cur_p reflects.
    llama_token token = LLAMA_TOKEN_NULL;
    if (cur_p.size > 0 && cur_p.selected >= 0 && static_cast<size_t>(cur_p.selected) < cur_p.size) {
      token = cur_p.data[cur_p.selected].id;
    }
    // Accept into sampler for penalty tracking
    if (token >= 0) {
      llama_sampler_accept(sampler.get(), token);
    }

    // Check for EOS/NULL before accessing logits to avoid out-of-bounds read
    if (token == eos_token || token == LLAMA_TOKEN_NULL || token < 0 || token >= n_vocab) {
      break;
    }

    // The token logit comes directly from cur_p (post-sampler), which is
    // consistent with the lse computed above from cur_p values.
    float const token_logit = cur_p.data[static_cast<size_t>(cur_p.selected)].logit;

    TokenProb tp;
    tp.token = token;
    tp.logprob = static_cast<float>(double(token_logit) - lse);
    if (top_logprobs > 0) {
      std::vector<std::pair<llama_token, float>> top_lp;
      std::vector<size_t> idx(cur_p.size);
      std::iota(idx.begin(), idx.end(), 0);
      size_t const n = std::min(static_cast<size_t>(top_logprobs), cur_p.size);
      std::partial_sort(
          idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(n), idx.end(),
          [&](size_t a, size_t b) { return cur_p.data[a].logit > cur_p.data[b].logit; });
      for (size_t j = 0; j < n; ++j) {
        float const lp = static_cast<float>(double(cur_p.data[idx[j]].logit) - lse);
        top_lp.emplace_back(cur_p.data[idx[j]].id, lp);
      }
      tp.top_logprobs = std::move(top_lp);
    }
    results.push_back(std::move(tp));

    generated.push_back(token);

    // stop sequence check on generated tokens
    bool matched_stop = false;
    size_t remove_n = 0;
    for (const auto& seq : stop_sequences) {
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
        if (results.size() > generated_start) {
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
std::vector<llama_token> generate_tokens_with_grammar(Context& ctx, SamplerChain& sampler,
                                                      GrammarSampler& grammar,
                                                      const std::vector<llama_token>& prompt,
                                                      int32_t max_new_tokens, bool add_bos,
                                                      llama_token eos_token,
                                                      const std::vector<llama_token>& stop_tokens) {
  std::vector<llama_token> output;
  output.reserve(static_cast<size_t>(max_new_tokens));

  std::vector<llama_token> priming = prompt;
  if (add_bos && (priming.empty() || priming.front() != ctx.model().bos())) {
    priming.insert(priming.begin(), ctx.model().bos());
  }

  // Accept prompt tokens into sampler for penalty tracking
  for (llama_token const t : priming) {
    llama_sampler_accept(sampler.get(), t);
  }

  if (!priming.empty()) {
    ctx.decode(priming, /*return_logits=*/true);
  }

  const int32_t n_vocab = ctx.model().n_vocab();
  // Allocate once outside the loop to avoid per-token heap allocation
  std::vector<llama_token_data> candidates(static_cast<size_t>(n_vocab));

  for (int i = 0; i < max_new_tokens; ++i) {
    float* logits = llama_get_logits(ctx.raw());
    if (!logits) {
      throw std::runtime_error("logits unavailable");
    }

    // Build token data array for grammar sampling
    for (int32_t j = 0; j < n_vocab; ++j) {
      candidates[static_cast<size_t>(j)] = {j, logits[j], 0.0F};
    }
    llama_token_data_array cur_p = {candidates.data(), static_cast<size_t>(n_vocab), -1, false};

    // Apply grammar constraint first (masks invalid tokens)
    llama_sampler_apply(grammar.get(), &cur_p);

    // Apply sampler chain (temperature, top_k, top_p, etc.) to grammar-filtered
    // candidates
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
        if (token == st) {
          should_stop = true;
          break;
        }
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
    Context& ctx, SamplerChain& sampler, const std::vector<llama_token>& prompt,
    int32_t max_new_tokens, bool add_bos, llama_token eos_token,
    const std::vector<std::vector<llama_token>>& stop_sequences) {
  std::vector<llama_token> output;
  output.reserve(static_cast<size_t>(max_new_tokens));

  std::vector<llama_token> priming = prompt;
  if (add_bos && (priming.empty() || priming.front() != ctx.model().bos())) {
    priming.insert(priming.begin(), ctx.model().bos());
  }

  // Accept prompt tokens into sampler for penalty tracking
  for (llama_token const t : priming) {
    llama_sampler_accept(sampler.get(), t);
  }

  if (!priming.empty()) {
    ctx.decode(priming, /*return_logits=*/true);
  }

  for (int i = 0; i < max_new_tokens; ++i) {
    // llama_sampler_sample (called by generate_next) already accepts the token
    llama_token const token = ctx.generate_next(sampler, -1);
    if (token == eos_token || token == LLAMA_TOKEN_NULL) {
      break;
    }
    output.push_back(token);

    // Check multi-token stop sequences
    bool matched = false;
    for (const auto& seq : stop_sequences) {
      if (seq.empty() || seq.size() > output.size()) continue;
      if (std::equal(seq.rbegin(), seq.rend(), output.rbegin())) {
        matched = true;
        output.erase(output.end() - static_cast<std::ptrdiff_t>(seq.size()), output.end());
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
    Context& ctx, SamplerChain& sampler, GrammarSampler& grammar,
    const std::vector<llama_token>& prompt, int32_t max_new_tokens, bool add_bos,
    llama_token eos_token, const std::vector<std::vector<llama_token>>& stop_sequences) {
  std::vector<llama_token> output;
  output.reserve(static_cast<size_t>(max_new_tokens));

  std::vector<llama_token> priming = prompt;
  if (add_bos && (priming.empty() || priming.front() != ctx.model().bos())) {
    priming.insert(priming.begin(), ctx.model().bos());
  }

  // Accept prompt tokens into sampler for penalty tracking
  for (llama_token const t : priming) {
    llama_sampler_accept(sampler.get(), t);
  }

  if (!priming.empty()) {
    ctx.decode(priming, /*return_logits=*/true);
  }

  const int32_t n_vocab = ctx.model().n_vocab();
  // Allocate once outside the loop to avoid per-token heap allocation
  std::vector<llama_token_data> candidates(static_cast<size_t>(n_vocab));

  for (int i = 0; i < max_new_tokens; ++i) {
    float* logits = llama_get_logits(ctx.raw());
    if (!logits) {
      throw std::runtime_error("logits unavailable");
    }

    for (int32_t j = 0; j < n_vocab; ++j) {
      candidates[static_cast<size_t>(j)] = {j, logits[j], 0.0F};
    }
    llama_token_data_array cur_p = {candidates.data(), static_cast<size_t>(n_vocab), -1, false};

    // Apply grammar constraint first (masks invalid tokens)
    llama_sampler_apply(grammar.get(), &cur_p);

    // Apply sampler chain (temperature, top_k, top_p, etc.) to grammar-filtered
    // candidates
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
    for (const auto& seq : stop_sequences) {
      if (seq.empty() || seq.size() > output.size()) continue;
      if (std::equal(seq.rbegin(), seq.rend(), output.rbegin())) {
        matched = true;
        output.erase(output.end() - static_cast<std::ptrdiff_t>(seq.size()), output.end());
        break;
      }
    }
    if (matched) break;

    ctx.decode_one(token, /*request_logits=*/true);
  }
  return output;
}

// Streaming generation with callback - yields tokens as they're generated
// Returns total number of tokens generated.
// GIL is released for heavy C++ work (decode, sampling) and only re-acquired
// around the Python callback, allowing the main thread to process the queue.
//
// Multi-token stop sequence handling: tokens are buffered up to the length of
// the longest stop sequence before being yielded. This prevents partial stop
// sequence tokens from reaching the consumer.
int32_t generate_tokens_streaming(Context& ctx, SamplerChain& sampler,
                                  const std::vector<llama_token>& prompt, int32_t max_new_tokens,
                                  bool add_bos, llama_token eos_token,
                                  const std::vector<std::vector<llama_token>>& stop_sequences,
                                  const std::function<bool(llama_token)>& callback) {
  // Release GIL for the duration of C++ computation.
  // Re-acquire only when calling the Python callback.
  nb::gil_scoped_release const release;

  std::vector<llama_token> output;
  output.reserve(static_cast<size_t>(max_new_tokens));

  // Find max stop sequence length for buffering.
  // Tokens within max_stop_len of the end could be part of a stop sequence,
  // so we only yield tokens that are further back than this threshold.
  size_t max_stop_len = 0;
  for (const auto& seq : stop_sequences) {
    max_stop_len = std::max(max_stop_len, seq.size());
  }
  size_t n_yielded = 0;  // Number of tokens already yielded via callback

  std::vector<llama_token> priming = prompt;
  if (add_bos && (priming.empty() || priming.front() != ctx.model().bos())) {
    priming.insert(priming.begin(), ctx.model().bos());
  }

  for (llama_token const t : priming) {
    llama_sampler_accept(sampler.get(), t);
  }

  if (!priming.empty()) {
    ctx.decode(priming, /*return_logits=*/true);
  }

  // Helper: yield tokens that are safe (too far from the end to be part of
  // any stop sequence). Returns false if callback requested cancellation.
  auto yield_safe_tokens = [&]() -> bool {
    size_t const safe = output.size() > max_stop_len ? output.size() - max_stop_len : 0;
    while (n_yielded < safe) {
      nb::gil_scoped_acquire const gil;
      if (!callback(output[n_yielded])) {
        return false;
      }
      ++n_yielded;
    }
    return true;
  };

  for (int i = 0; i < max_new_tokens; ++i) {
    // llama_sampler_sample (called by generate_next) already accepts the token
    llama_token const token = ctx.generate_next(sampler, -1);

    if (token == eos_token || token == LLAMA_TOKEN_NULL) {
      break;
    }

    output.push_back(token);

    // Check multi-token stop sequences
    bool matched = false;
    size_t remove_n = 0;
    for (const auto& seq : stop_sequences) {
      if (seq.empty() || seq.size() > output.size()) continue;
      if (std::equal(seq.rbegin(), seq.rend(), output.rbegin())) {
        matched = true;
        remove_n = seq.size();
        break;
      }
    }

    if (matched) {
      // Remove stop tokens; buffering guarantees none were yielded
      output.erase(output.end() - static_cast<std::ptrdiff_t>(remove_n), output.end());
      break;
    }

    // Yield tokens that are confirmed safe (outside stop sequence window)
    if (!yield_safe_tokens()) {
      break;  // Callback returned False, stop generation
    }

    ctx.decode_one(token, /*request_logits=*/true);
  }

  // Flush remaining buffered tokens. These are tokens that were held back
  // by the stop-sequence window but never completed a match. Partial stop
  // sequence prefixes at end-of-generation are NOT treated as stops — only
  // complete matches trigger removal. So these tokens are valid output.
  while (n_yielded < output.size()) {
    nb::gil_scoped_acquire const gil;
    if (!callback(output[n_yielded])) {
      break;
    }
    ++n_yielded;
  }

  return static_cast<int32_t>(output.size());
}

}  // namespace

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-cstyle-cast)
NB_MODULE(_llama, m) {
  m.doc() = "High-performance nanobind bindings for llama.cpp";

  nb::class_<ModelParams>(m, "ModelParams", "Parameters for loading a model")
      .def(nb::init<>())
      .def_prop_rw(
          "n_gpu_layers", [](ModelParams& p) { return p.raw.n_gpu_layers; },
          [](ModelParams& p, int32_t v) { p.raw.n_gpu_layers = v; },
          "Number of layers to offload to GPU (-1 = all)")
      .def_prop_rw(
          "main_gpu", [](ModelParams& p) { return p.raw.main_gpu; },
          [](ModelParams& p, int32_t v) { p.raw.main_gpu = v; },
          "Main GPU index for multi-GPU setups")
      .def_prop_rw(
          "split_mode", [](ModelParams& p) { return p.raw.split_mode; },
          [](ModelParams& p, int32_t v) { p.raw.split_mode = static_cast<llama_split_mode>(v); },
          "How to split model across GPUs")
      .def_prop_rw(
          "vocab_only", [](ModelParams& p) { return p.raw.vocab_only; },
          [](ModelParams& p, bool v) { p.raw.vocab_only = v; }, "Load only vocabulary, no weights")
      .def_prop_rw(
          "use_mmap", [](ModelParams& p) { return p.raw.use_mmap; },
          [](ModelParams& p, bool v) { p.raw.use_mmap = v; }, "Use memory-mapped file for model")
      .def_prop_rw(
          "use_mlock", [](ModelParams& p) { return p.raw.use_mlock; },
          [](ModelParams& p, bool v) { p.raw.use_mlock = v; }, "Lock model in RAM")
      .def_prop_rw(
          "check_tensors", [](ModelParams& p) { return p.raw.check_tensors; },
          [](ModelParams& p, bool v) { p.raw.check_tensors = v; }, "Validate tensor data on load")
      .def_prop_rw(
          "no_host", [](ModelParams& p) { return p.raw.no_host; },
          [](ModelParams& p, bool v) { p.raw.no_host = v; },
          "Don't allocate host memory for tensors")
      .def(
          "as_dict",
          [](const ModelParams& p) {
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
          },
          "Convert parameters to dictionary");

  nb::class_<ContextParams>(m, "ContextParams", "Parameters for creating an inference context")
      .def(nb::init<>())
      .def_prop_rw(
          "n_ctx", [](ContextParams& p) { return p.raw.n_ctx; },
          [](ContextParams& p, uint32_t v) { p.raw.n_ctx = v; }, "Context size (max tokens)")
      .def_prop_rw(
          "n_batch", [](ContextParams& p) { return p.raw.n_batch; },
          [](ContextParams& p, uint32_t v) { p.raw.n_batch = v; },
          "Batch size for prompt processing")
      .def_prop_rw(
          "n_ubatch", [](ContextParams& p) { return p.raw.n_ubatch; },
          [](ContextParams& p, uint32_t v) { p.raw.n_ubatch = v; }, "Micro-batch size")
      .def_prop_rw(
          "n_seq_max", [](ContextParams& p) { return p.raw.n_seq_max; },
          [](ContextParams& p, uint32_t v) { p.raw.n_seq_max = v; }, "Max number of sequences")
      .def_prop_rw(
          "n_threads", [](ContextParams& p) { return p.raw.n_threads; },
          [](ContextParams& p, int32_t v) { p.raw.n_threads = v; }, "Threads for generation")
      .def_prop_rw(
          "n_threads_batch", [](ContextParams& p) { return p.raw.n_threads_batch; },
          [](ContextParams& p, int32_t v) { p.raw.n_threads_batch = v; },
          "Threads for batch processing")
      .def_prop_rw(
          "rope_freq_base", [](ContextParams& p) { return p.raw.rope_freq_base; },
          [](ContextParams& p, float v) { p.raw.rope_freq_base = v; }, "RoPE base frequency")
      .def_prop_rw(
          "rope_freq_scale", [](ContextParams& p) { return p.raw.rope_freq_scale; },
          [](ContextParams& p, float v) { p.raw.rope_freq_scale = v; }, "RoPE frequency scale")
      .def_prop_rw(
          "embeddings", [](ContextParams& p) { return p.raw.embeddings; },
          [](ContextParams& p, bool v) { p.raw.embeddings = v; }, "Enable embedding extraction")
      .def_prop_rw(
          "offload_kqv", [](ContextParams& p) { return p.raw.offload_kqv; },
          [](ContextParams& p, bool v) { p.raw.offload_kqv = v; }, "Offload KQV to GPU")
      .def_prop_rw(
          "flash_attn_type", [](ContextParams& p) { return p.raw.flash_attn_type; },
          [](ContextParams& p, int v) {
            p.raw.flash_attn_type = static_cast<llama_flash_attn_type>(v);
          },
          "Flash attention type (0=disabled)")
      .def_prop_rw(
          "type_k", [](ContextParams& p) { return static_cast<int>(p.raw.type_k); },
          [](ContextParams& p, int v) { p.raw.type_k = static_cast<ggml_type>(v); },
          "Data type for K cache (ggml_type enum, e.g. 1=f16, 3=q4_1)")
      .def_prop_rw(
          "type_v", [](ContextParams& p) { return static_cast<int>(p.raw.type_v); },
          [](ContextParams& p, int v) { p.raw.type_v = static_cast<ggml_type>(v); },
          "Data type for V cache (ggml_type enum, e.g. 1=f16, 3=q4_1)")
      .def(
          "as_dict",
          [](const ContextParams& p) {
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
            d["type_k"] = static_cast<int>(p.raw.type_k);
            d["type_v"] = static_cast<int>(p.raw.type_v);
            return d;
          },
          "Convert parameters to dictionary");

  nb::class_<Model>(m, "Model", "Loaded LLM model")
      .def(nb::init<const std::string&, const ModelParams&>(), "path"_a, "params"_a,
           nb::call_guard<nb::gil_scoped_release>(), "Load model from GGUF file")
      .def("close", &Model::close, "Explicitly free model resources")
      .def("n_vocab", &Model::n_vocab, "Vocabulary size")
      .def("n_ctx_train", &Model::n_ctx_train, "Training context size")
      .def("desc", &Model::desc, "Model description string")
      .def("tokenize", &Model::tokenize, "text"_a, nb::kw_only(), "add_special"_a = true,
           "parse_special"_a = false, nb::call_guard<nb::gil_scoped_release>(),
           "Convert text to tokens")
      .def("detokenize", &Model::detokenize, "tokens"_a, nb::kw_only(), "remove_special"_a = true,
           "unparse_special"_a = false, nb::call_guard<nb::gil_scoped_release>(),
           "Convert tokens to text")
      .def("detokenize_bytes", &Model::detokenize_bytes, "tokens"_a, nb::kw_only(),
           "remove_special"_a = true, "unparse_special"_a = false,
           "Convert tokens to raw bytes (no UTF-8 validation)")
      .def("bos", &Model::bos, "Beginning-of-sequence token ID")
      .def("eos", &Model::eos, "End-of-sequence token ID")
      .def("eot", &Model::eot, "End-of-turn token ID")
      .def("sep", &Model::sep, "Separator token ID")
      .def("nl", &Model::nl, "Newline token ID")
      .def("pad", &Model::pad, "Padding token ID")
      .def("get_add_bos", &Model::get_add_bos, "Whether model prefers BOS token to be added")
      .def("n_embd", &Model::n_embd, "Embedding dimension")
      .def("meta_count", &Model::meta_count, "Number of metadata entries")
      .def("meta_val_str", &Model::meta_val_str, "key"_a, "Get metadata value by key")
      .def("meta_key_by_index", &Model::meta_key_by_index, "index"_a, "Get metadata key by index")
      .def("meta_val_by_index", &Model::meta_val_by_index, "index"_a, "Get metadata value by index")
      .def("model_size", &Model::model_size, "Model size in bytes")
      .def("n_params", &Model::n_params, "Number of parameters")
      .def("n_layer", &Model::n_layer, "Number of layers")
      .def("n_head", &Model::n_head, "Number of attention heads")
      .def("has_encoder", &Model::has_encoder, "Whether model has an encoder component")
      .def("has_decoder", &Model::has_decoder, "Whether model has a decoder component")
      .def("is_recurrent", &Model::is_recurrent, "Whether model uses recurrent architecture")
      .def("is_hybrid", &Model::is_hybrid, "Whether model uses hybrid attention architecture")
      .def("chat_template", &Model::chat_template, "name"_a = "", "Get chat template string")
      .def("token_to_piece", &Model::token_to_piece, "token"_a, "Convert single token to text");

  nb::class_<SamplerChain::Params>(m, "SamplerParams", "Sampling parameters for text generation")
      .def(nb::init<>())
      .def_rw("top_k", &SamplerChain::Params::top_k, "Top-K sampling (0 = disabled)")
      .def_rw("top_p", &SamplerChain::Params::top_p, "Top-P (nucleus) sampling")
      .def_rw("min_p", &SamplerChain::Params::min_p, "Min-P sampling threshold")
      .def_rw("min_keep", &SamplerChain::Params::min_keep, "Minimum tokens to keep")
      .def_rw("temp", &SamplerChain::Params::temp, "Temperature (1.0 = neutral)")
      .def_rw("penalty_last_n", &SamplerChain::Params::penalty_last_n,
              "Tokens to consider for penalties")
      .def_rw("repeat_penalty", &SamplerChain::Params::repeat_penalty,
              "Repetition penalty (1.0 = disabled)")
      .def_rw("freq_penalty", &SamplerChain::Params::freq_penalty, "Frequency penalty")
      .def_rw("presence_penalty", &SamplerChain::Params::presence_penalty, "Presence penalty")
      .def_rw("seed", &SamplerChain::Params::seed, "RNG seed (-1 = random)")
      .def_rw("temp_delta", &SamplerChain::Params::temp_delta,
              "Dynamic temperature delta (0 = disabled)")
      .def_rw("temp_exponent", &SamplerChain::Params::temp_exponent, "Dynamic temperature exponent")
      .def_rw("xtc_probability", &SamplerChain::Params::xtc_probability,
              "XTC probability (0 = disabled)")
      .def_rw("xtc_threshold", &SamplerChain::Params::xtc_threshold, "XTC threshold")
      .def_rw("top_n_sigma", &SamplerChain::Params::top_n_sigma,
              "Top-n-sigma threshold (negative = disabled)")
      .def_rw("dry_multiplier", &SamplerChain::Params::dry_multiplier,
              "DRY multiplier (0 = disabled)")
      .def_rw("dry_base", &SamplerChain::Params::dry_base, "DRY base")
      .def_rw("dry_allowed_length", &SamplerChain::Params::dry_allowed_length,
              "DRY minimum repeat length")
      .def_rw("dry_penalty_last_n", &SamplerChain::Params::dry_penalty_last_n,
              "DRY window size (-1 = context size)")
      .def_rw("dry_seq_breakers", &SamplerChain::Params::dry_seq_breakers,
              "DRY sequence breaker strings");

  nb::class_<SamplerChain>(m, "SamplerChain", "Sampler chain for token selection")
      .def(nb::init<const Model&, const SamplerChain::Params&>(), "model"_a, "params"_a,
           nb::keep_alive<1, 2>(), "Create sampler chain")
      .def("reset", &SamplerChain::reset, "Reset sampler state")
      .def("sample", &SamplerChain::sample, "ctx"_a, nb::arg("idx") = -1,
           "Sample next token from logits");

  nb::class_<Context>(m, "Context", "Inference context with KV cache")
      .def(nb::init<Model&, const ContextParams&>(), "model"_a, "params"_a, nb::keep_alive<1, 2>(),
           "Create inference context")
      .def("close", &Context::close, "Explicitly free context resources")
      .def("n_ctx", &Context::n_ctx, "Current context size")
      .def("set_thread_count", &Context::set_thread_count, "n_threads"_a, "n_threads_batch"_a,
           "Set thread counts")
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
      .def("save_state", &Context::save_state, "path"_a, nb::call_guard<nb::gil_scoped_release>(),
           "Save context state to file")
      .def("load_state", &Context::load_state, "path"_a, nb::call_guard<nb::gil_scoped_release>(),
           "Load context state from file")
      .def("get_state_data", &Context::get_state_data,
           "Get state as bytes (returns Python bytes directly)")
      .def("set_state_data", &Context::set_state_data, "data"_a,
           "Set state from bytes (accepts Python bytes directly)")
      .def("set_adapters_lora", &Context::set_adapters_lora, "adapters"_a, "scales"_a,
           "Set LoRA adapters with scales (replaces all)")
      .def("clear_lora", &Context::clear_lora, "Remove all LoRA adapters")
      .def("perf", &Context::perf, "Get performance metrics dict")
      .def("perf_reset", &Context::perf_reset, "Reset performance counters")
      .def("kv_cache_clear", &Context::kv_cache_clear, "Clear entire KV cache")
      .def("kv_cache_seq_rm", &Context::kv_cache_seq_rm, "seq_id"_a, "p0"_a = -1, "p1"_a = -1,
           "Remove KV cache for sequence")
      .def("kv_cache_seq_cp", &Context::kv_cache_seq_cp, "seq_id_src"_a, "seq_id_dst"_a,
           "p0"_a = -1, "p1"_a = -1, "Copy KV cache between sequences")
      .def("kv_cache_seq_keep", &Context::kv_cache_seq_keep, "seq_id"_a,
           "Keep only specified sequence")
      .def("kv_cache_seq_add", &Context::kv_cache_seq_add, "seq_id"_a, "p0"_a, "p1"_a, "delta"_a,
           "Add position delta to sequence")
      .def("kv_cache_seq_pos_max", &Context::kv_cache_seq_pos_max, "seq_id"_a = 0,
           "Get max position in sequence")
      .def("kv_cache_seq_pos_min", &Context::kv_cache_seq_pos_min, "seq_id"_a = 0,
           "Get min position in sequence")
      .def("memory_can_shift", &Context::memory_can_shift,
           "Whether memory supports KV cache shifting")
      .def("set_embeddings", &Context::set_embeddings, "enabled"_a,
           "Enable or disable embedding extraction at runtime")
      .def("set_causal_attn", &Context::set_causal_attn, "enabled"_a,
           "Enable or disable causal attention at runtime");

  nb::class_<LoraAdapter>(m, "LoraAdapter", "LoRA adapter for model fine-tuning")
      .def(nb::init<Model&, const std::string&>(), "model"_a, "path"_a, nb::keep_alive<1, 2>(),
           "Load LoRA adapter from file");

  m.def("generate_tokens", &generate_tokens, "ctx"_a, "sampler"_a, "prompt"_a, "max_new_tokens"_a,
        "add_bos"_a, "eos_token"_a, "stop_tokens"_a = std::vector<llama_token>{},
        nb::call_guard<nb::gil_scoped_release>(),
        "Generate tokens using sampler chain. Returns list of token IDs.");

  nb::class_<TokenProb>(m, "TokenProb", "Token with probability information")
      .def_ro("token", &TokenProb::token, "Token ID")
      .def_ro("logprob", &TokenProb::logprob, "Log probability")
      .def_ro("top_logprobs", &TokenProb::top_logprobs, "Top alternative tokens with logprobs");

  m.def("generate_tokens_with_details", &generate_tokens_with_details, "ctx"_a, "sampler"_a,
        "prompt"_a, "max_new_tokens"_a, "add_bos"_a, "eos_token"_a,
        "stop_sequences"_a = std::vector<std::vector<llama_token>>{}, "top_logprobs"_a = 0,
        "echo_prompt"_a = false, nb::call_guard<nb::gil_scoped_release>(),
        "Generate tokens with per-token logprobs. Returns list of TokenProb.");

  // logging controls
  m.def("set_log_level", &set_log_level, "min_level"_a,
        "Set minimum log level (0=none, 1=debug, 2=info, 3=warn, 4=error)");
  m.def("disable_logging", &disable_logging, "Disable all llama.cpp logging");
  m.def("reset_logging", &reset_logging, "Restore default llama.cpp logging");
  m.def(
      "print_system_info", []() { return std::string(llama_print_system_info()); },
      "Return llama.cpp system info string (CPU features, build info, etc.).");

  // Chat template
  m.def("chat_apply_template", &chat_apply_template, "model"_a, "messages"_a, "tmpl"_a = "",
        "add_generation_prompt"_a = true,
        "Apply chat template to messages. Returns formatted prompt string.");

  // Grammar sampler
  nb::class_<GrammarSampler>(m, "GrammarSampler")
      .def(nb::init<const Model&, const std::string&, const std::string&>(), "model"_a,
           "grammar_str"_a, "grammar_root"_a = "root")
      .def("accept", &GrammarSampler::accept, "token"_a)
      .def("reset", &GrammarSampler::reset);

  m.def("generate_tokens_with_grammar", &generate_tokens_with_grammar, "ctx"_a, "sampler"_a,
        "grammar"_a, "prompt"_a, "max_new_tokens"_a, "add_bos"_a, "eos_token"_a,
        "stop_tokens"_a = std::vector<llama_token>{}, nb::call_guard<nb::gil_scoped_release>(),
        "Generation with grammar constraint");

  m.def("generate_tokens_multi_stop", &generate_tokens_multi_stop, "ctx"_a, "sampler"_a, "prompt"_a,
        "max_new_tokens"_a, "add_bos"_a, "eos_token"_a,
        "stop_sequences"_a = std::vector<std::vector<llama_token>>{},
        nb::call_guard<nb::gil_scoped_release>(), "Generation with multi-token stop sequences");

  m.def("generate_tokens_grammar_multi_stop", &generate_tokens_grammar_multi_stop, "ctx"_a,
        "sampler"_a, "grammar"_a, "prompt"_a, "max_new_tokens"_a, "add_bos"_a, "eos_token"_a,
        "stop_sequences"_a = std::vector<std::vector<llama_token>>{},
        nb::call_guard<nb::gil_scoped_release>(),
        "Generation with grammar and multi-token stop sequences");

  m.def("generate_tokens_streaming", &generate_tokens_streaming, "ctx"_a, "sampler"_a, "prompt"_a,
        "max_new_tokens"_a, "add_bos"_a, "eos_token"_a,
        "stop_sequences"_a = std::vector<std::vector<llama_token>>{}, "callback"_a,
        "Streaming generation with callback. Callback receives token, returns "
        "False to stop.");

  // Backend cleanup - call before interpreter shutdown to prevent segfault
  m.def(
      "backend_free",
      []() {
        if (g_model_count.load() == 0) {
          llama_backend_free();
        }
      },
      "Free llama.cpp backend resources. Only frees if no models are loaded.");

  m.def(
      "backend_can_free", []() -> bool { return g_model_count.load() == 0; },
      "Check if backend can be safely freed (no models loaded).");

  m.def(
      "model_count", []() -> int { return g_model_count.load(); },
      "Return number of currently loaded models.");
}
