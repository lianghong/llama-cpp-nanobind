#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "common.h"
#include "llama-ext.h"
#include "llama.h"
#include "speculative.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
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
// - g_backend_init_flag: std::call_once handles thread-safety internally (no mutex needed)
// - g_resource_mutex: protects model_count and resource lifecycle (Model/Context close)
std::once_flag g_backend_init_flag;
std::atomic<int> g_model_count{0};
std::mutex g_resource_mutex;  // For model lifecycle (close, model_count)

class Model {
 public:
  explicit Model(const std::string& path, const ModelParams& params)
      : model_(load_with_backend_init(path, params)) {
    if (!model_) {
      throw std::runtime_error("failed to load model: " + path);
    }
    std::scoped_lock const lock(g_resource_mutex);
    ++g_model_count;
  }

  ~Model() { close(); }

  void close() {
    std::scoped_lock const lock(g_resource_mutex);
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
    return read_c_string("llama_model_desc", [this](char* buf, size_t size) {
      return llama_model_desc(model_, buf, static_cast<int32_t>(size));
    });
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
    return read_c_string("llama_model_meta_val_str", [&](char* buf, size_t size) {
      return llama_model_meta_val_str(model_, key.c_str(), buf, static_cast<int32_t>(size));
    });
  }

  std::string meta_key_by_index(int32_t i) const {
    check_model();
    return read_c_string("llama_model_meta_key_by_index", [&](char* buf, size_t size) {
      return llama_model_meta_key_by_index(model_, i, buf, static_cast<int32_t>(size));
    });
  }

  std::string meta_val_by_index(int32_t i) const {
    check_model();
    return read_c_string("llama_model_meta_val_str_by_index", [&](char* buf, size_t size) {
      return llama_model_meta_val_str_by_index(model_, i, buf, static_cast<int32_t>(size));
    });
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
    // Validate cast didn't truncate
    if (static_cast<size_t>(n_tokens) != tokens.size()) {
      throw std::runtime_error("integer overflow in token count");
    }
    // Two-call protocol: first call with size=0 returns -required_bytes
    // (or 0 for empty output). Unlike llama_model_desc, the returned count
    // is the exact byte count, not null-terminated, so no +1 is needed.
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

  // Helper that initializes the llama backend (once, globally) and loads the
  // model. CUDA/Metal backends must be ready before llama_model_load_from_file,
  // so the call_once lives here and runs as part of member initialization.
  static llama_model* load_with_backend_init(const std::string& path, const ModelParams& params) {
    std::call_once(g_backend_init_flag, llama_backend_init);
    return llama_model_load_from_file(path.c_str(), params.raw);
  }

  // Read a C-string out of a llama.cpp API that uses the two-call snprintf
  // protocol: fn(buf=nullptr, size=0) returns the required byte count
  // (excluding the NUL terminator), then fn(buf, size) fills the buffer.
  //
  // Used by desc() / meta_val_str() / meta_key_by_index() / meta_val_by_index()
  // to avoid repeating the boilerplate four times.
  // Taken by value: we call the functor twice (size query + fill), which
  // requires a stable callable — no forwarding.
  template <typename Fn>
  static std::string read_c_string(const char* api_name, Fn fn) {
    int32_t const needed = fn(nullptr, 0);
    if (needed <= 0) {
      return "";
    }
    // Allocate needed+1 so the implementation has room for its own NUL
    // terminator; resize back to `needed` afterward (std::string keeps its
    // own internal terminator).
    std::string buf(static_cast<size_t>(needed) + 1, '\0');
    int32_t const written = fn(buf.data(), buf.size());
    if (written != needed) {
      throw std::runtime_error(std::string("buffer size mismatch in ") + api_name);
    }
    buf.resize(static_cast<size_t>(needed));
    return buf;
  }

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
    // Locally-typical sampling (Meister et al., arXiv:2202.00666).
    // 1.0 = disabled (matches llama.cpp convention).
    float typical_p = 1.0F;
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
    // Adaptive-P (terminal sampler; replaces dist when enabled).
    // target < 0 = disabled. Range: [0.0, 1.0] when enabled.
    // decay: EMA decay; valid range [0.0, 0.99]; history ≈ 1/(1-decay) tokens.
    float adaptive_p_target = -1.0F;
    float adaptive_p_decay = 0.85F;
    // Logit bias: list of (token_id, bias) entries. Empty = disabled.
    // Applied first in the chain so penalties/truncation see the biased logits.
    // Use -INFINITY (or a large negative) to ban a token.
    std::vector<std::pair<llama_token, float>> logit_bias;
  };

  SamplerChain(const Model& model, const Params& params) {
    auto chain_params = llama_sampler_chain_default_params();
    sampler_ = llama_sampler_chain_init(chain_params);
    if (!sampler_) {
      throw std::runtime_error("failed to create sampler chain");
    }

    // Canonical sampler ordering:
    // 0. Logit bias (mutates raw logits; runs before everything else so the
    //    biased values propagate through penalties / truncation / sampling)
    if (!params.logit_bias.empty()) {
      const int32_t n_vocab = model.n_vocab();
      std::vector<llama_logit_bias> entries;
      entries.reserve(params.logit_bias.size());
      for (const auto& [token, bias] : params.logit_bias) {
        if (token < 0 || token >= n_vocab) {
          throw std::out_of_range("logit_bias token id out of range [0, n_vocab)");
        }
        entries.push_back({token, bias});
      }
      llama_sampler_chain_add(
          sampler_, llama_sampler_init_logit_bias(n_vocab, static_cast<int32_t>(entries.size()),
                                                  entries.data()));
    }

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

    // 6b. Typical-p (locally-typical sampling). 1.0 = disabled.
    if (params.typical_p < 1.0F) {
      llama_sampler_chain_add(sampler_,
                              llama_sampler_init_typical(params.typical_p, params.min_keep));
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

    // 9. Terminal sampler: adaptive-p (if enabled) replaces dist; otherwise dist.
    // Both produce a single token id; only one may be the chain terminator.
    uint32_t const rng_seed = params.seed >= 0
                                  ? static_cast<uint32_t>(params.seed)
                                  : static_cast<uint32_t>(llama_time_us() & 0xFFFFFFFF);
    if (params.adaptive_p_target >= 0.0F) {
      llama_sampler_chain_add(
          sampler_, llama_sampler_init_adaptive_p(params.adaptive_p_target, params.adaptive_p_decay,
                                                  rng_seed));
    } else {
      llama_sampler_chain_add(sampler_, llama_sampler_init_dist(rng_seed));
    }

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
    multi_batch_ = llama_batch_init(kMultiBatchCapacity, 0, 1);
    llama_set_n_threads(ctx_, params_.raw.n_threads, params_.raw.n_threads_batch);
  }

  ~Context() { close(); }

  void close() {
    std::scoped_lock const lock(g_resource_mutex);
    if (single_batch_.token) {
      llama_batch_free(single_batch_);
      single_batch_ = {};
    }
    if (multi_batch_.token) {
      llama_batch_free(multi_batch_);
      multi_batch_ = {};
    }
    if (ctx_dft_) {
      llama_free(ctx_dft_);
      ctx_dft_ = nullptr;
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
    // Hold resource mutex to serialize with close() — both free ctx_.
    std::scoped_lock const lock(g_resource_mutex);
    if (!model_) {
      throw std::runtime_error("context has been closed");
    }
    if (ctx_dft_) {
      llama_free(ctx_dft_);
      ctx_dft_ = nullptr;
    }
    if (ctx_) {
      llama_free(ctx_);
      ctx_ = nullptr;  // Null immediately after free to prevent double-free
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
    if (!multi_batch_.token) {
      multi_batch_ = llama_batch_init(kMultiBatchCapacity, 0, 1);
    }
  }

  // Lazily create the MTP draft context against the same model. Idempotent:
  // returns the existing ctx_dft_ on subsequent calls. Throws if the model
  // doesn't expose an MTP graph variant. The draft context is freed by
  // close() / reset() / destructor.
  llama_context* ensure_mtp_draft_context() {
    check_ctx();
    if (ctx_dft_ != nullptr) {
      return ctx_dft_;
    }
    if (!model_) {
      throw std::runtime_error("context has been closed");
    }
    llama_context_params dparams = params_.raw;
    dparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    dparams.embeddings = false;
    // The draft context must support recurrent-state rollback so we can trim
    // rejected drafts from its KV. Mirror the target's n_rs_seq (set on the
    // user-facing context via LlamaConfig.n_rs_seq); if it's zero (the
    // pre-MTP default), bump to 2 to match the default SamplingParams.n_draft_max.
    if (dparams.n_rs_seq < 2) {
      dparams.n_rs_seq = 2;
    }
    // Cap ctx_dft's worst-case graph size. The verify batch is at most
    // n_draft_max+1 (≤ 9 tokens), and prefill can chunk through ubatches.
    // common_speculative_init enables backend top-k sampling on ctx_dft, which
    // forces a second sched_reserve(); leaving n_ubatch at the user's default
    // (often 512) allocates two ~500 MiB compute buffers and risks CUDA OOM
    // even on cards with plenty of free VRAM (fragmentation / contiguous-range
    // requirement). 64 is comfortably above n_draft_max+1 for any allowed
    // value (range [1, 8]) and shrinks the graph ~8x.
    constexpr uint32_t kDraftUBatch = 64;
    if (dparams.n_ubatch == 0 || dparams.n_ubatch > kDraftUBatch) {
      dparams.n_ubatch = kDraftUBatch;
    }
    if (dparams.n_batch < dparams.n_ubatch) {
      dparams.n_batch = dparams.n_ubatch;
    }
    ctx_dft_ = llama_init_from_model(model_->get(), dparams);
    if (!ctx_dft_) {
      throw std::runtime_error(
          "failed to create MTP draft context (model has no MTP graph variant?)");
    }
    llama_set_n_threads(ctx_dft_, dparams.n_threads, dparams.n_threads_batch);
    return ctx_dft_;
  }

  llama_context* mtp_draft_context_or_null() const { return ctx_dft_; }

  int32_t cur_pos() const { return cur_pos_; }

  void advance_cur_pos(int32_t n) { cur_pos_ += n; }

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

  // Decode multiple tokens into a single forward pass. All positions request
  // logits (the speculative verify step needs them). Reuses multi_batch_.
  int32_t decode_multi(const std::vector<llama_token>& tokens) {
    check_ctx();
    if (tokens.empty()) return 0;
    if (static_cast<int32_t>(tokens.size()) > kMultiBatchCapacity) {
      throw std::runtime_error("decode_multi: batch size " + std::to_string(tokens.size()) +
                               " exceeds capacity " + std::to_string(kMultiBatchCapacity));
    }
    multi_batch_.n_tokens = static_cast<int32_t>(tokens.size());
    for (int32_t i = 0; i < multi_batch_.n_tokens; ++i) {
      multi_batch_.token[i] = tokens[static_cast<size_t>(i)];
      multi_batch_.pos[i] = cur_pos_ + i;
      multi_batch_.n_seq_id[i] = 1;
      multi_batch_.seq_id[i][0] = 0;
      multi_batch_.logits[i] = 1;
    }
    int32_t const rc = llama_decode(ctx_, multi_batch_);
    if (rc < 0) {
      throw std::runtime_error("llama_decode (multi) failed with code " + std::to_string(rc));
    }
    cur_pos_ += static_cast<int32_t>(tokens.size());
    return multi_batch_.n_tokens;
  }

  std::vector<float> logits() const {
    check_ctx();
    if (!model_) {
      throw std::runtime_error("context has been closed");
    }
    const int32_t n_vocab = model_->n_vocab();
    const float* ptr = llama_get_logits(const_cast<llama_context*>(ctx_));
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
    const float* ptr = llama_get_embeddings(const_cast<llama_context*>(ctx_));
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
    cur_pos_ = std::max(cur_pos_, 0);
    return n_token_count;
  }

  // Returns state as Python bytes. Writes directly into the Python bytes
  // buffer to avoid the extra heap allocation + memcpy that a
  // `std::vector<uint8_t>` -> `nb::bytes` copy would incur. For large KV
  // states this can save hundreds of MB of intermediate memory.
  nb::bytes get_state_data() {
    check_ctx();
    size_t const size = llama_state_get_size(ctx_);
    // Allocate an uninitialized Python bytes object (GIL held here).
    PyObject* py_obj = PyBytes_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(size));
    if (!py_obj) {
      throw std::runtime_error("failed to allocate Python bytes buffer for state");
    }
    char* buf = PyBytes_AsString(py_obj);
    if (!buf) {
      Py_DECREF(py_obj);
      throw std::runtime_error("failed to access Python bytes buffer");
    }
    size_t written = 0;
    {
      nb::gil_scoped_release const release;
      written = llama_state_get_data(ctx_, reinterpret_cast<uint8_t*>(buf), size);
    }
    // Shrink the bytes object if the serializer wrote fewer bytes than the
    // reserved size. _PyBytes_Resize requires GIL (which we hold here).
    if (written < size) {
      if (_PyBytes_Resize(&py_obj, static_cast<Py_ssize_t>(written)) != 0) {
        // py_obj is set to nullptr on failure.
        throw std::runtime_error("failed to resize Python bytes buffer");
      }
    }
    // nb::steal consumes the reference we already hold.
    return nb::steal<nb::bytes>(py_obj);
  }

  // Accepts Python bytes directly (pointer access, no per-element conversion).
  // GIL is managed manually: pointer extracted while held, released for heavy
  // C++ work.
  //
  // Failure semantics: if llama_state_set_data fails, the Python-side cur_pos_
  // counter is reset to its prior value, but the underlying llama KV cache may
  // have been partially overwritten by the time the failure was detected and
  // is therefore in an indeterminate state. Callers should treat the context
  // as invalid on failure and call reset() (or discard the instance) before
  // further use. Snapshotting the full KV state for true rollback is not done
  // here because it would double peak memory on every load.
  size_t set_state_data(const nb::bytes& data) {
    check_ctx();
    // Use the nb::bytes buffer directly — no copy. Lifetime argument:
    //   * `data` is a `const nb::bytes&` bound to the caller's Python bytes
    //     object. That object is kept alive by a strong reference on the
    //     calling Python frame's stack.
    //   * Dropping that reference requires executing Python bytecode, which
    //     requires the GIL.
    //   * Our thread is paused inside this C call for the duration of the
    //     GIL-released block — the calling frame cannot pop, so its
    //     reference cannot be released.
    //   * Another thread that takes the GIL could execute unrelated Python
    //     code, but cannot observe or mutate this particular reference.
    // Therefore the pointer stays valid for the whole scope. Copying the
    // buffer (~KV state size, often multi-GB) would double peak memory for
    // no safety gain.
    const auto* ptr = static_cast<const uint8_t*>(data.data());
    size_t const len = data.size();
    size_t result = 0;
    int32_t const old_pos = cur_pos_;  // Best-effort restore on failure
    {
      nb::gil_scoped_release const release;
      result = llama_state_set_data(ctx_, ptr, len);
      if (result == 0 || result > len) {
        // Failure - restore Python-side position. Note: the llama KV cache
        // itself may be partially overwritten; this only resets bookkeeping.
        cur_pos_ = old_pos;
      } else {
        // Success - update cur_pos_ from KV cache to maintain correct position bookkeeping
        cur_pos_ = kv_cache_seq_pos_max(0) + 1;
        cur_pos_ = std::max(cur_pos_, 0);
      }
    }
    if (result == 0 || result > len) {
      throw std::runtime_error(
          "failed to load state data (KV cache may be in an indeterminate "
          "state; call reset() before reuse)");
    }
    return result;
  }

  // Per-sequence on-device state save/load (llama.cpp 2026-04+).
  //
  // Uses LLAMA_STATE_SEQ_FLAGS_ON_DEVICE: tensor data stays in device buffers
  // (GPU memory) instead of being copied to host. The returned bytes are an
  // opaque handle/header that references the device-resident slot — they are
  // NOT a host-serializable copy of the KV cache.
  //
  // CRITICAL invariant from llama.h:
  //   "Getting the state for a seq_id with this flag invalidates all prior
  //    states gotten for that seq_id with this flag."
  // Only one on-device snapshot per seq_id may be live at a time. Using a
  // stale handle after a re-save for the same seq_id is undefined behavior.
  //
  // For host-serializable / multi-snapshot state, use get_state_data /
  // set_state_data (whole context, returns real bytes).
  nb::bytes save_seq_state_on_device(int32_t seq_id) {
    check_ctx();
    constexpr llama_state_seq_flags flag = LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;
    size_t const size = llama_state_seq_get_size_ext(ctx_, seq_id, flag);
    PyObject* py_obj = PyBytes_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(size));
    if (!py_obj) {
      throw std::runtime_error("failed to allocate Python bytes buffer for on-device state");
    }
    char* buf = PyBytes_AsString(py_obj);
    if (!buf) {
      Py_DECREF(py_obj);
      throw std::runtime_error("failed to access Python bytes buffer");
    }
    size_t written = 0;
    {
      nb::gil_scoped_release const release;
      written =
          llama_state_seq_get_data_ext(ctx_, reinterpret_cast<uint8_t*>(buf), size, seq_id, flag);
    }
    if (written < size) {
      if (_PyBytes_Resize(&py_obj, static_cast<Py_ssize_t>(written)) != 0) {
        throw std::runtime_error("failed to resize Python bytes buffer");
      }
    }
    return nb::steal<nb::bytes>(py_obj);
  }

  // Restore an on-device snapshot previously produced by save_seq_state_on_device.
  // Same lifetime argument as set_state_data: `data` is kept alive by the
  // calling Python frame for the duration of this call.
  size_t load_seq_state_on_device(const nb::bytes& data, int32_t dest_seq_id) {
    check_ctx();
    constexpr llama_state_seq_flags flag = LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;
    const auto* ptr = static_cast<const uint8_t*>(data.data());
    size_t const len = data.size();
    size_t result = 0;
    int32_t const old_pos = cur_pos_;
    {
      nb::gil_scoped_release const release;
      result = llama_state_seq_set_data_ext(ctx_, ptr, len, dest_seq_id, flag);
      if (result == 0 || result > len) {
        cur_pos_ = old_pos;
      } else if (dest_seq_id == 0) {
        // Same bookkeeping as load_state / set_state_data: only seq 0 is
        // tracked by cur_pos_.
        cur_pos_ = kv_cache_seq_pos_max(0) + 1;
        cur_pos_ = std::max(cur_pos_, 0);
      }
    }
    if (result == 0 || result > len) {
      throw std::runtime_error(
          "failed to load on-device sequence state (KV cache may be in an "
          "indeterminate state; call reset() before reuse)");
    }
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

  // True iff the model exposes an MTP graph variant. Probes by trying to
  // create the draft context lazily (cached on success). Returns false on
  // any failure so the predicate is safe to call from precondition checks.
  bool supports_speculative_mtp() {
    if (!ctx_ || !model_) return false;
    if (ctx_dft_ != nullptr) return true;
    try {
      ensure_mtp_draft_context();
      return true;
    } catch (...) {
      return false;
    }
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
  // Lazy MTP draft context — same model, ctx_type=LLAMA_CONTEXT_TYPE_MTP.
  // Created on first speculative call, freed by close()/reset()/dtor.
  llama_context* ctx_dft_ = nullptr;
  ContextParams params_;
  int32_t cur_pos_ = 0;
  llama_batch single_batch_ = {};  // Reusable single-token batch for decode_one
  // Reusable multi-token batch for the speculative draft-verify loop. Sized
  // for `1 + max(n_draft_max)` = 1 + 8 = 9. n_tokens is set per call.
  static constexpr int32_t kMultiBatchCapacity = 9;
  llama_batch multi_batch_ = {};

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

// Helper: Prime generation by adding BOS, accepting tokens into sampler, and decoding prompt.
// Returns the primed token sequence (with BOS if added).
// This is the common setup boilerplate for all generate_tokens_* functions.
//
// skip_decode_prefix: number of leading tokens of the (post-BOS) priming
// sequence whose KV state is already present in the context (longest-common-
// prefix reuse). Sampler still accepts the full priming for penalty tracking,
// but only `priming[skip_decode_prefix:]` is fed to llama_decode. Caller is
// responsible for ensuring the cached prefix actually matches what's in the
// KV cache (e.g. via Llama._cached_prompt_tokens).
inline std::vector<llama_token> prime_generation(Context& ctx, SamplerChain& sampler,
                                                 const std::vector<llama_token>& prompt,
                                                 bool add_bos, int32_t skip_decode_prefix = 0) {
  // Build priming with BOS prepended in a single pass — avoids the O(n)
  // memmove that std::vector::insert(begin(), ...) would cost for large
  // prompts.
  const bool need_bos = add_bos && (prompt.empty() || prompt.front() != ctx.model().bos());
  std::vector<llama_token> priming;
  priming.reserve(prompt.size() + (need_bos ? 1 : 0));
  if (need_bos) {
    priming.push_back(ctx.model().bos());
  }
  priming.insert(priming.end(), prompt.begin(), prompt.end());

  // Accept all prompt tokens into sampler for penalty tracking. This must
  // cover the full priming — penalty windows depend on the entire history,
  // not just the suffix that gets decoded.
  for (llama_token const t : priming) {
    llama_sampler_accept(sampler.get(), t);
  }

  // Validate skip range. Negative skip is treated as 0; skip >= priming.size()
  // means the caller already has the entire priming in KV — nothing to decode.
  // We still need cur_pos_ to reflect that prefix; caller (Python wrapper) is
  // responsible for ensuring kv_cache_seq_rm / cur_pos_ are consistent before
  // calling.
  const int32_t skip = std::max<int32_t>(0, skip_decode_prefix);
  const auto priming_size = static_cast<int32_t>(priming.size());
  if (skip < priming_size) {
    const std::vector<llama_token> suffix(priming.begin() + skip, priming.end());
    ctx.decode(suffix, /*return_logits=*/true);
  }

  return priming;
}

std::vector<llama_token> generate_tokens(Context& ctx, SamplerChain& sampler,
                                         const std::vector<llama_token>& prompt,
                                         int32_t max_new_tokens, bool add_bos,
                                         llama_token eos_token,
                                         const std::vector<llama_token>& stop_tokens,
                                         int32_t skip_decode_prefix = 0) {
  std::vector<llama_token> output;
  output.reserve(static_cast<size_t>(max_new_tokens));

  prime_generation(ctx, sampler, prompt, add_bos, skip_decode_prefix);

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
std::mutex g_log_mutex;  // Protects llama_log_set() calls

void log_filter_bridge(ggml_log_level level, const char* text, void* /*user*/) {
  if (level < g_min_log_level.load(std::memory_order_relaxed)) return;
  std::fputs(text, stderr);
  std::fflush(stderr);
}

void set_log_level(int min_level) {
  std::scoped_lock<std::mutex> const lock(g_log_mutex);
  g_min_log_level.store(static_cast<ggml_log_level>(min_level), std::memory_order_relaxed);
  llama_log_set(log_filter_bridge, nullptr);
}

void disable_logging() {
  std::scoped_lock<std::mutex> const lock(g_log_mutex);
  llama_log_set([](ggml_log_level, const char*, void*) {}, nullptr);
}

void reset_logging() {
  std::scoped_lock<std::mutex> const lock(g_log_mutex);
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

  // Lazy-grammar constructor: grammar activates on the first match of any
  // trigger pattern (regex against generated text from the start) or any
  // trigger token. Until activation, all tokens pass through unconstrained.
  // See llama.cpp PR #9639 / llama_sampler_init_grammar_lazy_patterns.
  GrammarSampler(const Model& model, const std::string& grammar_str,
                 const std::string& grammar_root, const std::vector<std::string>& trigger_patterns,
                 const std::vector<llama_token>& trigger_tokens) {
    std::vector<const char*> pattern_ptrs;
    pattern_ptrs.reserve(trigger_patterns.size());
    for (const auto& p : trigger_patterns) {
      pattern_ptrs.push_back(p.c_str());
    }
    sampler_ = llama_sampler_init_grammar_lazy_patterns(
        model.vocab(), grammar_str.c_str(), grammar_root.c_str(), pattern_ptrs.data(),
        pattern_ptrs.size(), trigger_tokens.data(), trigger_tokens.size());
    if (!sampler_) {
      throw std::runtime_error("failed to create lazy grammar sampler - check grammar syntax");
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
      : parent_(&model), adapter_(llama_adapter_lora_init(model.get(), path.c_str())) {
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
  const Model* parent() const { return parent_; }

 private:
  const Model* parent_ = nullptr;  // Non-owning: Model outlives adapter via nb::keep_alive
  llama_adapter_lora* adapter_ = nullptr;
};

// Context LoRA methods (defined after LoraAdapter)
inline int32_t Context::set_adapters_lora(const nb::list& py_adapters, const nb::list& py_scales) {
  if (!ctx_) return -1;
  size_t const n = nb::len(py_adapters);
  if (n != nb::len(py_scales)) {
    throw std::invalid_argument("adapters and scales must have same length");
  }
  if (n > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    throw std::invalid_argument("too many LoRA adapters");
  }
  if (n == 0) {
    return llama_set_adapters_lora(ctx_, nullptr, 0, nullptr);
  }
  std::vector<llama_adapter_lora*> adapters(n);
  std::vector<float> scales(n);
  for (size_t i = 0; i < n; i++) {
    auto& adapter = nb::cast<LoraAdapter&>(py_adapters[i]);
    if (adapter.parent() != model_) {
      throw std::invalid_argument(
          "LoRA adapter was loaded for a different model; cannot apply to this context");
    }
    adapters[i] = adapter.get();
    scales[i] = nb::cast<float>(py_scales[i]);
  }
  return llama_set_adapters_lora(ctx_, adapters.data(), static_cast<int32_t>(n), scales.data());
}

struct TokenProb {
  llama_token token{};
  float logprob{};
  std::vector<std::pair<llama_token, float>> top_logprobs;
};

std::vector<TokenProb> generate_tokens_with_details(
    Context& ctx, SamplerChain& sampler, const std::vector<llama_token>& prompt,
    int32_t max_new_tokens, bool add_bos, llama_token eos_token,
    const std::vector<std::vector<llama_token>>& stop_sequences, int32_t top_logprobs,
    bool echo_prompt, int32_t skip_decode_prefix = 0) {
  std::vector<TokenProb> results;
  std::vector<llama_token> const priming =
      prime_generation(ctx, sampler, prompt, add_bos, skip_decode_prefix);

  // Echo prompt tokens if requested
  if (echo_prompt && !priming.empty()) {
    for (const llama_token tok : priming) {
      TokenProb tp;
      tp.token = tok;
      tp.logprob = std::numeric_limits<float>::quiet_NaN();
      results.push_back(std::move(tp));
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

    // Build candidates and apply sampler to get adjusted probabilities.
    // Rebuild fully each iteration: samplers (top_k, top_p, ...) reorder
    // the candidates array in place via cur_p.data.
    for (int32_t j = 0; j < n_vocab; ++j) {
      candidates[static_cast<size_t>(j)] = {.id = j, .logit = logits[j], .p = 0.0F};
    }
    llama_token_data_array cur_p = {.data = candidates.data(),
                                    .size = static_cast<size_t>(n_vocab),
                                    .selected = -1,
                                    .sorted = false};
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
        sum += std::exp(static_cast<double>(cur_p.data[j].logit - max_l));
      }
      lse = std::log(sum) + static_cast<double>(max_l);
    }

    // Use token selected by the apply above — do NOT call generate_next
    // (llama_sampler_sample) which would re-apply the sampler chain,
    // advancing the dist sampler's RNG and potentially selecting a
    // different token than what cur_p reflects.
    // Validate selected index before ANY use. cur_p.selected is int64_t;
    // std::cmp_greater_equal compares signed/unsigned without UB.
    if (cur_p.size == 0 || cur_p.selected < 0 ||
        std::cmp_greater_equal(cur_p.selected, cur_p.size)) {
      throw std::runtime_error(
          "sampler failed to select valid token (empty candidate set after filtering?)");
    }
    llama_token const token = cur_p.data[cur_p.selected].id;

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
    tp.logprob = static_cast<float>(static_cast<double>(token_logit) - lse);
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
                                                      const std::vector<llama_token>& stop_tokens,
                                                      int32_t skip_decode_prefix = 0) {
  std::vector<llama_token> output;
  output.reserve(static_cast<size_t>(max_new_tokens));

  prime_generation(ctx, sampler, prompt, add_bos, skip_decode_prefix);

  const int32_t n_vocab = ctx.model().n_vocab();
  // Allocate once outside the loop to avoid per-token heap allocation
  std::vector<llama_token_data> candidates(static_cast<size_t>(n_vocab));

  for (int i = 0; i < max_new_tokens; ++i) {
    const float* logits = llama_get_logits(ctx.raw());
    if (!logits) {
      throw std::runtime_error("logits unavailable");
    }

    // Build token data array for grammar sampling
    for (int32_t j = 0; j < n_vocab; ++j) {
      candidates[static_cast<size_t>(j)] = {.id = j, .logit = logits[j], .p = 0.0F};
    }
    llama_token_data_array cur_p = {.data = candidates.data(),
                                    .size = static_cast<size_t>(n_vocab),
                                    .selected = -1,
                                    .sorted = false};

    // Apply grammar constraint first (masks invalid tokens)
    llama_sampler_apply(grammar.get(), &cur_p);

    // Apply sampler chain (temperature, top_k, top_p, etc.) to grammar-filtered
    // candidates
    llama_sampler_apply(sampler.get(), &cur_p);

    // Select token from the sampled distribution
    llama_token token = LLAMA_TOKEN_NULL;
    if (cur_p.size > 0 && cur_p.selected >= 0 && std::cmp_less(cur_p.selected, cur_p.size)) {
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
    const std::vector<std::vector<llama_token>>& stop_sequences, int32_t skip_decode_prefix = 0) {
  std::vector<llama_token> output;
  output.reserve(static_cast<size_t>(max_new_tokens));

  prime_generation(ctx, sampler, prompt, add_bos, skip_decode_prefix);

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
    llama_token eos_token, const std::vector<std::vector<llama_token>>& stop_sequences,
    int32_t skip_decode_prefix = 0) {
  std::vector<llama_token> output;
  output.reserve(static_cast<size_t>(max_new_tokens));

  prime_generation(ctx, sampler, prompt, add_bos, skip_decode_prefix);

  const int32_t n_vocab = ctx.model().n_vocab();
  // Allocate once outside the loop to avoid per-token heap allocation
  std::vector<llama_token_data> candidates(static_cast<size_t>(n_vocab));

  for (int i = 0; i < max_new_tokens; ++i) {
    const float* logits = llama_get_logits(ctx.raw());
    if (!logits) {
      throw std::runtime_error("logits unavailable");
    }

    for (int32_t j = 0; j < n_vocab; ++j) {
      candidates[static_cast<size_t>(j)] = {.id = j, .logit = logits[j], .p = 0.0F};
    }
    llama_token_data_array cur_p = {.data = candidates.data(),
                                    .size = static_cast<size_t>(n_vocab),
                                    .selected = -1,
                                    .sorted = false};

    // Apply grammar constraint first (masks invalid tokens)
    llama_sampler_apply(grammar.get(), &cur_p);

    // Apply sampler chain (temperature, top_k, top_p, etc.) to grammar-filtered
    // candidates
    llama_sampler_apply(sampler.get(), &cur_p);

    // Select token from the sampled distribution
    llama_token token = LLAMA_TOKEN_NULL;
    if (cur_p.size > 0 && cur_p.selected >= 0 && std::cmp_less(cur_p.selected, cur_p.size)) {
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
                                  const std::function<bool(llama_token)>& callback,
                                  int32_t skip_decode_prefix = 0) {
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

  prime_generation(ctx, sampler, prompt, add_bos, skip_decode_prefix);

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
      // Break without decoding stop tokens (intentional):
      // - Stop tokens are NOT part of conversation history (correct for reset_kv_cache=False)
      // - cur_pos_ remains at position before stop tokens (KV cache consistency)
      // - Sampler has accepted stop tokens (for penalty tracking across generations)
      // This is the expected behavior for session continuation.
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

// Speculative draft-MTP generation. Builds a `common_speculative_*` instance
// over an MTP context, draws up to n_draft_max draft tokens per round, batches
// them through `decode_multi`, and verifies via the existing sampler chain.
//
// The exit/cleanup contract mirrors generate_tokens_multi_stop: stop tokens
// are NOT emitted to output, sampler accepts the full priming + verified
// tokens for penalty tracking, KV cache is left in a state where the next
// call can append.
//
// Streaming is signaled by a non-null callback. Returns the generated token
// vector when callback is null; when callback is non-null, returns an empty
// vector after the streaming loop ends (the consumer reads via the callback).
std::vector<llama_token> generate_tokens_speculative_mtp(
    Context& ctx, SamplerChain& sampler, GrammarSampler* grammar,
    const std::vector<llama_token>& prompt, int32_t max_new_tokens, bool add_bos,
    llama_token eos_token, int32_t n_draft_max,
    const std::vector<std::vector<llama_token>>& stop_sequences,
    std::function<bool(llama_token)> callback, int32_t skip_decode_prefix) {
  std::vector<llama_token> output;
  output.reserve(static_cast<size_t>(max_new_tokens));

  // === Architectural note ============================================
  // draft-MTP requires two llama_contexts against the same model:
  //   ctx_tgt (DEFAULT graph): generates verified logits
  //   ctx_dft (MTP graph):     produces draft tokens
  // The user-facing Context owns ctx_tgt; ctx_dft is created lazily on
  // first speculative call. The draft impl mirrors prompt/verify batches
  // into ctx_dft via common_speculative_process(); we trim ctx_dft's KV
  // in lockstep with ctx_tgt's so the recurrent state stays aligned.
  // ===================================================================

  llama_context* const ctx_tgt = ctx.raw();

  // --- Build the priming sequence and accept into sampler --------------
  // Done up-front: if priming is empty (empty prompt + add_bos=false) we
  // return without constructing ctx_dft / common_speculative — there's
  // nothing to draft against and we don't want to begin() a spec impl
  // we'll never use.
  const bool need_bos = add_bos && (prompt.empty() || prompt.front() != ctx.model().bos());
  std::vector<llama_token> priming;
  priming.reserve(prompt.size() + (need_bos ? 1 : 0));
  if (need_bos) priming.push_back(ctx.model().bos());
  priming.insert(priming.end(), prompt.begin(), prompt.end());
  if (priming.empty()) return output;

  for (llama_token const t : priming) {
    llama_sampler_accept(sampler.get(), t);
  }

  llama_context* const ctx_dft = ctx.ensure_mtp_draft_context();
  llama_memory_t const mem_dft = llama_get_memory(ctx_dft);

  common_params_speculative spec_params;
  spec_params.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
  spec_params.draft.n_max = n_draft_max;
  spec_params.draft.ctx_tgt = ctx_tgt;
  spec_params.draft.ctx_dft = ctx_dft;

  common_speculative_ptr spec(common_speculative_init(spec_params, /*n_seq=*/1));
  if (!spec) {
    throw std::runtime_error(
        "common_speculative_init returned null (model has MTP graph but "
        "draft-MTP impl is unavailable)");
  }

  // The draft-MTP ctor already calls llama_set_embeddings_pre_norm() on both
  // contexts. We RAII-guard them off on exit so subsequent non-speculative
  // generation on ctx_tgt isn't perturbed.
  struct PreNormGuard {
    llama_context* tgt;
    llama_context* dft;
    PreNormGuard(llama_context* t, llama_context* d) : tgt(t), dft(d) {}
    ~PreNormGuard() {
      if (tgt) llama_set_embeddings_pre_norm(tgt, false, false);
      if (dft) llama_set_embeddings_pre_norm(dft, false, false);
    }
    PreNormGuard(const PreNormGuard&) = delete;
    PreNormGuard& operator=(const PreNormGuard&) = delete;
    PreNormGuard(PreNormGuard&&) = delete;
    PreNormGuard& operator=(PreNormGuard&&) = delete;
  } const pn_guard{ctx_tgt, ctx_dft};

  // --- Sync ctx_dft with ctx_tgt -------------------------------------
  // ctx_dft is cached across calls and may carry KV from a prior generation.
  // The Python wrapper has already trimmed ctx_tgt's KV (e.g. cache_prompt
  // prefix-reuse, or no-op for reset_kv_cache=True after the higher-level
  // kv_cache_clear). Mirror that trim here so common_speculative_process's
  // ctx_dft batch positions align with KV_dft.
  {
    const int32_t tgt_keep = ctx.cur_pos();
    // dft_max == -1 when seq 0 is empty (llama.cpp convention). Then
    // dft_max + 1 == 0 ≤ any valid tgt_keep, so the trim correctly no-ops.
    const llama_pos dft_max = llama_memory_seq_pos_max(mem_dft, /*seq=*/0);
    if (dft_max + 1 > tgt_keep) {
      llama_memory_seq_rm(mem_dft, /*seq=*/0, /*p0=*/tgt_keep, /*p1=*/-1);
    }
  }

  // --- Decode the priming suffix (skip_decode_prefix already in KV) ---
  // We must mirror to ctx_dft via common_speculative_process(), so we can't
  // use prime_generation() — build the batch ourselves, decode on ctx_tgt,
  // then hand the batch to the impl.
  const int32_t skip = std::max<int32_t>(0, skip_decode_prefix);
  const auto priming_size = static_cast<int32_t>(priming.size());
  if (skip < priming_size) {
    const int32_t n_prime = priming_size - skip;
    llama_batch prime_batch = llama_batch_init(n_prime, 0, 1);
    struct BatchGuard {
      llama_batch& b;
      explicit BatchGuard(llama_batch& batch) : b(batch) {}
      ~BatchGuard() { llama_batch_free(b); }
      BatchGuard(const BatchGuard&) = delete;
      BatchGuard& operator=(const BatchGuard&) = delete;
      BatchGuard(BatchGuard&&) = delete;
      BatchGuard& operator=(BatchGuard&&) = delete;
    } const guard(prime_batch);

    prime_batch.n_tokens = n_prime;
    const int32_t pos_start = ctx.cur_pos();
    for (int32_t i = 0; i < n_prime; ++i) {
      prime_batch.token[i] = priming[static_cast<size_t>(skip + i)];
      prime_batch.pos[i] = pos_start + i;
      prime_batch.n_seq_id[i] = 1;
      prime_batch.seq_id[i][0] = 0;
      prime_batch.logits[i] = (i == n_prime - 1) ? 1 : 0;
    }
    int32_t const rc = llama_decode(ctx_tgt, prime_batch);
    if (rc < 0) {
      throw std::runtime_error("speculative: prompt llama_decode (tgt) failed code " +
                               std::to_string(rc));
    }
    ctx.advance_cur_pos(n_prime);

    if (!common_speculative_process(spec.get(), prime_batch)) {
      throw std::runtime_error("speculative: common_speculative_process(prompt) failed");
    }
  }

  std::vector<llama_token> mirror = priming;

  // Per-round draft output buffer. Reused across rounds.
  llama_tokens drafted;

  // Initialize the speculative impl's per-seq state. Must come AFTER the
  // prompt has been processed (so begin()'s pos_max check passes).
  common_speculative_begin(spec.get(), /*seq_id=*/0, mirror);

  const int32_t n_vocab = ctx.model().n_vocab();
  std::vector<llama_token_data> candidates(static_cast<size_t>(n_vocab));

  llama_token id_last = mirror.back();
  int32_t n_emitted = 0;  // tokens added to `output` (counts toward max_new_tokens)

  while (n_emitted < max_new_tokens) {
    // --- step 1: draft ---
    drafted.clear();
    common_speculative_draft_params& dp =
        common_speculative_get_draft_params(spec.get(), /*seq_id=*/0);
    dp.drafting = true;
    dp.n_max = n_draft_max;
    dp.n_past = static_cast<llama_pos>(mirror.size());
    dp.id_last = id_last;
    dp.prompt = &mirror;
    dp.result = &drafted;

    common_speculative_draft(spec.get());

    int32_t const k = static_cast<int32_t>(drafted.size());

    // --- step 2: build [id_last, drafted_0, ..., drafted_{k-1}] and decode once ---
    // We've already decoded id_last (it's the last accepted token from the
    // previous round, or the last priming token). To re-use its logits via
    // the verify pass and have downstream KV positions for drafts laid out
    // contiguously, we trim the KV at id_last's position on BOTH ctx_tgt
    // and ctx_dft, then re-decode it alongside the drafts and mirror the
    // batch into ctx_dft via common_speculative_process().
    int32_t const id_last_pos = static_cast<int32_t>(mirror.size()) - 1;
    if (!ctx.kv_cache_seq_rm(0, id_last_pos, -1)) {
      throw std::runtime_error(
          "speculative: kv_cache_seq_rm (tgt) failed (memory_can_shift=false?)");
    }
    if (!llama_memory_seq_rm(mem_dft, 0, id_last_pos, -1)) {
      throw std::runtime_error(
          "speculative: kv_cache_seq_rm (dft) failed (memory_can_shift=false?)");
    }

    // Build a verify llama_batch [id_last, drafted_0..drafted_{k-1}], decode
    // on ctx_tgt (request logits at every position), then mirror into ctx_dft
    // via common_speculative_process to keep the recurrent state aligned.
    llama_batch verify_batch = llama_batch_init(k + 1, 0, 1);
    struct VerifyBatchGuard {
      llama_batch& b;
      explicit VerifyBatchGuard(llama_batch& batch) : b(batch) {}
      ~VerifyBatchGuard() { llama_batch_free(b); }
      VerifyBatchGuard(const VerifyBatchGuard&) = delete;
      VerifyBatchGuard& operator=(const VerifyBatchGuard&) = delete;
      VerifyBatchGuard(VerifyBatchGuard&&) = delete;
      VerifyBatchGuard& operator=(VerifyBatchGuard&&) = delete;
    } const vguard(verify_batch);

    verify_batch.n_tokens = k + 1;
    verify_batch.token[0] = id_last;
    verify_batch.pos[0] = id_last_pos;
    verify_batch.n_seq_id[0] = 1;
    verify_batch.seq_id[0][0] = 0;
    verify_batch.logits[0] = 1;
    for (int32_t i = 0; i < k; ++i) {
      verify_batch.token[i + 1] = drafted[static_cast<size_t>(i)];
      verify_batch.pos[i + 1] = id_last_pos + 1 + i;
      verify_batch.n_seq_id[i + 1] = 1;
      verify_batch.seq_id[i + 1][0] = 0;
      verify_batch.logits[i + 1] = 1;
    }

    int32_t const verify_rc = llama_decode(ctx_tgt, verify_batch);
    if (verify_rc < 0) {
      throw std::runtime_error("speculative: llama_decode (verify) failed code " +
                               std::to_string(verify_rc));
    }
    // After kv_cache_seq_rm above, cur_pos_ was reset to id_last_pos. The
    // verify batch decoded k+1 tokens at positions [id_last_pos .. id_last_pos+k],
    // so the new "next position" is id_last_pos + k + 1.
    ctx.advance_cur_pos(k + 1);
    if (!common_speculative_process(spec.get(), verify_batch)) {
      throw std::runtime_error("speculative: common_speculative_process(verify) failed");
    }

    // --- step 3: verify, position by position ---
    int32_t accepted = 0;  // number of drafts accepted this round
    llama_token corrected_id = LLAMA_TOKEN_NULL;
    for (int32_t i = 0; i <= k; ++i) {
      const float* logits = llama_get_logits_ith(ctx_tgt, i);
      if (!logits) {
        throw std::runtime_error("speculative: logits unavailable at offset " + std::to_string(i));
      }
      for (int32_t j = 0; j < n_vocab; ++j) {
        candidates[static_cast<size_t>(j)] = {.id = j, .logit = logits[j], .p = 0.0F};
      }
      llama_token_data_array cur_p = {.data = candidates.data(),
                                      .size = static_cast<size_t>(n_vocab),
                                      .selected = -1,
                                      .sorted = false};
      if (grammar) llama_sampler_apply(grammar->get(), &cur_p);
      llama_sampler_apply(sampler.get(), &cur_p);
      llama_token verified = LLAMA_TOKEN_NULL;
      if (cur_p.size > 0 && cur_p.selected >= 0 && std::cmp_less(cur_p.selected, cur_p.size)) {
        verified = cur_p.data[cur_p.selected].id;
      }
      if (verified == LLAMA_TOKEN_NULL) {
        throw std::runtime_error("speculative: sampler returned no token (grammar emptied set?)");
      }

      // Accept this token into sampler / grammar for penalty tracking.
      llama_sampler_accept(sampler.get(), verified);
      if (grammar) llama_sampler_accept(grammar->get(), verified);

      if (i < k && verified == drafted[static_cast<size_t>(i)]) {
        accepted++;
        continue;
      }
      // Mismatch (or final position): record the corrected token and stop.
      corrected_id = verified;
      break;
    }

    // --- step 4: trim rejected drafts from KV on BOTH contexts ----------
    // Keep id_last + accepted drafts. The corrected token was sampled but
    // not yet decoded; it becomes `id_last` of the next iteration when the
    // verify batch runs.
    int32_t const n_keep = id_last_pos + 1 /*id_last*/ + accepted;
    int32_t const decoded_end = id_last_pos + 1 + k;  // one past last decoded pos
    if (n_keep < decoded_end) {
      if (!ctx.kv_cache_seq_rm(0, n_keep, -1)) {
        throw std::runtime_error("speculative: kv_cache_seq_rm (tgt reject trim) failed");
      }
      if (!llama_memory_seq_rm(mem_dft, 0, n_keep, -1)) {
        throw std::runtime_error("speculative: kv_cache_seq_rm (dft reject trim) failed");
      }
    }

    // --- step 5: tell the speculative context how many we accepted ---
    common_speculative_accept(spec.get(), /*seq_id=*/0, static_cast<uint16_t>(accepted));

    // --- step 6: emit accepted tokens then the corrected one ---
    auto emit = [&](llama_token tok) -> bool {
      // EOS / stop-token check (single-token).
      if (tok == eos_token || tok == LLAMA_TOKEN_NULL) return false;
      // Multi-token stop sequences: check after appending.
      output.push_back(tok);
      mirror.push_back(tok);
      n_emitted++;
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
        output.erase(output.end() - static_cast<std::ptrdiff_t>(remove_n), output.end());
        // Mirror keeps the stop tokens because they're in KV; Python will
        // handle the mirror invalidation/commit accordingly.
        return false;
      }
      if (callback) {
        nb::gil_scoped_acquire const gil;
        if (!callback(tok)) return false;
      }
      return n_emitted < max_new_tokens;
    };

    bool keep_going = true;
    for (int32_t a = 0; a < accepted && keep_going; ++a) {
      keep_going = emit(drafted[static_cast<size_t>(a)]);
    }
    if (keep_going) {
      keep_going = emit(corrected_id);
    }
    if (!keep_going) break;

    id_last = mirror.back();
  }

  // --- Tail trim: enforce the prompt-cache mirror invariant
  // (`len(_cached_prompt_tokens) == kv_pos_max + 1`).
  //
  // When the loop terminates mid-round (max_tokens cap, EOS, stop-sequence,
  // or callback abort), KV may contain accepted drafts past mirror.back()
  // that were never emitted. Trim KV to mirror's last position on both
  // contexts so callers using cache_prompt see KV aligned to the mirror.
  if (!mirror.empty()) {
    int32_t const tail_keep = static_cast<int32_t>(mirror.size());
    if (ctx.cur_pos() > tail_keep) {
      ctx.kv_cache_seq_rm(0, tail_keep, -1);
      llama_memory_seq_rm(mem_dft, 0, tail_keep, -1);
    }
  }

  return output;
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
          "n_rs_seq", [](ContextParams& p) { return p.raw.n_rs_seq; },
          [](ContextParams& p, uint32_t v) { p.raw.n_rs_seq = v; },
          "Recurrent-state snapshots per seq for rollback (0 = no rollback). "
          "Required by draft-MTP speculative decoding on hybrid recurrent "
          "models like Qwen3.6-MoE; should be set to >= n_draft_max.")
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
          "ctx_type", [](ContextParams& p) { return static_cast<int>(p.raw.ctx_type); },
          [](ContextParams& p, int v) { p.raw.ctx_type = static_cast<llama_context_type>(v); },
          "Context type (0=default, 1=MTP). MTP requires a model that ships "
          "Multi-Token Prediction layers (e.g. Qwen3.5 MTP variants); other "
          "models will fail context construction.")
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
            d["n_rs_seq"] = p.raw.n_rs_seq;
            d["n_threads"] = p.raw.n_threads;
            d["n_threads_batch"] = p.raw.n_threads_batch;
            d["rope_freq_base"] = p.raw.rope_freq_base;
            d["rope_freq_scale"] = p.raw.rope_freq_scale;
            d["embeddings"] = p.raw.embeddings;
            d["offload_kqv"] = p.raw.offload_kqv;
            d["flash_attn_type"] = p.raw.flash_attn_type;
            d["ctx_type"] = static_cast<int>(p.raw.ctx_type);
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
      .def_rw("typical_p", &SamplerChain::Params::typical_p,
              "Locally-typical sampling threshold (1.0 = disabled)")
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
              "DRY sequence breaker strings")
      .def_rw("adaptive_p_target", &SamplerChain::Params::adaptive_p_target,
              "Adaptive-P target probability (negative = disabled, replaces dist when enabled)")
      .def_rw("adaptive_p_decay", &SamplerChain::Params::adaptive_p_decay,
              "Adaptive-P EMA decay; history ~ 1/(1-decay) tokens (range 0.0-0.99)")
      .def_rw("logit_bias", &SamplerChain::Params::logit_bias,
              "List of (token_id, bias) pairs. Empty = disabled. "
              "Use -inf (or a large negative) to ban a token.");

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
      .def("decode_multi", &Context::decode_multi, "tokens"_a,
           nb::call_guard<nb::gil_scoped_release>(),
           "Decode multiple tokens in a single forward pass; all positions "
           "request logits.")
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
      .def("save_seq_state_on_device", &Context::save_seq_state_on_device, "seq_id"_a = 0,
           "Save per-sequence state with ON_DEVICE flag (opaque handle; "
           "previous on-device snapshot for the same seq_id is invalidated)")
      .def("load_seq_state_on_device", &Context::load_seq_state_on_device, "data"_a,
           "dest_seq_id"_a = 0,
           "Restore per-sequence on-device state (only valid for handles "
           "produced by save_seq_state_on_device on this context)")
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
      .def("supports_speculative_mtp", &Context::supports_speculative_mtp,
           "True iff the context was constructed with ctx_type=LLAMA_CONTEXT_TYPE_MTP")
      .def("set_embeddings", &Context::set_embeddings, "enabled"_a,
           "Enable or disable embedding extraction at runtime")
      .def("set_causal_attn", &Context::set_causal_attn, "enabled"_a,
           "Enable or disable causal attention at runtime");

  nb::class_<LoraAdapter>(m, "LoraAdapter", "LoRA adapter for model fine-tuning")
      .def(nb::init<Model&, const std::string&>(), "model"_a, "path"_a, nb::keep_alive<1, 2>(),
           "Load LoRA adapter from file");

  m.def("generate_tokens", &generate_tokens, "ctx"_a, "sampler"_a, "prompt"_a, "max_new_tokens"_a,
        "add_bos"_a, "eos_token"_a, "stop_tokens"_a = std::vector<llama_token>{},
        "skip_decode_prefix"_a = 0, nb::call_guard<nb::gil_scoped_release>(),
        "Generate tokens using sampler chain. Returns list of token IDs. "
        "skip_decode_prefix: leading tokens already in KV cache (prefix reuse).");

  nb::class_<TokenProb>(m, "TokenProb", "Token with probability information")
      .def_ro("token", &TokenProb::token, "Token ID")
      .def_ro("logprob", &TokenProb::logprob, "Log probability")
      .def_ro("top_logprobs", &TokenProb::top_logprobs, "Top alternative tokens with logprobs");

  m.def("generate_tokens_with_details", &generate_tokens_with_details, "ctx"_a, "sampler"_a,
        "prompt"_a, "max_new_tokens"_a, "add_bos"_a, "eos_token"_a,
        "stop_sequences"_a = std::vector<std::vector<llama_token>>{}, "top_logprobs"_a = 0,
        "echo_prompt"_a = false, "skip_decode_prefix"_a = 0,
        nb::call_guard<nb::gil_scoped_release>(),
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
      .def(nb::init<const Model&, const std::string&, const std::string&,
                    const std::vector<std::string>&, const std::vector<llama_token>&>(),
           "model"_a, "grammar_str"_a, "grammar_root"_a, "trigger_patterns"_a, "trigger_tokens"_a,
           "Lazy grammar: activates on the first match of any trigger pattern "
           "(regex against generated text from the start) or trigger token id.")
      .def("accept", &GrammarSampler::accept, "token"_a)
      .def("reset", &GrammarSampler::reset);

  m.def("generate_tokens_with_grammar", &generate_tokens_with_grammar, "ctx"_a, "sampler"_a,
        "grammar"_a, "prompt"_a, "max_new_tokens"_a, "add_bos"_a, "eos_token"_a,
        "stop_tokens"_a = std::vector<llama_token>{}, "skip_decode_prefix"_a = 0,
        nb::call_guard<nb::gil_scoped_release>(), "Generation with grammar constraint");

  m.def("generate_tokens_multi_stop", &generate_tokens_multi_stop, "ctx"_a, "sampler"_a, "prompt"_a,
        "max_new_tokens"_a, "add_bos"_a, "eos_token"_a,
        "stop_sequences"_a = std::vector<std::vector<llama_token>>{}, "skip_decode_prefix"_a = 0,
        nb::call_guard<nb::gil_scoped_release>(), "Generation with multi-token stop sequences");

  m.def("generate_tokens_grammar_multi_stop", &generate_tokens_grammar_multi_stop, "ctx"_a,
        "sampler"_a, "grammar"_a, "prompt"_a, "max_new_tokens"_a, "add_bos"_a, "eos_token"_a,
        "stop_sequences"_a = std::vector<std::vector<llama_token>>{}, "skip_decode_prefix"_a = 0,
        nb::call_guard<nb::gil_scoped_release>(),
        "Generation with grammar and multi-token stop sequences");

  m.def("generate_tokens_streaming", &generate_tokens_streaming, "ctx"_a, "sampler"_a, "prompt"_a,
        "max_new_tokens"_a, "add_bos"_a, "eos_token"_a,
        "stop_sequences"_a = std::vector<std::vector<llama_token>>{}, "callback"_a,
        "skip_decode_prefix"_a = 0,
        "Streaming generation with callback. Callback receives token, returns "
        "False to stop.");

  m.def("generate_tokens_speculative_mtp", &generate_tokens_speculative_mtp, "ctx"_a, "sampler"_a,
        "grammar"_a.none(), "prompt"_a, "max_new_tokens"_a, "add_bos"_a, "eos_token"_a,
        "n_draft_max"_a, "stop_sequences"_a = std::vector<std::vector<llama_token>>{},
        "callback"_a.none(), "skip_decode_prefix"_a = 0, nb::call_guard<nb::gil_scoped_release>(),
        "Speculative draft-MTP generation. grammar/callback may be None.");

  // Backend cleanup - call before interpreter shutdown to prevent segfault
  m.def(
      "backend_free",
      []() {
        std::scoped_lock const lock(g_resource_mutex);
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
