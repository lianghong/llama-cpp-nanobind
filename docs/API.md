# API Reference

## llama_cpp.Llama

High-level client wrapping the nanobind extension. Compatible with the `llama_cpp_python.Llama` surface where feasible. Supports context manager protocol for automatic resource cleanup.

**Constructor**

```python
Llama(model_path: str, config: LlamaConfig | None = None, sampling: SamplingParams | None = None)
```

- `model_path`: Path to a GGUF model file.
- `config`: Optional `LlamaConfig` to fine-tune context/model options.
- `sampling`: Default `SamplingParams` used for generation when no override is passed.

**Context Manager**

```python
with Llama("model.gguf") as llm:
    text = llm.generate("Hello", max_tokens=32)
# Resources automatically cleaned up
```

**Methods**

- `generate(prompt, max_tokens=128, sampling=None, stop=None, echo=False, logprobs=None, stream=False, seed=None, reset_kv_cache=True, cache_prompt=True, speculative=False, n_draft_max=None)` → `str | Iterator[str] | dict`
- `generate_stream(prompt, max_tokens=128, sampling=None, stop=None, seed=None, reset_kv_cache=True, cache_prompt=True, speculative=False, n_draft_max=None)` → `Iterator[str]` – True streaming (yields as tokens decode)
- `generate_async(...)` → Async version of generate
- `create_chat_completion(messages, max_tokens=128, stream=False, stop=None, response_format=None, grammar=None, tools=None, tool_choice=None, reset_kv_cache=True, cache_prompt=True, speculative=False, n_draft_max=None, **kwargs)` → Chat completion dict or stream
- `create_chat_completion_async(...)` → Async version
- `create_embedding(input, model=None)` → OpenAI-compatible embedding API (requires `embeddings=True`)
- `create_embedding_async(...)` → Async version
- `embed(text)` → `List[float]` – Simple embedding (requires `embeddings=True`)
- `embed_async(text)` → Async version
- `tokenize(text, add_special=True, parse_special=False)` → `List[int]`
- `detokenize(tokens, remove_special=True, unparse_special=False)` → `str`
- `n_tokens(text, add_special=False)` → `int` – Count tokens (set add_special=True to include BOS)
- `token_bos()`, `token_eos()`, `token_eot()` → Token IDs
- `n_ctx()`, `n_vocab()`, `n_embd()` → Model dimensions
- `model_size()`, `n_params()`, `n_layer()` → Model info
- `metadata` → `dict` – Model metadata (property)
- `get_chat_template(name="")` → `str`
- `token_to_piece(token)` → `str`
- `reset()` → Reset context/KV cache
- `close()` → Release resources (Context freed before Model for safety)
- `save_state(path)`, `load_state(path)` → State persistence
- `get_state()`, `set_state(data)` → State as bytes
- `load_lora(path, scale=1.0)`, `remove_lora(adapter)`, `clear_lora()` → LoRA management
- `perf()`, `perf_reset()` → Performance metrics
- `n_head()` → `int` – Number of attention heads
- `has_encoder()`, `has_decoder()` → `bool` – Architecture type
- `is_recurrent()`, `is_hybrid()` → `bool` – Architecture variant (e.g., Qwen3.5 hybrid)
- `token_sep()`, `token_nl()`, `token_pad()` → `int` – Special token IDs
- `get_add_bos()` → `bool` – Model's BOS preference
- `kv_cache_clear()` → Clear KV cache without recreating context
- `kv_cache_seq_rm()`, `kv_cache_seq_cp()`, `kv_cache_seq_keep()`, `kv_cache_seq_add()`, `kv_cache_seq_pos_max()` → KV cache management
- `kv_cache_seq_pos_min()` → `int` – Minimum position in KV cache for sequence
- `memory_can_shift()` → `bool` – Whether KV cache supports shifting
- `supports_speculative_mtp()` → `bool` – Whether the model exposes an MTP graph usable as a speculative draft context (gated on `<arch>.nextn_predict_layers > 0` metadata)
- `mtp_predict_layers()` → `int` – Declared next-token-prediction layer count from `<arch>.nextn_predict_layers` (0 for every non-MTP checkpoint)
- `set_embeddings(enabled)` → None – Toggle embedding computation at runtime
- `set_causal_attn(enabled)` → None – Toggle causal attention at runtime

### True Streaming

Use `generate_stream()` for true incremental streaming (yields tokens as they're decoded):

```python
for chunk in llm.generate_stream("Tell me a story", max_tokens=100):
    print(chunk, end="", flush=True)
```

#### Streaming behavior matrix

Not every `stream=True` entry point is fully incremental — a couple of
paths still buffer the entire completion before yielding. The table below
spells out the guarantees so you can size SSE/WebSocket timeouts correctly.

| Entry point | `stream=True` behavior |
|---|---|
| `generate_stream(...)` | **Incremental** — chunks emit as the worker thread decodes each token. |
| `generate(prompt, stream=True)` | **Buffered** — generation completes, then chunks are yielded. Use `generate_stream` if you need TTFT < total. |
| `generate_async(stream=True)` | **Incremental** — bridges the sync generator through `asyncio.Queue`. |
| `create_chat_completion(stream=True)` | **Incremental** in the default path; **buffered** when `grammar=` or `tools=` is set (the C++ grammar entry point is not incremental, and tool-call parsing needs the full message). |
| `create_chat_completion_async(stream=True)` | Same as the sync chat path: incremental by default, buffered when `grammar=` / `tools=` is set. |

Cancellation is safe on every incremental path: breaking out of the
generator (or `await stream.aclose()` on the async paths) signals the
worker to stop and releases `self._lock` before returning. After cancel,
the instance is reusable. If a worker thread fails to exit within
`Llama._STREAM_JOIN_TIMEOUT` seconds (because the underlying C++ call has
not returned), `Llama.is_stuck` flips to `True` and the lock is held
intentionally — restart the process to recover.

### Session-Style Continuation & Prompt-Prefix Cache Reuse

Two complementary knobs govern KV cache behavior between calls:

| Kwarg | Default | Effect |
|---|---|---|
| `reset_kv_cache` | `True` | Clear KV before this call. Set `False` to keep KV from prior turn. |
| `cache_prompt` | `True` | When `reset_kv_cache=False`, trim KV to the longest common prefix with the new prompt and decode only the divergent suffix. Ignored when `reset_kv_cache=True`. |

#### Recommended pattern: chat with automatic prefix reuse

```python
# First turn — fresh KV
llm.create_chat_completion(
    [{"role": "user", "content": "Hello"}],
    max_tokens=64,
    reset_kv_cache=True,
)

# Subsequent turns — keep prior KV, decode only the new user message
# (defaults are reset_kv_cache=False not yet wired in chat helper —
#  pass explicitly until upstream lands a session manager)
llm.create_chat_completion(
    [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
    ],
    max_tokens=64,
    reset_kv_cache=False,   # don't clear KV
    cache_prompt=True,      # trim to LCP, decode new tail only
)
```

For an 8K-token chat history with a 200-token user turn, prompt-decode time
drops from O(8K) to O(200) — roughly a 40× speedup on time-to-first-token.

#### Hybrid-attention models

Models reporting `memory_can_shift() == False` (Qwen3.5, Granite 4 hybrid,
some flash-attention configs) cannot trim KV mid-sequence. For these,
`cache_prompt=True` falls back to a full `kv_cache_clear()` + full reprime —
correctness is preserved, but the speedup is lost. Check `llm.memory_can_shift()`
to know whether you're on the fast path. The fallback is automatic; no
caller change is needed.

#### Mirror invariants

The `Llama` instance keeps an internal `_cached_prompt_tokens` mirror that
tracks what's actually in seq 0 of the KV cache. The mirror is invalidated
automatically by:

- `kv_cache_clear()`, `reset()`
- `kv_cache_seq_rm/cp/keep/add(...)` — direct escape-hatch APIs
- `set_state(data)`, `load_state(path)`
- `embed()` and `create_embedding()` (each one clears KV)
- `close()`

So mixing direct KV manipulation with `cache_prompt=True` is safe — the
next generation falls back to a clean full prime instead of trusting a
stale mirror.

#### Opting out

Pass `cache_prompt=False` to keep the legacy "append-only" semantics where
`reset_kv_cache=False` decodes the entire new prompt at the end of seq 0.
This is rarely what you want for chat, but it preserves bit-for-bit
compatibility with code written against pre-cache_prompt behavior.

```python
# Legacy stitching: KV grows monotonically; the model sees prior KV +
# the new prompt as one giant sequence.
llm.generate("Hello", max_tokens=10, reset_kv_cache=True)
llm.generate("World", max_tokens=10, reset_kv_cache=False, cache_prompt=False)
```

## Exceptions

```python
from llama_cpp import LlamaError, ModelLoadError, ValidationError, GenerationError
```

- `LlamaError` – Base exception for all llama-cpp-nanobind errors
- `ModelLoadError` – Failed to load model file
- `ValidationError` – Invalid input parameters
- `GenerationError` – Text generation failed

**Example:**

```python
from llama_cpp import Llama, ModelLoadError, ValidationError

try:
    llm = Llama("nonexistent.gguf")
except ModelLoadError as e:
    print(f"Model error: {e}")

try:
    llm.generate("test", max_tokens=0)
except ValidationError as e:
    print(f"Validation error: {e}")
```

## Module-level functions

- `print_system_info()` → `str` – llama.cpp system info
- `set_log_level(level)` → None – Set log level
- `disable_logging()` → None – Silence logging
- `reset_logging()` → None – Restore default logging
- `shutdown()` → None – Explicitly shutdown all Llama instances and free backend resources

### shutdown()

Call at the end of your program before `exit()` to avoid segfaults when using logging or other modules that hold references during cleanup.

```python
from llama_cpp import Llama, shutdown

def main():
    with Llama("model.gguf") as llm:
        text = llm.generate("Hello", max_tokens=32)
    shutdown()  # Clean up before Python's shutdown sequence

if __name__ == "__main__":
    main()
```

## Configuration dataclasses

### LlamaConfig

| Field | Default | Notes |
| --- | --- | --- |
| `model_path` | required | GGUF model path |
| `n_ctx` | 4096 | Context window (must be ≥ 1) |
| `n_batch` | 2048 | Logical batch (must be ≥ 1) |
| `n_ubatch` | `n_batch` | Physical micro-batch |
| `n_seq_max` | 1 | Max parallel sequences (1 = single sequence) |
| `n_threads` | `os.cpu_count()` | Threads for generation |
| `n_threads_batch` | `n_threads` | Threads for prompt/batch |
| `n_gpu_layers` | -1 | GPU layers (-1 = all) |
| `main_gpu` | 0 | Primary GPU index |
| `split_mode` | 0 | `llama_split_mode` enum |
| `use_mmap` | True | Memory-map model |
| `use_mlock` | False | mlock model into RAM |
| `offload_kqv` | True | Offload K/Q/V to GPU |
| `flash_attn` | 1 | Flash-attention mode (required for quantized V cache) |
| `ctx_type` | `LLAMA_CONTEXT_TYPE_DEFAULT` (0) | Context graph variant. `LLAMA_CONTEXT_TYPE_MTP` (1) selects the MTP graph as the **user-facing** context (used by `tests/test_mtp.py`); for speculative decoding, leave at default — the MTP graph is constructed internally as the draft context. See "MTP context type" below |
| `cache_type_k` | `GGML_TYPE_F16` (1) | ggml_type for K cache — see "Quantized KV cache" below |
| `cache_type_v` | `GGML_TYPE_F16` (1) | ggml_type for V cache — see "Quantized KV cache" below |
| `embeddings` | False | Enable embeddings (required for `embed()` and `create_embedding()`) |
| `add_bos` | None | Add BOS during tokenization (None = auto-detect from model) |
| `parse_special` | False | Parse special tokens |
| `chat_format` | None | Chat template name |
| `verbose` | True | Control logging |
| `seed` | -1 | RNG seed (-1 = random) |

Raises `ValidationError` if:
- `n_ctx < 1`, `n_batch < 1`, `n_seq_max < 1`, or `n_gpu_layers < -1`
- `ctx_type` is not `LLAMA_CONTEXT_TYPE_DEFAULT` (0) or `LLAMA_CONTEXT_TYPE_MTP` (1)
- `cache_type_k` / `cache_type_v` is not one of the supported ggml_types listed below
- `cache_type_v` is quantized (not F32/F16/BF16) and `flash_attn == 0` — quantized V without flash attention produces NaN/garbage output in llama.cpp

#### Quantized KV cache

Constants exported from `llama_cpp` (mirror `ggml.h`'s `enum ggml_type`):

| Constant | Value | Notes |
| --- | --- | --- |
| `GGML_TYPE_F32` | 0 | Full precision |
| `GGML_TYPE_F16` | 1 | Default |
| `GGML_TYPE_BF16` | 30 | Recommended for Qwen 3.5 per upstream |
| `GGML_TYPE_Q4_0` | 2 | |
| `GGML_TYPE_Q4_1` | 3 | |
| `GGML_TYPE_Q5_0` | 6 | |
| `GGML_TYPE_Q5_1` | 7 | |
| `GGML_TYPE_Q8_0` | 8 | Good size/quality balance |
| `GGML_TYPE_IQ4_NL` | 20 | |

```python
from llama_cpp import Llama, LlamaConfig, GGML_TYPE_Q8_0

llm = Llama(
    "models/Qwen3.5-4B-Q4_K_M.gguf",
    config=LlamaConfig(
        model_path="models/Qwen3.5-4B-Q4_K_M.gguf",
        n_ctx=8192,
        cache_type_k=GGML_TYPE_Q8_0,
        cache_type_v=GGML_TYPE_Q8_0,
        flash_attn=1,  # required for quantized V
    ),
)
```

k-quants (Q4_K, Q5_K, Q6_K, etc.) are NOT supported by llama.cpp for KV cache and are rejected by validation.

#### MTP context type

Constants exported from `llama_cpp` (mirror `llama.h`'s `enum llama_context_type`):

| Constant | Value | Notes |
| --- | --- | --- |
| `LLAMA_CONTEXT_TYPE_DEFAULT` | 0 | Default graph (current behavior) |
| `LLAMA_CONTEXT_TYPE_MTP` | 1 | Multi-Token Prediction graph variant |

MTP requires a checkpoint that ships MTP layers — currently Qwen3.5 / Qwen3.5-MoE / Qwen3.6-MoE MTP variants (metadata `*.nextn_predict_layers > 0`, plus `blk.*.nextn.*` tensors). Loading a non-MTP model with `ctx_type=LLAMA_CONTEXT_TYPE_MTP` raises `ModelLoadError` (`"context type MTP requested but model doesn't contain MTP layers"`). The generation API is otherwise unchanged — MTP is a graph-construction-time decision, not a runtime sampler.

**For speculative decoding, leave `ctx_type` at the default.** The speculative path constructs the MTP graph internally as the *draft* context; setting `ctx_type=LLAMA_CONTEXT_TYPE_MTP` on the user-facing context and passing `speculative=True` is a precondition error (`_validate_speculative` raises `ValidationError`). The user-facing `LLAMA_CONTEXT_TYPE_MTP` setting exists for direct MTP-graph generation (no draft-verify loop).

```python
from llama_cpp import Llama, LlamaConfig, LLAMA_CONTEXT_TYPE_MTP

llm = Llama(
    "models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf",
    config=LlamaConfig(
        model_path="models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf",
        n_ctx=4096,
        ctx_type=LLAMA_CONTEXT_TYPE_MTP,
    ),
)
```

### Draft-MTP speculative decoding

`generate()`, `generate_stream()`, and `create_chat_completion()` accept a
`speculative=True` flag. When set, generation runs through a draft-verify
loop that uses the model's own MTP graph to draft `n_draft_max` tokens
per round and validates them with the standard sampler chain on the
user-facing context.

The architecture is **dual-context**: the user-facing context (`ctx_tgt`,
`LLAMA_CONTEXT_TYPE_DEFAULT`) is the verifier, and an internal draft
context (`ctx_dft`, `LLAMA_CONTEXT_TYPE_MTP`) is constructed against the
same `llama_model` on the first speculative call and reused thereafter.
Callers do **not** set `ctx_type=LLAMA_CONTEXT_TYPE_MTP` on the
user-facing context — the MTP graph is used internally as the draft
context only.

Kwargs on the generation entry points:

- `speculative: bool = False` — opt-in flag. When `True`, the generate
  loop runs the draft-MTP draft-verify path; when `False`, the per-token
  path runs unchanged.
- `n_draft_max: int | None = None` — max draft tokens per verify round.
  Range `[1, 8]`. When `None`, falls back to `SamplingParams.n_draft_max`
  (default `2`).

```python
from llama_cpp import Llama, LlamaConfig, SamplingParams

llm = Llama(config=LlamaConfig(
    model_path="models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf",
    # ctx_type=LLAMA_CONTEXT_TYPE_DEFAULT is the default — do NOT set MTP here.
))

text = llm.generate(
    "Explain MoE briefly.",
    max_tokens=256,
    sampling=SamplingParams(temperature=0.0),
    speculative=True,
    n_draft_max=2,
)
```

**Preconditions** (validated by `_validate_speculative`, raised as
`ValidationError`):

- User-facing `ctx_type` must be `LLAMA_CONTEXT_TYPE_DEFAULT`. The MTP
  graph is the draft context, not the user-facing ctx.
- The model must expose an MTP graph variant
  (`Context.supports_speculative_mtp()` — gated on the GGUF metadata key
  `<arch>.nextn_predict_layers > 0`, e.g. Qwen3.6-MoE `*-MTP.gguf`
  checkpoints). The metadata is authoritative: llama.cpp will *allocate* a
  degenerate `LLAMA_CONTEXT_TYPE_MTP` context for any `qwen35`-arch model
  even when it ships zero MTP layers, so the probe must not infer support
  from allocation success — a plain Qwen3.5 checkpoint correctly reports
  `False`.
- `LlamaConfig.embeddings` must be `False`.
- `n_draft_max` must be in `[1, 8]` (validated on `SamplingParams` **and** on
  per-call `n_draft_max=` overrides).
- `speculative=True` is incompatible with `logprobs=...` on `generate()`.

**Hybrid-attention MTP checkpoints** (Qwen3.6-MoE) report
`memory_can_shift()=False` on their user-facing context, but speculative
**still works** for them: the loop trims rejected drafts via the draft
context's `n_rs_seq` recurrent-state rollback, which is a different
mechanism than plain `kv_cache_seq_rm`. `_validate_speculative`
deliberately does not check `memory_can_shift()`.

**Reproducibility note:** the speculative path advances the dist sampler's
RNG once per *position* in each batch rather than once per *step*, so
seeded *non-greedy* runs give matching distributions but differ in the
realized sample sequence vs. the per-token baseline. Greedy
(`temperature=0.0`) is bit-exact.

**Session continuation & mode switching** (`reset_kv_cache=False`,
`cache_prompt=True`): mixing speculative and non-speculative turns on the same
KV is handled automatically. A `speculative=True` turn leaves the user KV one
position behind the prompt-cache mirror (the final corrected token is emitted
but re-decoded on the next speculative turn — this is intentional). When the
*next* turn is non-speculative, the wrapper decodes that one undecoded token in
place so the continuation is correct and the prefix-reuse speedup is preserved.
A non-speculative → speculative switch forces a KV reset (the draft context's
recurrent state can only rebuild from scratch). Same-mode continuations are
untouched. You do **not** need to pass `reset_kv_cache=True` manually when
switching modes — but doing so is always safe.

**Benchmarks** (RTX 4090, see `examples/bench_speculative.py`):

| Model | Variant | Speedup | Unsloth band |
|---|---|---|---|
| Qwen3.6-27B-Q4_K_S | dense | 1.53× | 1.4–2.2× |
| Qwen3.6-35B-A3B-UD-IQ4_XS | MoE (A3B) | 1.31× | 1.15–1.2× |

### SamplingParams

| Field | Default |
| --- | --- |
| `temperature` | 0.8 |
| `top_k` | 40 |
| `top_p` | 0.95 |
| `min_p` | 0.0 |
| `min_keep` | 1 |
| `repeat_penalty` | 1.1 |
| `repeat_last_n` | 64 |
| `presence_penalty` | 0.0 |
| `frequency_penalty` | 0.0 |
| `seed` | None |
| `dry_multiplier` | 0.0 | DRY anti-repetition multiplier (0 = disabled) |
| `dry_base` | 1.75 | DRY base for penalty scaling |
| `dry_allowed_length` | 2 | DRY minimum repeat length |
| `dry_penalty_last_n` | -1 | DRY lookback window (-1 = context size) |
| `dry_seq_breakers` | `["\n",":",'"',"*"]` | DRY sequence breakers |
| `xtc_probability` | 0.0 | XTC removal probability (0 = disabled) |
| `xtc_threshold` | 0.1 | XTC minimum probability threshold |
| `temp_delta` | 0.0 | Dynamic temperature range delta |
| `temp_exponent` | 1.0 | Dynamic temperature exponent |
| `top_n_sigma` | -1.0 | Top-n-sigma cutoff (negative = disabled) |
| `typical_p` | 1.0 | Locally-typical sampling cutoff (1.0 = disabled, range `(0, 1]`) |
| `logit_bias` | None | `dict[int, float]` of per-token additive bias (applied first; `-inf` bans a token). Out-of-range token ids raise `IndexError` |
| `adaptive_p_target` | -1.0 | Adaptive-p target (≥ 0 enables; replaces `dist`). Range `[0, 1]` |
| `adaptive_p_decay` | 0.85 | Adaptive-p decay. Range `[0, 0.99]` |
| `n_draft_max` | 2 | Max draft tokens per verify round when `speculative=True`. Range `[1, 8]` |

Sampler chain ordering: logit_bias → DRY → penalties → top_n_sigma → top_k → top_p → min_p → typical_p → XTC → temp → dist (or adaptive_p when enabled)

## llama_cpp.LlamaGrammar

Grammar for constrained text generation.

```python
from llama_cpp import LlamaGrammar

# From GBNF string
grammar = LlamaGrammar.from_string('root ::= "yes" | "no"')

# From JSON schema
grammar = LlamaGrammar.from_json_schema({"type": "object", "properties": {"name": {"type": "string"}}})

response = llm.create_chat_completion(messages=[...], grammar=grammar)
```

### Lazy (trigger-activated) grammar

Bind `llama_sampler_init_grammar_lazy_patterns` (llama.cpp PR #9639). The grammar stays inactive until the model emits text matching one of the trigger patterns (regex anchored at the start of generated output) or one of the trigger token ids — useful for tool-calling and mixed free-form / structured output.

```python
grammar = LlamaGrammar.lazy(
    'root ::= "{" "}"',
    trigger_patterns=[r"<tool_call>"],
    trigger_tokens=[12345],          # optional
)
assert grammar.is_lazy is True
```

`LlamaGrammar.lazy(...)` raises `ValidationError` if no triggers are supplied (a lazy grammar with no triggers would never activate). Eager grammars (the legacy constructor / `from_string` / `from_json_schema`) report `is_lazy is False` and are unaffected.

## llama_cpp.unified.UnifiedLLM

Unified interface for a curated set of LLM families: **Qwen 3.5**, **Qwen 3.6**, **Gemma 4**, and **IBM Granite 4.1**. Other architectures raise `UnsupportedModelError` at construction — use the lower-level `Llama` class for them.

**Constructor**

```python
UnifiedLLM(
    model_path: str,
    n_ctx: int = 8192,
    n_batch: int = 2048,
    n_ubatch: int = 512,
    n_gpu_layers: int = -1,
    verbose: bool = False,
    family: str | ModelFamily | None = None,
    cache_type_k: int = 1,   # GGML_TYPE_F16
    cache_type_v: int = 1,   # GGML_TYPE_F16
)
```

- `model_path`: Path to GGUF model file.
- `n_ctx`: Context size (clamped to model's max).
- `n_batch`: Batch size for prompt processing.
- `n_ubatch`: Micro-batch size.
- `n_gpu_layers`: Layers to offload to GPU (-1 = all).
- `verbose`: Enable verbose logging.
- `family`: Explicit model family override (auto-detects if None).
- `cache_type_k` / `cache_type_v`: ggml_type for K/V cache. Defaults to F16. Pass e.g. `GGML_TYPE_Q8_0` or `GGML_TYPE_BF16` from `llama_cpp` to quantize. Flash attention is enabled by default (`flash_attn=1`), which is required for quantized V. See [Quantized KV cache](#quantized-kv-cache) above for the full constant list and validation rules.

**Quantized KV cache example**

```python
from llama_cpp import GGML_TYPE_Q8_0
from llama_cpp.unified import UnifiedLLM

# Halves KV cache VRAM vs F16 with minimal quality loss
llm = UnifiedLLM(
    "models/Qwen3.5-4B-Q4_K_M.gguf",
    n_ctx=8192,
    cache_type_k=GGML_TYPE_Q8_0,
    cache_type_v=GGML_TYPE_Q8_0,
)
```

**Context Manager**

```python
with UnifiedLLM("models/Qwen3.5-4B-Q4_K_M.gguf") as llm:
    response = llm.generate("Hello")
# Resources automatically cleaned up
```

**Properties**

- `family` → `ModelFamily` – Detected model family enum (one of `QWEN3_5`, `QWEN3_6`, `GEMMA4`, `GRANITE`)
- `supports_thinking` → `bool` – Whether the resolved preset has thinking enabled by default

**Methods**

- `generate(prompt, system_prompt=None, max_tokens=None, thinking=False, stop=None)` → `str` – Single-turn generation
- `generate_with_thinking(prompt, system_prompt=None, max_tokens=None, stop=None)` → `tuple[str, str]` – Returns (thinking, answer)
- `chat(messages, *, max_tokens=None, thinking=False, stop=None, sanitize_history=True, reset_kv_cache=True, cache_prompt=True)` → `str` – Multi-turn entry point. Auto-sanitizes prior assistant turns by default for thinking-capable families.
- `sanitize_history(messages)` → `list[dict]` – Strip thinking blocks from historical assistant messages (called automatically by `chat`).
- `strip_thinking(text)` → `str` – Remove thinking tags from a single response.
- `n_tokens(text)` → `int` – Count tokens
- `n_ctx()` → `int` – Get context size
- `kv_cache_clear()` → None – Clear KV cache
- `close()` → None – Release resources

**Example**

```python
from llama_cpp.unified import UnifiedLLM

# Auto-detect from filename (raises UnsupportedModelError for unsupported families)
llm = UnifiedLLM("models/Qwen3.5-4B-Q4_K_M.gguf")
print(llm.family.name)         # QWEN3_5
print(llm.supports_thinking)   # False (4B is in the small-variant set)

# Multi-turn chat with automatic history hygiene
messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "<think>greeting</think>Hi!"},
    {"role": "user", "content": "Who are you?"},
]
reply = llm.chat(messages, max_tokens=64)        # thinking blocks stripped
reply = llm.chat(messages, thinking=True)        # /think suffix added on Qwen
reply = llm.chat(messages, sanitize_history=False)  # opt out

# Explicit family override (force the coding preset on a 9B small model)
llm = UnifiedLLM("models/Qwen3.5-9B-Q4_K_M.gguf", family="qwen3.5-coding")
```

### ModelFamily

Enum of supported model families:

- `QWEN3_5`, `QWEN3_6`, `GEMMA4`, `GRANITE`

### UnsupportedModelError

Raised by `UnifiedLLM(...)` and `detect_model_family(path)` when the model file does not match any of the four supported families. Subclass of `ValueError`.

### ModelConfig

Model-specific configuration (auto-selected based on family):

| Field | Description |
| --- | --- |
| `family` | ModelFamily enum |
| `chat_format` | llama.cpp chat format name |
| `temperature` | Default sampling temperature |
| `top_p` | Default nucleus sampling |
| `top_k` | Default top-k sampling |
| `min_p` | Default min-p threshold |
| `max_ctx` | Maximum context length |
| `supports_thinking` | Thinking mode support |
| `stop_sequences` | Default stop sequences |

**Runtime Temperature Override**

The `model_config` is mutable after construction, allowing per-task temperature tuning:

```python
with UnifiedLLM("models/Qwen3.5-4B-Q4_K_M.gguf") as llm:
    # Lower temperature for translation (more faithful, less creative)
    llm.model_config.temperature = 0.3
    response = llm.generate("Translate: ...", system_prompt=TRANSLATION_PROMPT)
```

### detect_model_family

```python
from llama_cpp.unified import detect_model_family

config = detect_model_family("models/Qwen3-30B-A3B-Instruct-2507-Q4_K_S.gguf")
print(config.family)  # ModelFamily.QWEN3
print(config.supports_thinking)  # False (Instruct-2507 variant)
```

Auto-detects model family from file path. Raises `ValueError` if unknown.

**Supported model families and detection patterns:**

| Family | Detection Pattern | Key Features |
| --- | --- | --- |
| `GEMMA` | `gemma` in path (not `gemma-4`) | Google Gemma 2/3, 128K context |
| `GEMMA4` | `gemma-4` / `gemma4` in path | Google Gemma 4. E2B/E4B → 128K; `26b`/`31b`/`a4b` markers → 256K. Thinking via `<\|think\|>` in system prompt |
| `GLM4` | `glm-4` in path | Zhipu GLM-4; `glm-4.7` variant adds thinking mode (202K context) |
| `GRANITE` | `granite` in path or arch `granite*` (incl. `granitehybrid`, `granitemoe`) | IBM Granite 3.x/4.x; thinking mode on; 131K ctx; `temp=0.7`, `top_p=0.9`, `top_k=40`; stops on `<\|end_of_text\|>` / `<\|endoftext\|>` |
| `MINICPM` | `minicpm` in path | MiniCPM, ChatML format |
| `MISTRAL` | `ministral` in path | Mistral (reasoning and instruct variants) |
| `PHI` | `phi-4` in path | Microsoft Phi-4, custom `<\|im_sep\|>` template |
| `QWEN3` | `qwen3` in path | Alibaba Qwen3 with `/think`/`/no_think` toggle |
| `QWEN3_5` | `qwen3.5` in path | Qwen3.5 hybrid attention, 262K context, thinking default-on (no `/think` suffix) |
| `GPT_OSS` | `gpt-oss` in path | GPT-OSS with dual-channel (analysis/final) output |

## JSON Mode

```python
response = llm.create_chat_completion(
    messages=[{"role": "user", "content": "Return JSON"}],
    response_format={"type": "json_object"}
)

# With schema
response = llm.create_chat_completion(
    messages=[...],
    response_format={
        "type": "json_object",
        "schema": {"type": "object", "properties": {"name": {"type": "string"}}}
    }
)
```

## Function Calling

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
    }
}]

response = llm.create_chat_completion(
    messages=[{"role": "user", "content": "Weather in Tokyo?"}],
    tools=tools,
    tool_choice="auto"
)
```

## LoRA Adapters

```python
adapter = llm.load_lora("adapter.gguf", scale=1.0)
response = llm.generate("Hello", max_tokens=32)
llm.remove_lora(adapter)  # or llm.clear_lora()
```

## Resource Management

Both `Llama` and `UnifiedLLM` support proper resource cleanup:

**Context Manager (Recommended)**

```python
# Resources automatically released on exit
with Llama("model.gguf") as llm:
    text = llm.generate("Hello", max_tokens=32)

with UnifiedLLM("model.gguf") as llm:
    text = llm.generate("Hello")
```

**Explicit Close**

```python
llm = Llama("model.gguf")
try:
    text = llm.generate("Hello", max_tokens=32)
finally:
    llm.close()  # Always call close() when done
```

**Notes:**
- `close()` is safe to call multiple times (idempotent)
- After `close()`, the instance cannot be used for inference (`LlamaError` is raised)
- Native resources (Context, Model) are freed in the correct order to prevent segfaults
- For `UnifiedLLM`, the backend reference is cleared before closing the underlying `Llama` instance
- C++ destructors set pointers to `nullptr` after free to prevent double-free
- Neither `Llama` nor `UnifiedLLM` uses `__del__` (avoids GIL issues during shutdown); cleanup via atexit handler + RAII

**Memory Safety Verification:**

Run `tests/test_double_free_scenarios.py` to exercise all cleanup paths (20 scenarios covering both `Llama` and `UnifiedLLM`). For allocator-level corruption detection, run under glibc's heap checker:

```bash
MALLOC_CHECK_=3 uv run pytest tests/test_double_free_scenarios.py -v
```

## llama_cpp.LlamaPool

Pool of Llama instances for true parallel inference. Supports async context manager.

**Constructor**

```python
LlamaPool(model_path: str, pool_size: int = 4, config: LlamaConfig | None = None, warmup: bool = False)
```

- `model_path`: Path to GGUF model file.
- `pool_size`: Number of parallel worker instances.
- `config`: Optional configuration shared by all instances.
- `warmup`: Run dummy inference on each instance to pre-load GPU caches.

**Methods**

- `generate(prompt, max_tokens=128, sampling=None, stop=None, timeout=None)` → `str` – Generate using next available instance
- `generate_batch(prompts, max_tokens=128, sampling=None, stop=None, timeout=None)` → `list[str]` – Generate for multiple prompts in parallel
- `create_chat_completion(messages, max_tokens=128, temperature=None, timeout=None)` → `dict` – Chat completion using next available instance
- `create_chat_completion_batch(message_lists, max_tokens=128, temperature=None, timeout=None)` → `list[dict]` – Batch chat completions in parallel
- `close()` → None – Close all instances immediately (warns if in-flight requests exist)
- `close_graceful(timeout=30.0)` → None – Wait for in-flight requests, then close

**Context Manager**

```python
async with LlamaPool("model.gguf", pool_size=4) as pool:
    results = await pool.generate_batch(["Q1", "Q2"], max_tokens=64)
# close_graceful() called automatically on exit
```

## Thread Safety

- **Sync methods** (`generate()`, `create_chat_completion()`, etc.) are NOT thread-safe. Do not call them concurrently from multiple threads on the same instance.
- **Async methods** (`generate_async()`, etc.) use an internal lock and serialize concurrent calls.
- For true parallelism, use multiple `Llama` or `UnifiedLLM` instances.
- The `verbose=False` setting affects logging globally, not per-instance.
- The GIL is released during heavy C++ operations (decode, generate, tokenize) to allow other Python threads to run.
- **`generate_stream()`** releases the GIL during C++ decode/sample, re-acquiring only for the Python callback. Early termination is handled via `threading.Event`.
- **`close()` is thread-safe** at the C++ level — `Model::close()` and `Context::close()` hold a mutex to prevent races between GC/`__del__` and explicit calls.
