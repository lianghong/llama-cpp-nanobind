# Changelog - 2026-04-18

**Date:** 2026-04-18
**Focus:** Gemma 4 support, removal of Aya / TranslateGemma, third code-review pass

---

## Added: Gemma 4 Model Family

New `ModelFamily.GEMMA4` covers Google's Gemma 4 lineup per the
[Unsloth spec](https://unsloth.ai/docs/models/gemma-4).

### Variants

| Config key       | Variants covered        | Context | Architecture    |
| ---------------- | ----------------------- | ------- | --------------- |
| `gemma-4`        | E2B, E4B                | 131,072 | Dense + PLE     |
| `gemma-4-large`  | 26B-A4B (MoE), 31B      | 262,144 | MoE / Dense     |

Variant selection uses size markers `26b`, `31b`, `a4b` in filename or
GGUF `general.name` metadata — absent, the small (128K) config is chosen.

### Sampling defaults (per Unsloth)

```
temperature = 1.0
top_p       = 0.95
top_k       = 64
min_p       = 0.0
repeat_penalty = 1.0   # keep disabled unless looping occurs
```

### Thinking mode

Gemma 4 enables thinking by prepending `<|think|>` to the **system prompt**
(not a per-user suffix like Qwen3's `/think`). The output uses the
`<|channel>thought[content]<channel|>` block, which is now parsed by
`_parse_thinking` and stripped by `_clean_response` in
`ChatTemplateBackend`.

Call via:

```python
from llama_cpp.unified import UnifiedLLM

llm = UnifiedLLM("models/gemma-4-e4b-it-Q8_0.gguf")
thinking, answer = llm.generate_with_thinking(
    "Explain quantum entanglement",
    system_prompt="You are a physics tutor.",
)
```

### Stop sequences

`<turn|>` and `<end_of_turn>` (the Gemma-4-specific end-of-sentence token
plus the classic Gemma turn delimiter). Both added to
`ChatTemplateBackend._CONTROL_TOKENS` so they are stripped from the
response even if emitted mid-stream.

### Detection

Both filename fallback (`detect_model_family`) and GGUF metadata
(`detect_from_metadata`) now route Gemma 4 before generic Gemma:

```python
detect_model_family("models/gemma-4-e4b-it-Q8_0.gguf")       # → gemma-4 (128K)
detect_model_family("models/gemma-4-26b-a4b-it-Q4_K_XL.gguf") # → gemma-4-large (256K)
detect_model_family("models/gemma-2-9b-it-Q6_K.gguf")        # → gemma (unchanged)
```

---

## Removed: Aya and TranslateGemma

These families are no longer supported by `UnifiedLLM`. Users loading
such models should use the base `Llama` class directly, or register their
own `ModelConfig` if they want auto-detection behavior.

### Changes

- Dropped `ModelFamily.AYA` and `ModelFamily.TRANSLATEGEMMA` enum members
- Removed `MODEL_CONFIGS["aya"]` and `MODEL_CONFIGS["translategemma"]`
- Removed metadata/filename detection branches (Aya never had dedicated
  metadata logic; TranslateGemma's `translate` name-check was removed)
- Dropped corresponding `BACKEND_MAP` entries
- Deleted `examples/aya_demo.py` and `examples/translategemma_example.py`
- Removed `command-r`/`aya` stop-string entries from
  `examples/model_helper_utils.py`
- Updated `tests/test_unified.py` — removed Aya/TranslateGemma detection
  tests; added four Gemma 4 tests (small, large 26B, large 31B, legacy
  Gemma 2 negative)

### Migration

If you were using `UnifiedLLM` with Aya or TranslateGemma:

```python
# Before
from llama_cpp.unified import UnifiedLLM
llm = UnifiedLLM("models/tiny-aya-global-q8_0.gguf")

# After — use the low-level Llama class
from llama_cpp import Llama, LlamaConfig, SamplingParams
llm = Llama(
    "models/tiny-aya-global-q8_0.gguf",
    config=LlamaConfig(
        model_path="models/tiny-aya-global-q8_0.gguf",
        n_ctx=8192,
    ),
    sampling=SamplingParams(temperature=0.3, top_p=0.95, top_k=50),
)
```

---

## Code Review Fixes (Third Pass)

### HIGH Priority

**H1: Missing `_check_closed()` on four KV-cache methods**
`Llama.kv_cache_seq_cp`, `kv_cache_seq_keep`, `kv_cache_seq_add`,
`kv_cache_seq_pos_max` now validate closed state, matching the rest of
the public API surface. Previously these raised `AttributeError` on
closed instances.

**H2: Logits OOB read in `compute_top_logprobs`**
The `partial_sort` comparator now rejects token IDs outside
`[0, n_vocab)`. `LLAMA_TOKEN_NULL` (-1) or corrupted token IDs no longer
cause an out-of-bounds read into the logits array.

**H3: `LlamaPool` queue init race**
`_ensure_queue_initialized()` now uses a `threading.Lock` with
double-checked locking. Safe under free-threaded Python 3.13+.

### MEDIUM Priority

**M1: `UnifiedLLM.__repr__` during failed init**
Added `hasattr(self, "llm")` check so `__repr__` doesn't raise
`AttributeError` if called before `self.llm` is assigned (e.g., in a
debugger during a failed constructor).

**M2: `set_adapters_lora` `size_t → int32_t` truncation**
Added explicit `INT32_MAX` guard in `Context::set_adapters_lora` before
casting. Silent truncation is now an `std::invalid_argument`.

---

## Documentation Updates

- `CLAUDE.md` Model Support section: removed Aya/TranslateGemma,
  added Gemma 4 entry
- `README.md`: updated `UnifiedLLM` family list
- `docs/API.md`: updated `ModelFamily` enum listing and detection table

---

## Test Results

- `tests/test_unified.py`: 37/37 pass (added 4 new Gemma 4 tests)
- `tests/test_close_exception_safety.py`: 4/4 pass
- `tests/test_pool.py`: 13/13 pass

Other test failures observed in the full suite are pre-existing
VRAM-exhaustion flakes documented in `TEST_FAILURES_ANALYSIS.md` and are
unrelated to the changes in this release.
