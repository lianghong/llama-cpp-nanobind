# Changelog — 2026-05-27

**Focus:** track recent llama.cpp upstream additions.
No breaking API changes.

---

## New features

### Adaptive-p sampler (llama.cpp PR #17927)

Terminal sampler that replaces `dist` when enabled. Picks an adaptive
truncation threshold per step instead of using a fixed top-p.

**C++ bindings (`src/bindings/llama_cpp.cpp`)**
- `SamplerChain::Params::adaptive_p_target` — float, default `-1.0` (disabled).
  When `>= 0`, the chain swaps the terminal `dist` sampler for
  `llama_sampler_init_adaptive_p(target, decay, seed)`.
- `SamplerChain::Params::adaptive_p_decay` — float, default `0.85`.
- Both fields exposed via `def_rw`.

**Python (`src/llama_cpp/llama.py`)**
- `SamplingParams.adaptive_p_target: float = -1.0`
- `SamplingParams.adaptive_p_decay: float = 0.85`
- Validation: `target ∈ [0, 1]` when enabled (negative disables);
  `decay ∈ [0, 0.99]`.
- `to_native()` passes both fields through.

Default-off, so existing callers see no change. Pass through the explicit
`sampling=` kwarg on `Llama.generate()` (or `**sampling_overrides` on
`create_chat_completion`).

### Per-sequence on-device state save/load

Wraps `llama_state_seq_{get,set}_data_ext` with the
`LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` flag — tensor data stays in GPU buffers
instead of round-tripping through host memory. Useful for fast in-process
branching (speculative continuations, beam-style exploration) where the
snapshot is consumed during the same generation session.

**C++ bindings**
- `Context::save_seq_state_on_device(seq_id) -> nb::bytes` (opaque handle)
- `Context::load_seq_state_on_device(handle, dest_seq_id) -> size_t`
- Same buffer-protocol pattern as `get_state_data` / `set_state_data`:
  `PyBytes_FromStringAndSize` for allocation, GIL-released llama call,
  `_PyBytes_Resize` on actual size.

**Python wrapper**
- `Llama.save_seq_state_on_device(seq_id=0)`
- `Llama.load_seq_state_on_device(data, dest_seq_id=0)`
- `_check_closed()` guard on both.
- Load invalidates the prompt-cache mirror only when `dest_seq_id == 0`.

**Handle lifetime (important):** the snapshot is invalidated by any
KV-clearing op — `reset()`, `kv_cache_clear()`, `set_state_data()`,
`load_state()` — and by re-saving the same `seq_id`. Loading a stale
handle calls `ggml_abort` (process termination); the C API performs no
validation. Treat the handle as a short-lived in-session reference, not
a persistent snapshot. For durable state use `get_state()`.

### Logit bias (OpenAI-API parity)

Per-token additive bias on the logits, applied before all other samplers
so the biased values flow through DRY / penalties / truncation. Mirrors
the `logit_bias` field in the OpenAI chat/completions API and in
`llama-cpp-python`. Common uses: ban a token (`-inf`), encourage specific
completions, suppress profanity.

**C++ bindings**
- `SamplerChain::Params::logit_bias` is
  `std::vector<std::pair<llama_token, float>>`. Empty = disabled.
- `llama_sampler_init_logit_bias` is added first in the chain.
- Token-id range is validated at chain construction; out-of-range raises
  `std::out_of_range`, surfaced as `IndexError` in Python.

**Python wrapper**
- `SamplingParams.logit_bias: dict[int, float] | None = None`.
- Validation: keys are non-negative ints; values are real numbers
  (rejects `NaN` and non-numeric). `-inf` is accepted (ban semantics).
- `to_native()` converts the dict to `list[tuple[int, float]]`.

### Locally-typical sampling

Bind `llama_sampler_init_typical` (Meister et al.,
[arXiv:2202.00666](https://arxiv.org/abs/2202.00666)). Truncates by
local typicality (entropy distance) instead of cumulative probability;
slots into the chain right after `min_p`.

**C++ bindings**
- `SamplerChain::Params::typical_p`, default `1.0` (disabled — matches
  llama.cpp convention). When `< 1.0` the chain inserts
  `llama_sampler_init_typical(typical_p, min_keep)`.

**Python wrapper**
- `SamplingParams.typical_p: float = 1.0`.
- Validation: `(0.0, 1.0]` (zero would discard everything; > 1.0 is
  meaningless).
- `to_native()` passes the float through.

### Lazy grammar (trigger-activated GBNF)

Bind `llama_sampler_init_grammar_lazy_patterns` (llama.cpp PR #9639).
The grammar stays inactive until the model emits text matching one of
the trigger patterns (regex anchored at the start of generated output)
or one of the trigger token ids — useful for tool-calling and mixed
free-form / structured output (e.g. emit `<tool_call>` then be
constrained to a JSON schema).

**C++ bindings**
- New `GrammarSampler` constructor overload taking
  `(grammar_str, grammar_root, trigger_patterns, trigger_tokens)`
  alongside the existing eager constructor. Nanobind dispatches on
  signature.

**Python wrapper**
- `LlamaGrammar(..., trigger_patterns=..., trigger_tokens=...)` and a
  new classmethod `LlamaGrammar.lazy(grammar_str, root, *,
  trigger_patterns, trigger_tokens)`.
- Validation: `LlamaGrammar.lazy` rejects empty triggers (would never
  activate).
- New read-only `LlamaGrammar.is_lazy` property.
- `_ensure_sampler()` picks the eager or lazy native constructor based
  on whether triggers are present — eager grammars are unaffected.

### Forward-compat ggml type constants

Exported but **not** added to `_VALID_CACHE_TYPES` — these are weight-only
quantization formats; routing them into the KV cache is rejected at
config time.

| Constant | Value |
|---|---|
| `GGML_TYPE_MXFP4` | 39 |
| `GGML_TYPE_NVFP4` | 40 |
| `GGML_TYPE_Q1_0` | 41 |

Importable from `llama_cpp` package root.

## Tests

- `tests/test_adaptive_p.py` — 11 tests. Default-off behavior, validation
  edges (`target > 1`, `decay ∉ [0, 0.99]`, `target = 0` and `target = 1`
  boundaries), constant exposure + KV-whitelist exclusion, end-to-end
  generation, equivalence-with-disabled-path.
- `tests/test_state_on_device.py` — 4 tests. Smoke (`save` returns
  non-empty bytes), round-trip (save → extend KV → load → continuation
  matches), prompt-cache mirror invalidation on seq-0 load, close-guard
  raises `LlamaError`.
- Intentionally **no** "garbage handle raises" test for the on-device
  path — `ggml_abort` terminates the process, and the opaque handle
  format has no public layout to pre-validate.
- `tests/test_lazy_grammar.py` — 9 tests. Eager-grammar `is_lazy`
  remains `False`; lazy construction with patterns / tokens / both;
  validation rejects empty triggers; constructor-kwargs path; native
  lazy-sampler construction; end-to-end "no-trigger-no-constraint"
  generation.
- `tests/test_typical_p.py` — 8 tests. Default-disabled, `to_native`
  pass-through, validation edges (zero / negative / > 1 rejected, 1.0
  allowed as the disabled sentinel), end-to-end equivalence-when-disabled
  and smoke-generation-when-enabled.
- `tests/test_logit_bias.py` — 10 tests. Validation edges (default
  disabled, dict round-trip, empty dict, `-inf` accepted, negative key /
  NaN / non-numeric rejected); end-to-end behavior (`None`/empty matches
  baseline, banning the baseline token via `-inf` changes the output,
  out-of-range token id raises `IndexError`).

## Verification

- 214/214 tests pass
- `ruff check`: clean on modified files
- `clang-format --dry-run -Werror` on `llama_cpp.cpp`: clean

Pytest requires a rebuilt extension — rerun after:

```bash
./scripts/build_wheel.sh --clean --install
uv run pytest -q
```
