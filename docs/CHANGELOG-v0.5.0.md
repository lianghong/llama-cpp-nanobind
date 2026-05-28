# Changelog — v0.5.0 (2026-05-28)

**Focus:** draft-MTP speculative decoding (the headline feature), plus a batch of sampling, grammar, and state-management additions landed since v0.4.0.

No breaking API changes. Version bumped from `0.4.0` → `0.5.0` in `pyproject.toml`, `CMakeLists.txt`, and `src/llama_cpp/_about.py`.

---

## Headline: draft-MTP speculative decoding

`Llama.generate(...)`, `Llama.generate_stream(...)`, and `Llama.create_chat_completion(...)` now accept a `speculative=True` flag. When set, generation runs through a draft-verify loop that uses the model's own MTP graph to draft `n_draft_max` tokens per round and validates them with the standard sampler chain on the user-facing context.

### Architecture

- **Dual-context.** The user-facing context (`ctx_tgt`, `LLAMA_CONTEXT_TYPE_DEFAULT`) is the verifier. An internal draft context (`ctx_dft`, `LLAMA_CONTEXT_TYPE_MTP`) is constructed against the same `llama_model` on first speculative call and reused thereafter (`Context::ensure_mtp_draft_context`).
- **Loop.** `generate_tokens_speculative_mtp` (in `src/bindings/llama_cpp.cpp`) initializes a `common_speculative *` configured for `COMMON_SPECULATIVE_TYPE_DRAFT_MTP`, drafts up to `n_draft_max` tokens via `common_speculative_draft`, decodes the `[id_last, drafted...]` batch on `ctx_tgt`, mirrors that batch onto `ctx_dft` via `common_speculative_process`, runs the standard sampler chain (and grammar, if any) at each position, accepts matches, and corrects on the first mismatch. Rejected drafts are trimmed via `kv_cache_seq_rm` on both contexts.
- **MTP KV trimming.** Hybrid-attention MTP checkpoints (Qwen3.6-MoE) report `memory_can_shift()=False` on `ctx_tgt`, but the speculative path trims rejected drafts via `ctx_dft`'s `n_rs_seq` recurrent-state rollback — a different mechanism. `_validate_speculative` deliberately does **not** check `memory_can_shift()` because doing so would block the supported MTP models.

### API surface

```python
# SamplingParams gets one new field:
sp = SamplingParams(n_draft_max=4)   # range [1, 8], default 2

# Three entry points gain the kwarg:
out = llm.generate(prompt, sampling_params=sp, speculative=True)
for chunk in llm.generate_stream(prompt, sampling_params=sp, speculative=True):
    ...
resp = llm.create_chat_completion(messages, sampling_params=sp, speculative=True)
```

### Preconditions (validated by `Llama._validate_speculative`)

- User-facing `ctx_type=LLAMA_CONTEXT_TYPE_DEFAULT` (the MTP graph belongs to the internal draft context).
- Model exposes an MTP graph (`Context::supports_speculative_mtp()` — checks for `nextn_predict_layers > 0` metadata).
- `LlamaConfig.embeddings=False`.

### Benchmarks (Qwen3.6 family, RTX 4090)

| Model | Variant | Speedup | Unsloth band |
|---|---|---|---|
| Qwen3.6-27B-Q4_K_S | dense | **1.53×** | 1.4–2.2× |
| Qwen3.6-35B-A3B-UD-IQ4_XS | MoE (A3B) | **1.31×** | 1.15–1.2× |

Both inside or above unsloth's published bands; greedy outputs are bit-exact vs. the per-token path. With non-zero temperature, distributions match but trajectories differ because speculative advances the dist sampler's RNG once per *position* in each batch rather than once per *step*.

### C++ bindings

- **Multi-token batch buffer + `decode_multi`** (`Context::decode_multi`) — used by the speculative path and exposed for direct multi-token decode.
- **`Context::supports_speculative_mtp() -> bool`** — predicate exposed for callers who want to gate speculative use without raising.
- **`Context::ensure_mtp_draft_context()`** — constructs (and caches) `ctx_dft` against the same model with `LLAMA_CONTEXT_TYPE_MTP`. Caps `n_ubatch` at 64 to avoid a backend-top-k second-reserve OOM at `n_draft_max=8`.
- **`generate_tokens_speculative_mtp`** — the draft-verify loop. Empty-priming early-return runs before `common_speculative` construction (review fix).
- **Linkage.** `CMakeLists.txt` now also links `libllama-common` for the `common_speculative_*` symbols.

### Python wrappers

- **`SamplingParams.n_draft_max`** (default 2, range [1, 8]).
- **`Llama._validate_speculative(speculative: bool) -> None`** — precondition helper. Docstring records the deliberate omission of the `memory_can_shift()` check and points at this changelog and the dual-context architecture.
- **Mirror invariant under speculative.** When `cache_prompt=True`, mirror trim/extend logic now mirrors the verified token sequence (not the drafted sequence) to keep `len(_cached_prompt_tokens) == kv_pos_max + 1` after rejection cycles.

### Tests

- **`tests/test_speculative_validation.py`** (11 cases): default `n_draft_max`, range bounds, `ctx_type` precondition, `embeddings=True` rejection, `_validate_speculative(False)` no-op, `Context::supports_speculative_mtp` on the default ctx, `decode_multi` smoke, and a presence check for the C++ entry point.
- **`tests/test_speculative_mtp.py`** (7 cases, requires MTP model): end-to-end greedy parity, stream parity, stop-sequence behavior, `cache_prompt` interaction, grammar interaction, `n_draft_max` bounds (including the n=8 stress path), and `create_chat_completion` integration.
- **`examples/bench_speculative.py`** — micro-benchmark harness gated on the `LLAMA_MTP_TEST_MODEL` env var.

---

## Sampling additions

- **Adaptive-p terminal sampler** (`f83c858`). When `SamplingParams(adaptive_p_target=...)` is `≥ 0`, the chain replaces `dist` with `llama_sampler_init_adaptive_p`. Tested in `tests/test_adaptive_p.py`.
- **MXFP4 / NVFP4 / Q1_0 ggml type wiring** (`f83c858`) — surface new ggml block types for KV-cache experimentation.
- **`logit_bias` on `SamplingParams`** (`08b8d99`). Per-token bias dict, applied as the first node of the sampler chain (OpenAI-API parity). Tested in `tests/test_logit_bias.py`.
- **`typical_p` (locally-typical sampling)** (`0f06780`). Inserted in the canonical sampler order between `min_p` and `XTC`. Tested in `tests/test_typical_p.py`.

## Grammar

- **Lazy grammar (trigger-activated GBNF)** (`d89e418`). `LlamaGrammar.lazy(grammar_str, *, trigger_patterns=None, trigger_tokens=None)` produces a sampler that becomes active only when the trigger fires. Empty triggers are rejected at construction (would never activate). Tested in `tests/test_lazy_grammar.py`.

## State management

- **On-device per-seq state save / load** (`a5c559a`). `Context::save_seq_state_on_device(seq_id) -> nb::bytes` and `Context::load_seq_state_on_device(handle, dest_seq_id)` wrap `llama_state_seq_{get,set}_data_ext` with `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE`. Handles are short-lived in-session references (invalidated by any KV-clearing op, including re-saving the same seq); for durable snapshots use `get_state()`. Tested in `tests/test_state_on_device.py`.

## Performance

- **Prompt-prefix KV reuse (`cache_prompt=True`, default)** (`3bec031`). When paired with `reset_kv_cache=False`, computes the LCP between cached and new prompts, calls `kv_cache_seq_rm(0, n_match, -1)`, and decodes only the divergent suffix. Falls back to clear+re-prime when `memory_can_shift()=False` (correctness preserved, speedup lost). Mirror invariant: `len(_cached_prompt_tokens) == kv_pos_max + 1`. Tested in `tests/test_prefix_reuse.py`.
- **Quantized KV cache ergonomics + validation** (`4a9b732`). `LlamaConfig` now rejects K-quants (Q4_K, …) and any non-{F32, F16, BF16} V-cache type unless `flash_attn=1`. Allowed K/V types are documented in CLAUDE.md.

---

## Documentation

- **CLAUDE.md** rewritten and compacted (646 → 217 lines) (`da2e4f5`); subsequently extended with MTP, speculative, and on-device-state sections.
- **README** corrected to reflect MTP graph-variant scope (`b4733d8`).
- **API docs** updated for the speculative path, `n_draft_max`, lazy grammar, on-device state, and adaptive-p.
- **Design docs** added under `docs/superpowers/specs/` and `docs/superpowers/plans/` for draft-MTP and the MTP doc-correction sweep.

---

## Tests

All new suites listed under their feature sections above. Full suite (`uv run pytest -q`) is green; speculative end-to-end suite skips cleanly when `LLAMA_MTP_TEST_MODEL` is unset.

---

## Lint / format

Clean on all tooling:

- `clang-format` / `clang-tidy` (0 user warnings)
- `ruff check` / `ruff format`
- `mypy`

---

## Upgrade

```bash
./scripts/build_wheel.sh --clean --install
uv run pytest -q
```

Produces `dist/llama_cpp_nanobind-0.5.0-cp314-cp314-linux_x86_64.whl`.

To exercise the headline feature locally:

```bash
LLAMA_MTP_TEST_MODEL=/path/to/Qwen3.6-MoE-MTP.gguf \
  uv run pytest tests/test_speculative_mtp.py -v
LLAMA_MTP_TEST_MODEL=/path/to/Qwen3.6-MoE-MTP.gguf \
  uv run python examples/bench_speculative.py
```
