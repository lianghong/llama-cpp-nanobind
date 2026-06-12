# Changelog — 2026-06-03

**Focus:** fix a false-positive MTP capability probe that crashed the process
when speculative decoding engaged on a non-MTP checkpoint.
No breaking API changes.

---

## Bug fix

### `supports_speculative_mtp()` false-positived on plain Qwen3.5 checkpoints → SIGABRT

**Symptom.** `UnifiedLLM` loaded a non-MTP model (`Qwen3.5-9B-Q4_K_M.gguf`,
arch `qwen35`, zero MTP layers) with the default `speculative="auto"`. The
probe reported MTP support, speculative engaged, and the first prompt longer
than the draft context's `n_ubatch` (64) aborted the process:

```
llama-context.cpp:1701: GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all)
  && "non-causal attention requires n_ubatch >= n_tokens") failed
  → common_speculative_impl_draft_mtp::process → llama_decode → ggml_abort
```

**Root cause.** `Context::supports_speculative_mtp()` inferred MTP capability
from whether an `LLAMA_CONTEXT_TYPE_MTP` context could be *allocated*. The
genuine signal — the GGUF metadata key `<arch>.nextn_predict_layers` — was
never consulted. Allocation success is not a capability signal: llama.cpp
builds **before b9180** (2026-05-16, the same PR that introduced the
`common_speculative` draft-MTP API) allocate a degenerate MTP context for any
`qwen35`-arch model regardless of whether it ships next-token-prediction
layers, so against such a build the probe returned `True` on plain Qwen3.5
checkpoints. (b9180+ rejects the allocation — `llama_init_from_model` returns
null when `hparams.n_layer_nextn == 0` — under which the old probe happened to
return `False` for the right reason on the wrong mechanism, and an allocation
probe still builds a throwaway draft context just to answer a capability
query.) The draft-MTP draft context runs non-causal and forbids ubatch
chunking, so once speculative engaged, the first >64-token prompt tripped the
assert. The existing
`tests/test_speculative_validation.py::test_supports_speculative_mtp_default_ctx_returns_false`
encoded the correct expectation (`False`) and was failing against the
shipped 0.6.0 build.

**Fix.** Gate the probe on the authoritative metadata signal.

**C++ bindings (`src/bindings/llama_cpp.cpp`)**
- Added `Context::mtp_predict_layers()` — reads `general.architecture`, then
  the arch-prefixed `<arch>.nextn_predict_layers`, parses strictly (trailing
  junk → treated as absent), and returns `0` when the key is missing/empty.
  Never throws.
- `Context::supports_speculative_mtp()` now returns `False` immediately when
  `mtp_predict_layers() <= 0`, *before* the lazy draft-context allocation.
  Genuine MTP checkpoints (which declare `nextn_predict_layers > 0`) are
  unaffected.
- Both methods are exposed to Python; docstrings updated.

**Behavior.**
- Plain Qwen3.5 (4B/9B): `mtp_predict_layers() == 0`,
  `supports_speculative_mtp() is False`. `UnifiedLLM(..., speculative="auto")`
  resolves to disabled and generates normally — no crash. An explicit
  `speculative=True` raises a recoverable error instead of aborting the
  process: `ValueError` from the `UnifiedLLM` constructor, `ValidationError`
  from `Llama.generate(...)`-level calls.
- Genuine MTP (Qwen3.6-35B-A3B / 27B, `nextn_predict_layers=1`):
  `supports_speculative_mtp() is True` — speculative still works end-to-end.

**Tests (`tests/test_speculative_validation.py`)**
- `test_supports_speculative_mtp_default_ctx_returns_false` (was failing →
  now passes) additionally asserts `mtp_predict_layers() == 0`.
- New `test_supports_speculative_mtp_real_mtp_model_returns_true` — positive
  case on a genuine MTP checkpoint (skips via `LLAMA_MTP_TEST_MODEL` when
  absent), guards against regressing real MTP support.

Verified: 70/70 across `test_speculative_validation.py`, `test_mtp.py`, and
`test_inference.py` on live 4B / 9B / 35B-MoE models.

---

## Known follow-up (not addressed here)

The draft-MTP generation path (`generate_tokens_speculative_mtp`) still aborts
via `GGML_ASSERT` if a prime batch exceeds the draft context's `n_ubatch`
(64) — reachable on a *genuine* MTP model fed a >64-token prompt. The probe
fix prevents speculative from engaging on non-MTP models (the reported crash),
but the prime-batch chunking / clean-error hardening for real MTP models
remains open.
