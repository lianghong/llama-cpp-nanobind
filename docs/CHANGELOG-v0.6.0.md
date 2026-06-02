# Changelog — v0.6.0 (2026-06-02)

**Focus:** correctness of the KV cache / prompt-cache mirror invariant across multi-token stop sequences and speculative↔non-speculative mode switches. Originated as a code-review pass (5 high, 2 medium); live verification on the 35B MTP model surfaced three deeper bugs that are also fixed here.

Version bumped `0.5.0` → `0.6.0` in `pyproject.toml`, `CMakeLists.txt`, and `src/llama_cpp/_about.py`.

**Behavior change (why minor, not patch):** the on-device state handle format changed — handles now embed a per-instance owner token, so a handle produced by v0.5.0 is not loadable by v0.6.0 (and vice-versa). These handles were already documented as short-lived in-session references, so no durable state is affected.

---

## Code-review findings

### High

- **Multi-token stop sequences left stop-prefix tokens stranded in KV.** For an N-token stop, the first N−1 tokens were decoded into KV across earlier iterations and then erased from the returned output, leaving native KV ahead of `_cached_prompt_tokens`. All four stop handlers (`generate_tokens_with_details`, `generate_tokens_multi_stop`, `generate_tokens_grammar_multi_stop`, `generate_tokens_streaming`) now rewind KV to match the returned output **and** refresh the logits buffer to the last returned token (`Context::rewind_keep_refresh`). The rewind is guarded by `memory_can_shift()`: on hybrid/recurrent models a mid-sequence `seq_rm` corrupts recurrent state rather than no-op'ing, so it is skipped there (and the Python layer reconciles instead — see below).
- **On-device state handles lacked instance identity.** `save_seq_state_on_device` now embeds a per-instance 16-byte owner UUID (envelope is magic + owner + epoch). `load_seq_state_on_device` rejects a handle from a different `Llama` instance with `LlamaError` instead of risking a native crash on a foreign device-resident reference.
- **Logprobs reset path did not invalidate on-device handles.** `generate(..., logprobs=..., reset_kv_cache=True)` now calls `_invalidate_prompt_cache()` after `kv_cache_clear()`, bumping the state epoch so a stale handle is rejected.
- **Streaming mutated the context before acquiring the lock.** `generate_stream` now performs all native context work (speculative precondition probe, `kv_cache_clear`, tokenization, prompt-cache prep) inside the locked region.
- **(Reverted, see “Deeper bugs”)** the original Finding-1 fix (force KV == mirror at the speculative loop exit) was withdrawn after it was shown to corrupt the common spec→spec case.

### Medium

- **`n_draft_max` per-call override bypassed validation.** `generate` / `generate_stream` / `create_chat_completion` now validate the effective `n_draft_max` to `[1, 8]` (`Llama._validate_n_draft_max`), matching `SamplingParams`.
- **`LlamaPool` warmup bypassed Python bookkeeping.** `_warmup_instances` now calls `Llama.kv_cache_clear()` (not `instance.ctx.kv_cache_clear()`) so the prompt-cache mirror and state epoch stay in sync.

---

## Deeper bugs found during live verification

- **The speculative off-by-1 is load-bearing.** At the speculative loop's exit the user-facing KV ends exactly one position *behind* the mirror — the final `corrected_id` is emitted but deliberately left undecoded to become next round's `id_last`; a following *speculative* turn re-decodes it and self-heals. Forcing `KV == mirror` in C++ (decode-forward, or mirror truncation) routes that token through a second `common_speculative_process`, double-applying the draft's recurrent MTP state → garbage. The only safe exit reconciliation in C++ is trimming KV when it is *ahead* (accepted-but-unemitted drafts on EOS/stop).
- **Speculative + multi-token stop → process abort.** A speculative generation that hit a multi-token stop, followed by a speculative continuation, aborted at `speculative.cpp GGML_ASSERT(impl)` because stop tokens were left in the C++ mirror, desyncing `common_speculative_begin`'s `pos_max` bookkeeping. Stop tokens are now erased from *both* `output` and the C++ `mirror` so they stay equal.
- **Mixed-mode (spec↔non-spec) continuation produced drift/garbage.** Now reconciled asymmetrically in Python (`Llama._guard_speculative_mode_switch`, tracked via `_last_gen_speculative`, reset to `None` by `_invalidate_prompt_cache`):
  - **spec → non-spec:** heal the off-by-1 *in place* — `decode_one` the undecoded tail so KV aligns with the mirror, then continue with prefix reuse. Output correct, prefix-reuse speedup preserved.
  - **non-spec → spec:** the draft recurrent state can only rebuild from position 0 (a full prefix decode regardless), so force `reset_kv_cache=True`.
- **Hybrid stop-stranding reconciliation.** `Llama._reconcile_kv_ahead_of_mirror()` (run after every cached commit in all generation paths) clears KV when it ends *ahead* of the mirror — the hybrid-model case where a multi-token stop stranded prefix tokens that `seq_rm` could not safely trim — so the next continuation re-primes cleanly. It is asymmetry-safe: it never fires on the speculative off-by-1 (KV *behind*, not ahead).

---

## C++ bindings (`src/bindings/llama_cpp.cpp`)

- **`Context::rewind_keep_refresh(keep_len, last_token)`** — trims KV seq 0 to `keep_len` and re-decodes `last_token` so the logits buffer reflects the last returned token (not a trimmed-away position). Used by the four stop handlers, guarded by `memory_can_shift()`.
- Speculative loop-exit reconciliation reverted to trim-only (KV-ahead); the off-by-1 is documented in-code as load-bearing.

## Python wrappers (`src/llama_cpp/llama.py`)

- **`Llama._validate_n_draft_max(value) -> int`** — `[1, 8]` bound for per-call overrides.
- **`Llama._guard_speculative_mode_switch(...)`** — asymmetric mode-switch reconciliation (above).
- **`Llama._reconcile_kv_ahead_of_mirror()`** — KV-ahead-of-mirror cleanup (above).
- **On-device handle envelope** extended to magic + owner-UUID + epoch; `_state_owner` set in `__init__`.

---

## Tests

- **`tests/test_prefix_reuse.py`** — `test_multi_token_stop_keeps_mirror_aligned`: differential check that after a multi-token string stop, `len(mirror) == kv_pos + 1` on `memory_can_shift()` models and the continuation byte-matches a fresh re-prime. Verified to fail on the pre-fix tree (`16 == 20`).
- **`tests/test_speculative_mtp.py`** (requires MTP model; tests share one module-scoped 35B instance to bound peak VRAM):
  - `test_speculative_continuation_modes` — all four spec↔non-spec transitions give `cont == fresh`.
  - `test_speculative_to_nonspeculative_preserves_prefix` — spec→non-spec heals in place (mirror extended, not reset) so the prefix-reuse speedup is kept.
  - `test_speculative_stop_then_continue_does_not_crash` — regression for the `GGML_ASSERT(impl)` abort.
  - `test_speculative_max_tokens_one`, `test_speculative_streaming_callback_cancel` — early-termination paths.

Full suite (`uv run pytest -q`) green: **292 passed**; speculative suites skip cleanly when `LLAMA_MTP_TEST_MODEL` is unset.

---

## Lint / format

Clean on all tooling: `clang-format` (0 user warnings), `ruff check` / `ruff format`, `mypy`.

---

## Upgrade

```bash
./scripts/build_wheel.sh --clean --install
uv run pytest -q
```

Produces `dist/llama_cpp_nanobind-0.6.0-cp314-cp314-linux_x86_64.whl`.

**Note:** on-device state handles (`save_seq_state_on_device`) from v0.5.0 are not loadable in v0.6.0 (owner token added). Re-save any in-session snapshots after upgrading; durable state via `get_state()` is unaffected.
