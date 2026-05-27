# Draft-MTP Speculative Decoding Design (2026-05-27)

**Status:** approved (brainstorming complete; awaiting plan).

**Goal:** Turn `LlamaConfig.ctx_type=LLAMA_CONTEXT_TYPE_MTP` from a no-op scaffold into a real throughput improvement on Qwen3.6-MoE MTP checkpoints by wiring a draft-verify speculative-decode path. Target: ≥ 1.10× tok/s vs. the per-token baseline (unsloth claims 1.15–1.2× MoE, 1.4–2.2× dense; the 1.10× floor is our regression-detection threshold).

**Why this is reachable now:** the user installed `common.h`, `sampling.h`, `speculative.h`, and `llama-ext.h` to `/usr/local/include/` and `libllama-common.so.0` exports the full `common_speculative_*` C++ API including `common_speculative_impl_draft_mtp`. The wiring is no longer blocked on missing dependencies; it's a feature build.

---

## 1. Architecture

A new C++ method `Context::generate_speculative_loop(...)` owns the draft-verify inner loop. The six existing per-token generate sites in `src/bindings/llama_cpp.cpp` (lines 1044, 1361, 1440, 1475, 1551, 1640) gain a single runtime branch: if `speculative=true` and the context was constructed with `ctx_type=LLAMA_CONTEXT_TYPE_MTP`, dispatch to the new loop; otherwise the per-token path runs unchanged. This isolates risk to the new code path — every existing caller's behavior is bit-exact when `speculative=false` (the default).

The new loop wraps the upstream public C++ API:

- `common_speculative_init(params, n_seq=1)` once per generation, freed via `common_speculative_ptr` RAII.
- Per draft round: `common_speculative_get_draft_params(spec, seq_id=0)` → set `n_past`, `id_last`, `prompt`, `drafting=true`, `n_max=n_draft_max` → `common_speculative_draft(spec)` → read `*params.result` → build a multi-token `llama_batch` of `[id_last, drafted_0, ..., drafted_{k-1}]` with `logits=true` for every position → `llama_decode` once → run the sampler chain at each position to get the verified token, accept while drafted == verified, stop at first mismatch → `common_speculative_accept(spec, seq_id, n_accepted)`.
- `llama_set_embeddings_pre_norm(ctx, true, masked=true)` set once at loop entry if `common_speculative_need_embd_pre_norm(spec)` returns true; cleared on exit (RAII guard).

Python's `generate()`, `generate_stream()`, `create_chat_completion()` gain two kwargs: `speculative: bool = False` and `n_draft_max: int = 2`. Validation runs in Python (`SamplingParams.__post_init__` for `n_draft_max ∈ [1, 8]`; per-call `_validate_speculative` for the `ctx_type=MTP` precondition).

CMake gains `find_library(LLAMA_COMMON_LIB llama-common REQUIRED)` and a corresponding `target_link_libraries(_llama PRIVATE ${LLAMA_COMMON_LIB})`. Hard requirement; build fails with a clear message if `libllama-common.so` is missing.

---

## 2. Components

### C++ side (`src/bindings/llama_cpp.cpp`)

| Component | Responsibility |
|---|---|
| `Context::generate_speculative_loop(sampler, grammar, max_tokens, n_draft_max, callback) -> size_t` | Owns the draft-verify inner loop. RAII manages `common_speculative_ptr`. GIL released for the entire loop except when invoking the Python callback. Returns total tokens produced. |
| `Context::supports_speculative_mtp() const -> bool` | Returns `ctx_params_.ctx_type == LLAMA_CONTEXT_TYPE_MTP`. Predicate used by Python validation. |
| Branching in existing six generate sites | Single `if (speculative && ctx.supports_speculative_mtp())` at the top of each path; redirects to the new loop. Per-token path otherwise unchanged. |
| `Context::decode_multi(tokens, request_logits_mask) -> int32_t` | New helper. `n_tokens = 1 + n_drafted`. All positions request logits. Reuses a per-`Context` `llama_batch` buffer sized once for `1 + max_n_draft_max=8`. |
| `multi_batch_` member on `Context` | Reusable multi-token batch (mirrors the existing `single_batch_`). Allocated in `Context` constructor, freed in destructor. |

### Python side (`src/llama_cpp/llama.py`)

| Component | Responsibility |
|---|---|
| `SamplingParams.n_draft_max: int = 2` | New field. Validated `[1, 8]` in `__post_init__`; out-of-range raises `ValidationError`. |
| `Llama.generate(..., speculative=False, n_draft_max=None)` | New kwargs (also threaded through `generate_stream`, `create_chat_completion`). When `speculative=True`: validate, dispatch to C++ speculative loop. |
| `Llama._validate_speculative(speculative, ctx_type, embeddings)` | Single source of truth for the precondition check. Called by every entry point. |

### Build (`CMakeLists.txt`)

| Component | Responsibility |
|---|---|
| `find_library(LLAMA_COMMON_LIB llama-common REQUIRED)` | Hard requirement. Matches existing `find_library` pattern for `LLAMA_LIB`/`GGML_LIB`. Build fails with a clear message on machines without `libllama-common.so`. |
| Link addition | `LLAMA_COMMON_LIB` joins `LLAMA_SHARED_LIBS`. Logged via the existing `message(STATUS …)` block. |

### Out-of-scope for v1

- Eagle3 / draft-simple types (`COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3`, `COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE`). Future work; the `Context::generate_speculative_loop` API is shaped so future extension is non-breaking.
- `LlamaPool` integration is unchanged — each pool member is an independent `Llama`; if its config has `ctx_type=MTP`, speculative just works.
- No CLI/example script for benchmarking in CI; an ad-hoc `examples/bench_speculative.py` is included but not gated.

---

## 3. API surface

### Python — user-visible changes

```python
from llama_cpp import Llama, LlamaConfig, LLAMA_CONTEXT_TYPE_MTP, SamplingParams

llm = Llama(
    config=LlamaConfig(
        model_path="models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf",
        n_ctx=4096,
        ctx_type=LLAMA_CONTEXT_TYPE_MTP,
    ),
)

# Synchronous, non-streaming
text = llm.generate(
    "Explain MoE in one paragraph.",
    max_tokens=256,
    speculative=True,        # opt-in
    n_draft_max=2,           # default; range [1, 8]
    sampling=SamplingParams(temperature=0.7, top_p=0.8),
)

# Streaming
for chunk in llm.generate_stream(prompt, speculative=True, n_draft_max=2):
    print(chunk, end="", flush=True)

# Chat
resp = llm.create_chat_completion(
    messages=[...],
    speculative=True,
    n_draft_max=2,
)
```

`SamplingParams` gains `n_draft_max: int = 2`. The `speculative` flag stays a per-call kwarg (not on `SamplingParams`) because it's a runtime path selector, not a sampling hyperparameter.

### Validation contract (raises `ValidationError`)

- `speculative=True` with `ctx_type != LLAMA_CONTEXT_TYPE_MTP` → "speculative=True requires LlamaConfig(ctx_type=LLAMA_CONTEXT_TYPE_MTP); got ctx_type=DEFAULT (0)."
- `n_draft_max < 1` or `> 8` → "n_draft_max=N out of range [1, 8]."
- `speculative=True` with `config.embeddings=True` → "speculative=True is incompatible with embeddings-only contexts."

### C++ binding additions

- `Context::generate_speculative_loop(sampler, grammar, max_tokens, n_draft_max, callback) -> size_t`
- `Context::supports_speculative_mtp() const -> bool`
- Internal helpers (`decode_multi`, `multi_batch_` buffer) — not exposed via nanobind.

### Backward compatibility

`speculative` defaults to `False` and `n_draft_max` is unused on that path, so every existing caller is unaffected. `SamplingParams(n_draft_max=2)` is a no-op when `speculative=False`. The existing `ctx_type=MTP` "scaffold only" semantics still hold for users who set `ctx_type=MTP` without `speculative=True` — the runtime path is unchanged unless both flags are set.

### UnifiedLLM

`UnifiedLLM.generate()` / `chat()` thread the same two kwargs through `**kwargs` to the underlying `Llama` (no UnifiedLLM-side validation needed — the `Llama` layer rejects).

---

## 4. Data flow

### One draft-verify round (the inner loop)

```
state at round entry:
  kv_pos       = N            (KV has tokens [0..N-1])
  id_last      = T_{N-1}       (the last accepted token)
  prompt       = full token history (for spec API; ref to mirror)

step 1: draft
  params = common_speculative_get_draft_params(spec, seq_id=0)
  params.drafting = true
  params.n_max    = n_draft_max
  params.n_past   = N
  params.id_last  = T_{N-1}
  params.prompt   = &mirror_tokens
  common_speculative_draft(spec)
  drafted = *params.result        // up to n_draft_max draft ids: D_0..D_{k-1}

step 2: build multi-token batch
  batch = [T_{N-1}, D_0, ..., D_{k-1}]   // k+1 tokens
  batch.pos      = [N-1, N, ..., N+k-1]
  batch.logits   = [1, 1, ..., 1]        // all positions
  batch.seq_id   = [0, 0, ..., 0]
  llama_decode(ctx, batch)               // one call, n_tokens = k+1

  // logits[0]   = P(t_N    | T_0..T_{N-1})
  // logits[i]   = P(t_{N+i}| T_0..T_{N-1}, D_0..D_{i-1})  for i >= 1

step 3: verify
  for i in 0..k:
    cur_p = build_candidates_from_logits(logits[i])
    if grammar:  llama_sampler_apply(grammar, &cur_p)   // grammar gates verification
    llama_sampler_apply(sampler, &cur_p)                // full chain (logit_bias..dist)
    verified_i = cur_p.data[cur_p.selected].id
    if i < k and verified_i == drafted[i]:
        accept verified_i, continue
    else:
        accept verified_i (the corrected token), stop drafting this round

  n_accepted = i + 1
  for j in 0..n_accepted-1: llama_sampler_accept(sampler, accepted[j])
  if grammar: for j in 0..n_accepted-1: llama_sampler_accept(grammar, accepted[j])

step 4: housekeeping
  common_speculative_accept(spec, seq_id=0, n_accepted)
  if n_accepted < k+1:
      llama_kv_cache_seq_rm(ctx, 0, N + n_accepted, -1)   // trim rejected drafts
  cur_pos += n_accepted
  id_last = accepted[n_accepted - 1]
  emit accepted tokens to callback (streaming) or buffer (non-streaming)
  check stop sequences against accepted tokens (multi-token-stop logic unchanged)
  check max_tokens budget; if exhausted, exit loop
```

### Loop entry / exit (once per `generate_*` call)

```
entry:
  spec = common_speculative_init(params, n_seq=1)    // RAII: common_speculative_ptr
  if common_speculative_need_embd_pre_norm(spec):
      llama_set_embeddings_pre_norm(ctx, true, /*masked=*/true)
exit (normal or exceptional):
  if entry called set_embeddings_pre_norm:
      llama_set_embeddings_pre_norm(ctx, false, false)
  spec.reset()                                       // RAII frees on scope exit
```

### Cache_prompt + speculative interaction

The existing `_apply_prefix_reuse` path runs *before* the speculative loop starts, exactly as it does today for the per-token path:

1. Compute LCP between mirror and new prompt → `n_match`.
2. `kv_cache_seq_rm(0, n_match, -1)`.
3. Decode `priming[n_match:]` per-token (re-prime).
4. Set `mirror = new_prompt`, `kv_pos = len(new_prompt)`, `id_last = new_prompt[-1]`.
5. **Then** enter the speculative draft-verify loop.

Mirror invariant `len(mirror) == kv_pos_max + 1` is preserved by appending only verified tokens to the mirror, never drafts. On hybrid models (`memory_can_shift()=False`) the existing fallback to `kv_cache_clear` + full re-prime fires before speculative starts; correctness preserved, prefix-reuse speedup lost (same as today's behavior).

### Stop sequences

When a multi-token stop sequence partially matches across drafted tokens, the existing `generate_tokens_multi_stop` buffering logic still applies — accepted tokens are buffered up to `max_stop_len` before emitting through the callback. Drafts that haven't been verified are never buffered (they don't exist yet from the Python side).

---

## 5. Error handling, threading, lifetimes

### Threading and GIL

The speculative loop runs under `nb::call_guard<nb::gil_scoped_release>()` for the full body, exactly like the existing per-token paths. The GIL is re-acquired only when invoking the Python streaming callback (matching `generate_stream`'s `nb::gil_scoped_acquire` block). No new threads. `LlamaPool` is unaffected — each pool member is an independent `Llama` and `Context`, so the `common_speculative *` lifetime is per-instance.

### Resource lifetimes

| Object | Owner | Freed |
|---|---|---|
| `common_speculative *` | `Context::generate_speculative_loop` local `common_speculative_ptr` | Scope exit (RAII), or earlier on exception |
| `multi_batch_` (multi-token `llama_batch`) | `Context` member, sized once for `1 + 8` | `~Context` via existing pattern |
| `pre_norm` toggle | RAII guard local to the loop | Cleared on scope exit (normal or exception) |
| `LLAMA_COMMON_LIB` linkage | CMake, build-time | n/a |

The `common_speculative_ptr` typedef from `<speculative.h>` already wraps `common_speculative_free` in a unique_ptr deleter — we reuse it directly.

### Failure modes

| Failure | Where caught | Surface |
|---|---|---|
| `speculative=True` on non-MTP context | Python `_validate_speculative()` at call entry | `ValidationError` |
| `n_draft_max ∉ [1, 8]` | `SamplingParams.__post_init__` | `ValidationError` |
| `llama_decode` fails on multi-token batch | C++ `decode_multi`, return code check | `std::runtime_error` → Python `LlamaError` |
| `common_speculative_init` returns null | C++ at loop entry | `std::runtime_error` (model has MTP graph but no draft-MTP impl available) |
| Sampler chain emptied by grammar (verify selects no token) | C++ verify step, existing `cur_p.selected` validation | `std::runtime_error` (existing path) |
| `kv_cache_seq_rm` returns false during reject-cleanup | C++; treat as fatal because mirror invariant would diverge | `std::runtime_error` |
| `libllama-common.so` missing at build time | CMake `find_library(REQUIRED)` | Build fails with clear message |
| `libllama-common.so` missing at runtime (post-build) | dynamic linker | `ImportError` on `import llama_cpp._llama` (existing path for ggml-cuda etc.) |

### Cancellation / streaming early-exit

The streaming callback returns a bool; when it returns `false` (consumer disconnected), the loop exits cleanly: free `spec`, restore pre-norm flag, return current token count. Drafts in flight at that moment have already been decoded into KV; we trim with `kv_cache_seq_rm(0, kv_pos + n_accepted, -1)` before returning, so the next call sees a clean state. This mirrors the existing per-token path's behavior; the only addition is the trim of unaccepted drafts.

### LoRA, grammar, logit_bias, DRY interaction

- **LoRA adapters:** applied at the model/context level — unaffected by speculative. The existing `_reapply_lora_adapters()` after `reset()` continues to fire.
- **Grammar:** applied to each *verified* token's candidate set (not to drafts). Grammar's internal accept advances only on accepted tokens; rejected-draft positions are not seen by the grammar. The documented rule "grammar applies before sampler chain" still holds at each verify step.
- **Logit bias / DRY / penalties / typical_p / etc.:** the full sampler chain runs at each verify step from the multi-token batch's logits. Output is bit-exact with the per-token path for the same seed when no draft is accepted, and statistically equivalent (modulo RNG advancement order across positions) when drafts are accepted. We document the "same seed → same output" guarantee for non-speculative; speculative guarantees the same *distribution*, not the same *trajectory*, because the dist sampler's RNG advances per-position rather than per-step.

---

## 6. Testing & verification

### Unit tests (no model required, run on CI)

`tests/test_speculative_validation.py` (~6 tests):

- `n_draft_max=0` rejected with `ValidationError`; `n_draft_max=9` rejected; `n_draft_max=2` accepted as default.
- `speculative=True` with `ctx_type=DEFAULT` raises `ValidationError` with actionable message.
- `speculative=False` is the default; existing kwargs unchanged.
- `Context.supports_speculative_mtp()` returns the expected boolean for both `ctx_type` values (uses standard test model — does not require MTP support, just probes the predicate).
- `SamplingParams(n_draft_max=2)` round-trips via `to_native()`.
- `speculative=True` + `config.embeddings=True` raises `ValidationError`.

### End-to-end tests

`tests/test_speculative_mtp.py`, gated on `LLAMA_MTP_TEST_MODEL` (defaults to `models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`); skip with a clear message when absent (~8 tests):

| Test | Coverage |
|---|---|
| `test_speculative_smoke` | `speculative=True, n_draft_max=2` produces non-empty output, no errors |
| `test_speculative_matches_baseline_greedy` | `temperature=0`, `speculative=True` and `speculative=False` produce **identical** token sequences |
| `test_speculative_matches_baseline_seeded` | With fixed seed and equivalent sampler config, both paths produce equal *length* output (acceptance-rate-dependent, but same stop point) |
| `test_speculative_streaming` | `generate_stream(..., speculative=True)` yields the same concatenated text as non-streaming |
| `test_speculative_with_grammar` | JSON grammar + `speculative=True` produces grammar-valid output |
| `test_speculative_with_cache_prompt` | Two-turn chat with `cache_prompt=True, reset_kv_cache=False, speculative=True` extends the mirror correctly; `_cached_prompt_tokens` invariant holds |
| `test_speculative_n_draft_max_bounds` | `n_draft_max=1` and `n_draft_max=8` both succeed end-to-end |
| `test_speculative_stop_sequence` | Multi-token stop sequence is honored under speculative path; no leak past the stop |

### Manual benchmark (ad-hoc, not in CI)

`examples/bench_speculative.py`:

- Same prompt, same sampler config, same model, two runs (`speculative=False`, then `speculative=True, n_draft_max=2`).
- Reports tok/s for each, plus speedup ratio.
- Writes nothing, prints to stdout. Used for tuning and reproducibility.

### CI verification commands

```bash
uv run pytest tests/test_speculative_validation.py -q   # unit tests, fast
uv run pytest tests/test_speculative_mtp.py -q          # end-to-end, needs MTP model (skips when absent)
ruff check src/ tests/
clang-format --dry-run -Werror src/bindings/llama_cpp.cpp
```

### Acceptance criteria for v1

1. All speculative tests pass on a machine with the Qwen3.6-MoE checkpoint.
2. All other tests continue to pass (no regressions).
3. `ruff` clean; `clang-format` clean.
4. `examples/bench_speculative.py` shows ≥ 1.10× tok/s on Qwen3.6-MoE with `n_draft_max=2` vs. `speculative=False` baseline (gives a margin under unsloth's claimed 1.15–1.2×; below 1.10× signals an implementation bug).
5. README, API.md, CHANGELOG, CLAUDE.md updated to replace "scaffold only — no acceleration today" with the new working semantics.

---

## 7. Risks and open questions

### Risks

- **Sampler RNG trajectory divergence.** The non-speculative path advances the dist sampler's RNG once per token; the speculative path advances it once per *position* in each batch. Same seed → same distribution but not the same exact sequence. We document this explicitly in API.md so users running reproducibility tests aren't surprised.
- **Hybrid models with `memory_can_shift() == False`.** Speculative requires `kv_cache_seq_rm` to trim rejected drafts. If `seq_rm` returns `false`, we treat it as fatal (the per-token fallback for prefix-reuse doesn't apply here — there's no clean way to back out an already-decoded multi-token batch without `seq_rm`). We document this as "speculative requires `memory_can_shift()=True`."
- **`libllama-common.so` ABI changes.** Upstream's `common/` is C++ (not C); ABI is not stable across llama.cpp versions. We pin to the linked-against version at build time and rebuild on llama.cpp upgrades — same discipline as the rest of the binding.
- **Acceptance rate floor.** If the MTP head is poorly calibrated for a given prompt, the verify step rejects all drafts and we pay the multi-token-batch cost without payoff. The bench script catches this; users with marginal hardware may want to set `n_draft_max=1`.

### Open questions (resolved during brainstorming)

All resolved. None remaining.

### Revisit triggers

- Benchmark below 1.10× on Qwen3.6-MoE → bug in the verify or accept logic.
- Upstream changes the `common_speculative_*` API (e.g., adds a `n_seq > 1` requirement) → re-spec.
- A user requests Eagle3 / draft-simple — design supports it as a non-breaking extension (new enum value on `params.types`, no API change).
