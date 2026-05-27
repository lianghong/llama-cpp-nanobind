# MTP Documentation Correction — Design

**Date:** 2026-05-27
**Status:** approved scope, pending user review of written spec
**Owner:** lianghong

## Problem

The 2026-05-27 release shipped `LlamaConfig.ctx_type=LLAMA_CONTEXT_TYPE_MTP` (1) and presented it in user-facing docs as a path to speed up generation on Qwen3.5 / Qwen3.5-MoE / Qwen3.6-MoE checkpoints that ship Multi-Token Prediction layers. That presentation is wrong for the current code state and produces a net regression for any user who follows it.

### Verified facts

1. **Per-token decode loop, no draft-verify consumer.** All six generation paths in `src/bindings/llama_cpp.cpp` call `ctx.decode_one(token, /*request_logits=*/true)`:
   - `llama_cpp.cpp:1044, 1361, 1440, 1475, 1551, 1640`
   - `Context::decode_one` (`llama_cpp.cpp:616`) sets `single_batch_.n_tokens = 1` before `llama_decode`. Single-token batches; no multi-token verify; no acceptance/rollback machinery.

2. **MTP heads compute then their output is discarded.** Setting `ctx_type=LLAMA_CONTEXT_TYPE_MTP` selects the MTP graph variant in upstream `llama.cpp` at context construction. With no consumer reading the auxiliary head outputs, every step pays the extra FLOPs and memory traffic for no throughput benefit.

3. **Effect of `ctx_type=MTP` today:**
   - Per-step compute: **higher** (extra MTP head FLOPs + memory traffic)
   - Tokens generated per `llama_decode`: **still 1**
   - VRAM: **higher** (MTP layer activations)
   - Throughput: **lower** than `LLAMA_CONTEXT_TYPE_DEFAULT` on the same model

4. **The acceleration unsloth advertises lives in upstream `common/`, not in `llama.h`.** Unsloth's `--spec-type draft-mtp --spec-draft-n-max 2` flag (renamed from `--spec-type mtp` on 2026-05-13) is satisfied by `common_speculative_impl_draft_mtp` in `common/speculative.cpp:409`, which:
   - Requires **two contexts** (`ctx_tgt` + `ctx_dft`) — `GGML_ASSERT` at `common/speculative.cpp:443-445`
   - Calls `llama_set_embeddings_pre_norm`, declared in `src/llama-ext.h` (header documented as *"this is a staging header for new llama.cpp API. breaking changes and C++ are allowed. everything here should be considered WIP"*) and **not installed** to `/usr/local/include/`
   - Depends on `libllama-common`, which is built but **not installed** by upstream `make install`

   None of these are reachable from our bindings against the system-installed llama.cpp.

## Decision

**Doc-only correction now. No code changes. Real implementation deferred behind concrete upstream-API triggers.**

The `ctx_type` plumbing (`src/bindings/llama_cpp.cpp:1751-1755`, `LlamaConfig.ctx_type`, `LLAMA_CONTEXT_TYPE_*` constants, `tests/test_mtp.py`) is kept. It is correct as a graph-variant scaffold and is the right trigger point for when upstream lands a stable consumer API. The bug is the user-facing presentation in README, API reference, and changelog, which oversold the user-visible effect.

### Why not implement now

Cost/benefit at 2026-05-27:

- Unsloth's published numbers: **1.4–2.2× on dense, 1.15–1.2× on MoE.** The repo's MTP test target is `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` — MoE — so the relevant gain is 15–20%.
- Implementation cost to reach that 15–20%:
  - Vendor `common/` source tree (transitive deps: `sampling.cpp`, `log.cpp`, `ngram-*.cpp`, `arg.cpp`, etc., or surgical excision)
  - Take a hard dependency on `llama-ext.h`, which upstream documents as WIP and renamed `--spec-type mtp` → `--spec-type draft-mtp` two weeks before this date
  - ~2× VRAM for target + draft contexts on the same model
  - Refactor all six generate paths into draft → batch-verify → accept-prefix → KV-rollback structure with sampler-state rollback for rejected tokens
  - New equivalence + acceptance-rate test matrix

- Violates project discipline: `CLAUDE.md` states *"Extension links against system-installed llama.cpp (no bundling, no RPATH)."* Vendoring `common/` is a real break, not a tweak.

This is not a permanent refusal. It is a deferral until the upstream cost drops.

### Revisit triggers

Reopen this design when **any** of:

1. `llama_set_embeddings_pre_norm` (and the rest of the speculative consumer surface) is promoted from `src/llama-ext.h` to public `include/llama.h` and installed to `/usr/local/include/`.
2. Upstream `make install` ships `libllama-common.{so,a}` plus its public headers (`common/speculative.h`, `common/common.h`).
3. Unsloth publishes draft-MTP MoE numbers above ~1.5× (would shift the MoE-only cost/benefit enough to justify the work).

## Scope

### In scope

Edits to four files:

1. **`README.md`** — replace the "MTP (Multi-Token Prediction)" section (currently around lines 525–542). New copy must contain:
   - One-sentence what-it-is.
   - Explicit warning: this does not accelerate generation today. Without a draft-verify consumer in the bindings, `ctx_type=LLAMA_CONTEXT_TYPE_MTP` adds extra per-step compute (MTP heads run, output discarded). Leave at the default unless testing the graph path.
   - Pointer to upstream: real speedup requires the `common_speculative` draft-verify loop, which depends on staging APIs (`llama-ext.h`) not yet in public `llama.h`. Tracked for a future revision.

2. **`docs/API.md`** — same correction in the API reference (currently around lines 268–287). Add a "Status" line under the section heading: *"Scaffold only. No throughput benefit in v0.4.x."*

3. **`docs/CHANGELOG-2026-05-27.md`** — append an "Errata" block to the MTP entry (currently around lines 139–202) clarifying that the original wording overstated the user-visible effect. Name `COMMON_SPECULATIVE_TYPE_DRAFT_MTP` and `--spec-type draft-mtp` (renamed 2026-05-13) as the upstream path that would deliver actual acceleration.

4. **`CLAUDE.md`** — extend the existing `**MTP (Multi-Token Prediction)**` paragraph in the "Key design patterns" section. Keep the architectural facts (graph-variant decision at context-construction time, `ModelLoadError` on non-MTP models). Add: *"No throughput benefit in this codebase; runtime/decode loop is unchanged. Acceleration would require draft-verify (`common_speculative`) wiring against APIs currently only in `llama-ext.h` (staging). Not present here."*

   This is the canonical contributor-facing doc; correctness here matters most.

### Out of scope

- Removing `ctx_type` plumbing or `LLAMA_CONTEXT_TYPE_*` constants.
- Removing `tests/test_mtp.py`. It guards the graph-construction path and the `ModelLoadError` contract — both still valid.
- Adding a runtime warning print when users set `ctx_type=MTP`. Docs are the right channel; per-construction prints become noise.
- Any changes to `src/bindings/llama_cpp.cpp` or `src/llama_cpp/`.

## Test plan

None. No code changes. Existing `tests/test_mtp.py` continues to pass unchanged.

Verification before commit:
- `git diff --stat` shows only the four files above.
- `uv run pytest tests/test_mtp.py -q` still passes.

## Risks

1. **Doc drift.** The corrected docs name specific upstream symbols (`llama-ext.h`, `common_speculative`, `COMMON_SPECULATIVE_TYPE_DRAFT_MTP`, `--spec-type draft-mtp`). If upstream renames these again, the corrected docs go stale. Mitigation: the revisit triggers above name *behaviors* (header promotion, install artifact, throughput numbers), not symbol names — so the gating logic survives renames even if the prose ages.

2. **User confusion.** Users who already set `ctx_type=LLAMA_CONTEXT_TYPE_MTP` based on the old docs will not see a warning at runtime — only at next docs read. Accepted: a runtime warning would be noise, and the existing `ModelLoadError` on non-MTP models already prevents the worst-case foot-gun.

3. **Reviewer asks "why not just delete it."** Answer is in this doc: the wiring is correct as a graph-variant scaffold, deletion costs us when the upstream API stabilizes, and the test guards stay valuable.

## Commit

Single commit on the working branch:

```
docs: correct MTP scope (graph variant only, no acceleration consumer)

The 2026-05-27 release oversold ctx_type=LLAMA_CONTEXT_TYPE_MTP as a path
to speed up generation. The bindings have no draft-verify consumer; setting
the MTP graph variant adds compute with no throughput benefit. Correct
README, API reference, changelog, and CLAUDE.md to match reality. Wiring
and tests retained as the trigger point for when upstream promotes the
common_speculative consumer API to public llama.h.
```
