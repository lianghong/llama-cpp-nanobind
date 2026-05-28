# MTP Documentation Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct user-facing and contributor-facing documentation so that `LlamaConfig.ctx_type=LLAMA_CONTEXT_TYPE_MTP` is described as a graph-variant scaffold with no acceleration consumer, not a path to faster generation.

**Architecture:** Doc-only edits across four files (`README.md`, `docs/API.md`, `docs/CHANGELOG-2026-05-27.md`, `CLAUDE.md`). No source or test changes. The `ctx_type` plumbing in `src/bindings/llama_cpp.cpp` and `tests/test_mtp.py` remain unchanged — they are the trigger point for revisiting the feature when upstream APIs stabilize.

**Tech Stack:** Markdown only. No build, no compilation, no Python.

**Spec:** `docs/superpowers/specs/2026-05-27-mtp-doc-correction-design.md` (commit `484177b`).

---

## Background context

Read this before starting. Skip nothing.

### Why this plan exists

The 2026-05-27 release shipped `LlamaConfig.ctx_type=LLAMA_CONTEXT_TYPE_MTP` and presented it in user-facing docs as a feature toggle for Qwen3.5 / Qwen3.5-MoE / Qwen3.6-MoE checkpoints with MTP layers. That presentation is wrong:

1. The bindings have no draft-verify consumer. All six generation paths in `src/bindings/llama_cpp.cpp` (lines `1044, 1361, 1440, 1475, 1551, 1640`) call `ctx.decode_one(token, /*request_logits=*/true)` — single-token batches. `single_batch_.n_tokens = 1` at `llama_cpp.cpp:619`.
2. Setting `ctx_type=LLAMA_CONTEXT_TYPE_MTP` selects the MTP graph variant in `llama.cpp`. With no consumer reading the auxiliary head outputs, the MTP layers compute every step and their output is discarded.
3. Net effect: extra per-step compute, extra VRAM, **lower** throughput than the default.

Unsloth's `--spec-type draft-mtp --spec-draft-n-max 2` flag (renamed from `--spec-type mtp` on 2026-05-13) reaches its 1.4–2.2× dense / 1.15–1.2× MoE speedup via `common_speculative_impl_draft_mtp` in upstream `common/speculative.cpp:409`, which:
- requires two contexts (`ctx_tgt` + `ctx_dft`) — `GGML_ASSERT` at `common/speculative.cpp:443-445`
- calls `llama_set_embeddings_pre_norm`, declared in `src/llama-ext.h` (upstream documents this header as *"this is a staging header for new llama.cpp API. breaking changes and C++ are allowed. everything here should be considered WIP"*) and **not installed** to `/usr/local/include/`
- depends on `libllama-common`, which upstream builds but does **not install**

None of these are reachable from the bindings against the system-installed llama.cpp today. That's why this is a doc-only correction, not a feature build.

### Style rules for edits in this plan

- Match the surrounding tone of each file. README is user-facing prose. API.md is reference-style with tables. The changelog is past-tense bullet lists. CLAUDE.md is dense, one-paragraph-per-pattern guidance for contributors.
- Do not introduce new headings or restructure sections. Replace prose in place.
- Keep code examples that demonstrate the API — they're still correct as code. The fix is the surrounding warning text.
- No emojis.

---

## File structure

| File | Change |
|---|---|
| `README.md` | Replace the body of "MTP (Multi-Token Prediction)" (lines 525–542). |
| `docs/API.md` | Add a "Status" line under the "MTP context type" heading (line 268) and a warning paragraph after the table (after line 277). |
| `docs/CHANGELOG-2026-05-27.md` | Append an "Errata" subsection to the MTP entry (after line 165, before the `## Tests` heading at line 167). |
| `CLAUDE.md` | Extend the existing MTP paragraph at line 88. |

No new files. No code changes. No test changes.

---

## Task 1: README.md correction

**Files:**
- Modify: `README.md:525-542`

- [ ] **Step 1: Verify current state**

Run: `sed -n '525,542p' README.md`

Expected output (verbatim, including blank lines):

```
### MTP (Multi-Token Prediction)

For Qwen3.5 / Qwen3.5-MoE / Qwen3.6-MoE checkpoints that ship MTP layers:

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

Loading a non-MTP model with `ctx_type=LLAMA_CONTEXT_TYPE_MTP` raises `ModelLoadError`.
```

If it does not match, stop and re-read the file before editing — the line numbers may have drifted.

- [ ] **Step 2: Replace the section**

Use Edit to replace this exact `old_string`:

```
### MTP (Multi-Token Prediction)

For Qwen3.5 / Qwen3.5-MoE / Qwen3.6-MoE checkpoints that ship MTP layers:

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

Loading a non-MTP model with `ctx_type=LLAMA_CONTEXT_TYPE_MTP` raises `ModelLoadError`.
```

with this `new_string`:

```
### MTP (Multi-Token Prediction)

`LlamaConfig.ctx_type=LLAMA_CONTEXT_TYPE_MTP` selects the MTP graph variant in llama.cpp at context-construction time, for Qwen3.5 / Qwen3.5-MoE / Qwen3.6-MoE checkpoints that ship MTP layers (`*.nextn_predict_layers > 0` metadata + `blk.*.nextn.*` tensors).

> **This does not accelerate generation today.** The bindings' generate loop is strictly per-token (`llama_decode` with `n_tokens = 1`). Without a draft-verify consumer, setting `ctx_type=LLAMA_CONTEXT_TYPE_MTP` runs the auxiliary MTP heads each step and discards their output — extra compute and extra VRAM for no throughput benefit. Leave it at the default (`LLAMA_CONTEXT_TYPE_DEFAULT`) unless you are deliberately exercising the graph path.

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

Loading a non-MTP model with `ctx_type=LLAMA_CONTEXT_TYPE_MTP` raises `ModelLoadError`.

The acceleration that upstream's `--spec-type draft-mtp` flag delivers (1.4–2.2× dense, 1.15–1.2× MoE per unsloth) is implemented in `common/speculative.cpp` and depends on staging APIs (`llama-ext.h`) not yet promoted to public `llama.h`. Tracked for a future revision once those APIs stabilize and ship in the installed headers.
```

- [ ] **Step 3: Verify the replacement**

Run: `sed -n '523,548p' README.md`

Expected: the new section appears in place. The blockquote (`>`) line is present. The closing paragraph mentioning `--spec-type draft-mtp` is present. The "Loading a non-MTP model …" sentence is preserved verbatim.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs(readme): correct MTP scope (graph variant only, no acceleration)

The 2026-05-27 release oversold ctx_type=LLAMA_CONTEXT_TYPE_MTP as a
path to speed up generation. The generate loop is strictly per-token
(n_tokens=1); MTP heads compute and their output is discarded. Add a
prominent warning and a pointer to the upstream draft-verify path that
would actually deliver the unsloth-published numbers.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: docs/API.md correction

**Files:**
- Modify: `docs/API.md:268-290`

- [ ] **Step 1: Verify current state**

Run: `sed -n '268,290p' docs/API.md`

Expected output (verbatim):

```
#### MTP context type

Constants exported from `llama_cpp` (mirror `llama.h`'s `enum llama_context_type`):

| Constant | Value | Notes |
| --- | --- | --- |
| `LLAMA_CONTEXT_TYPE_DEFAULT` | 0 | Default graph (current behavior) |
| `LLAMA_CONTEXT_TYPE_MTP` | 1 | Multi-Token Prediction graph variant |

MTP requires a checkpoint that ships MTP layers — currently Qwen3.5 / Qwen3.5-MoE / Qwen3.6-MoE MTP variants (metadata `*.nextn_predict_layers > 0`, plus `blk.*.nextn.*` tensors). Loading a non-MTP model with `ctx_type=LLAMA_CONTEXT_TYPE_MTP` raises `ModelLoadError` (`"context type MTP requested but model doesn't contain MTP layers"`). The generation API is otherwise unchanged — MTP is a graph-construction-time decision, not a runtime sampler.

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
```

If it does not match, stop and re-read before editing.

- [ ] **Step 2: Insert "Status" line and warning paragraph**

Use Edit to replace this exact `old_string`:

```
#### MTP context type

Constants exported from `llama_cpp` (mirror `llama.h`'s `enum llama_context_type`):

| Constant | Value | Notes |
| --- | --- | --- |
| `LLAMA_CONTEXT_TYPE_DEFAULT` | 0 | Default graph (current behavior) |
| `LLAMA_CONTEXT_TYPE_MTP` | 1 | Multi-Token Prediction graph variant |

MTP requires a checkpoint that ships MTP layers — currently Qwen3.5 / Qwen3.5-MoE / Qwen3.6-MoE MTP variants (metadata `*.nextn_predict_layers > 0`, plus `blk.*.nextn.*` tensors). Loading a non-MTP model with `ctx_type=LLAMA_CONTEXT_TYPE_MTP` raises `ModelLoadError` (`"context type MTP requested but model doesn't contain MTP layers"`). The generation API is otherwise unchanged — MTP is a graph-construction-time decision, not a runtime sampler.
```

with this `new_string`:

```
#### MTP context type

**Status:** Scaffold only. No throughput benefit in v0.4.x — the generate loop is per-token and there is no draft-verify consumer. See the warning below.

Constants exported from `llama_cpp` (mirror `llama.h`'s `enum llama_context_type`):

| Constant | Value | Notes |
| --- | --- | --- |
| `LLAMA_CONTEXT_TYPE_DEFAULT` | 0 | Default graph (current behavior) |
| `LLAMA_CONTEXT_TYPE_MTP` | 1 | Multi-Token Prediction graph variant |

MTP requires a checkpoint that ships MTP layers — currently Qwen3.5 / Qwen3.5-MoE / Qwen3.6-MoE MTP variants (metadata `*.nextn_predict_layers > 0`, plus `blk.*.nextn.*` tensors). Loading a non-MTP model with `ctx_type=LLAMA_CONTEXT_TYPE_MTP` raises `ModelLoadError` (`"context type MTP requested but model doesn't contain MTP layers"`). The generation API is otherwise unchanged — MTP is a graph-construction-time decision, not a runtime sampler.

> **No acceleration consumer in this binding.** Setting `LLAMA_CONTEXT_TYPE_MTP` runs the MTP auxiliary heads on every step and discards their output, because the generate loop calls `llama_decode` with `n_tokens = 1` and never verifies a draft batch. Net effect today is **lower** throughput than the default. Acceleration of the kind unsloth's `--spec-type draft-mtp` advertises (1.4–2.2× dense, 1.15–1.2× MoE) lives in upstream `common/speculative.cpp` and depends on `llama-ext.h` staging APIs (`llama_set_embeddings_pre_norm`) and `libllama-common`, neither of which is exposed by the system-installed llama.cpp today. The wiring here is retained as a graph-variant probe and as the trigger point for when those APIs are promoted.
```

- [ ] **Step 3: Verify the replacement**

Run: `sed -n '266,295p' docs/API.md`

Expected: the **Status:** line is present. The blockquote warning is present after the existing paragraph. The Python example below is unchanged.

- [ ] **Step 4: Commit**

```bash
git add docs/API.md
git commit -m "$(cat <<'EOF'
docs(api): mark MTP context type as scaffold only

Add a Status line and a warning paragraph clarifying that
ctx_type=LLAMA_CONTEXT_TYPE_MTP has no acceleration consumer in the
bindings and is a net regression today. Reference the upstream path
(common/speculative.cpp + llama-ext.h) that would deliver actual
speedup once it stabilizes.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: CHANGELOG errata

> **Plan deviation (executed 2026-05-27):** the MTP changelog entry was discovered to be in-flight working-tree content, not committed at HEAD. An "Errata" block on an unpublished entry would be confusing. Instead, the in-flight wording was extended in place with a "**Scope — scaffold only, no acceleration today.**" paragraph covering the same content as the planned errata. No separate commit was made; the file remains in the working tree as M for the user to commit alongside the rest of the in-flight 2026-05-27 changelog work. The original step-by-step instructions below are retained for historical reference.

**Files:**
- Modify: `docs/CHANGELOG-2026-05-27.md:165-167`

- [ ] **Step 1: Verify current state around the insertion point**

Run: `sed -n '160,170p' docs/CHANGELOG-2026-05-27.md`

Expected output (verbatim):

```
**Behavior on incompatible models:** llama.cpp raises at context
construction with `"context type MTP requested but model doesn't
contain MTP layers"`; surfaced as `ModelLoadError` here. There is no
runtime/sampler-level surface — MTP is a graph-time decision and the
generation loop is otherwise unchanged.

## Tests
```

- [ ] **Step 2: Insert errata subsection between line 165 and the `## Tests` heading**

Use Edit to replace this exact `old_string`:

```
**Behavior on incompatible models:** llama.cpp raises at context
construction with `"context type MTP requested but model doesn't
contain MTP layers"`; surfaced as `ModelLoadError` here. There is no
runtime/sampler-level surface — MTP is a graph-time decision and the
generation loop is otherwise unchanged.

## Tests
```

with this `new_string`:

```
**Behavior on incompatible models:** llama.cpp raises at context
construction with `"context type MTP requested but model doesn't
contain MTP layers"`; surfaced as `ModelLoadError` here. There is no
runtime/sampler-level surface — MTP is a graph-time decision and the
generation loop is otherwise unchanged.

**Errata.** The original wording above understated one consequence:
because the generate loop in this binding is strictly per-token
(`llama_decode` with `n_tokens = 1`), enabling
`LLAMA_CONTEXT_TYPE_MTP` is a **net throughput regression** on the
same model — the auxiliary heads compute every step and their output
is discarded. The acceleration unsloth advertises via
`--spec-type draft-mtp --spec-draft-n-max 2` (renamed from
`--spec-type mtp` on 2026-05-13) is implemented in upstream
`common/speculative.cpp` as `COMMON_SPECULATIVE_TYPE_DRAFT_MTP`,
which requires a separate draft context, calls `llama-ext.h` staging
APIs (`llama_set_embeddings_pre_norm`), and links against
`libllama-common` — none of which are reachable from the system-installed
llama.cpp today. The `ctx_type` plumbing and `tests/test_mtp.py`
are retained as the trigger point for when the upstream consumer
API is promoted to public `llama.h`.

## Tests
```

- [ ] **Step 3: Verify the replacement**

Run: `sed -n '160,190p' docs/CHANGELOG-2026-05-27.md`

Expected: the `**Errata.**` paragraph appears between the existing "Behavior on incompatible models" paragraph and the `## Tests` heading. The `## Tests` heading is still on its own line.

- [ ] **Step 4: Commit**

```bash
git add docs/CHANGELOG-2026-05-27.md
git commit -m "$(cat <<'EOF'
docs(changelog): errata for MTP context type

The 2026-05-27 entry described ctx_type=MTP as having no
runtime-level surface, which understated the actual cost: enabling it
is a net throughput regression because the generate loop is per-token
and never consumes the auxiliary MTP heads. Document the regression
explicitly and name the upstream draft-verify path
(common_speculative + llama-ext.h) that would deliver the unsloth
numbers once promoted.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: CLAUDE.md correction

> **Plan deviation (executed 2026-05-27):** like the changelog, the `**MTP (Multi-Token Prediction)**` paragraph at line 88 was discovered to be in-flight working-tree content, not committed at HEAD. The plan called for an extension of an "existing" paragraph; in practice the entire paragraph is unpublished, so the in-flight wording was rewritten in place with the scaffold-only / no-throughput-benefit framing. No separate commit was made; the file remains in the working tree as M for the user to commit alongside the rest of the in-flight 2026-05-27 work. The original step-by-step instructions below are retained for historical reference.

**Files:**
- Modify: `CLAUDE.md:88`

- [ ] **Step 1: Verify current state of the MTP paragraph**

Run: `sed -n '88,88p' CLAUDE.md`

Expected output (single line, verbatim):

```
**MTP (Multi-Token Prediction)** — `LlamaConfig.ctx_type=LLAMA_CONTEXT_TYPE_MTP` (1) selects the MTP graph variant in llama.cpp at context-construction time. Requires a checkpoint that ships MTP layers (`*.nextn_predict_layers > 0` metadata + `blk.*.nextn.*` tensors); currently produced for Qwen3.5 / Qwen3.5-MoE / Qwen3.6-MoE MTP variants. On a non-MTP model, libllama errors with `"context type MTP requested but model doesn't contain MTP layers"`, surfaced here as `ModelLoadError`. No runtime/sampler-level surface — generation loop is unchanged.
```

- [ ] **Step 2: Replace the paragraph**

Use Edit to replace this exact `old_string`:

```
**MTP (Multi-Token Prediction)** — `LlamaConfig.ctx_type=LLAMA_CONTEXT_TYPE_MTP` (1) selects the MTP graph variant in llama.cpp at context-construction time. Requires a checkpoint that ships MTP layers (`*.nextn_predict_layers > 0` metadata + `blk.*.nextn.*` tensors); currently produced for Qwen3.5 / Qwen3.5-MoE / Qwen3.6-MoE MTP variants. On a non-MTP model, libllama errors with `"context type MTP requested but model doesn't contain MTP layers"`, surfaced here as `ModelLoadError`. No runtime/sampler-level surface — generation loop is unchanged.
```

with this `new_string`:

```
**MTP (Multi-Token Prediction)** — `LlamaConfig.ctx_type=LLAMA_CONTEXT_TYPE_MTP` (1) selects the MTP graph variant in llama.cpp at context-construction time. Requires a checkpoint that ships MTP layers (`*.nextn_predict_layers > 0` metadata + `blk.*.nextn.*` tensors); currently produced for Qwen3.5 / Qwen3.5-MoE / Qwen3.6-MoE MTP variants. On a non-MTP model, libllama errors with `"context type MTP requested but model doesn't contain MTP layers"`, surfaced here as `ModelLoadError`. **Scaffold only — no throughput benefit in this binding.** The generate loop is strictly per-token (`llama_decode` with `n_tokens = 1`); MTP heads compute and their output is discarded, so enabling `LLAMA_CONTEXT_TYPE_MTP` is a net regression on the same model. Real acceleration (1.4–2.2× dense, 1.15–1.2× MoE per unsloth) requires draft-verify (`common_speculative_impl_draft_mtp` in upstream `common/speculative.cpp`), which calls `llama-ext.h` staging APIs (`llama_set_embeddings_pre_norm`) and links `libllama-common` — neither exposed by the system-installed llama.cpp today. Wiring retained as the trigger point for when those APIs are promoted to public `llama.h`. Until then: leave at `LLAMA_CONTEXT_TYPE_DEFAULT`.
```

- [ ] **Step 3: Verify the replacement**

Run: `sed -n '88,88p' CLAUDE.md`

Expected: a single (long) line containing the new paragraph. The bolded `**Scaffold only — no throughput benefit in this binding.**` phrase is present. The closing "Until then: leave at `LLAMA_CONTEXT_TYPE_DEFAULT`." is present.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
docs(claude): mark MTP context type as scaffold only

CLAUDE.md is the canonical contributor doc and was missing the cost
side of the MTP entry. Add the explicit "no throughput benefit"
warning, name the upstream draft-verify path that would deliver the
unsloth-published speedups, and tell the reader to leave ctx_type at
the default until those APIs land.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Final verification

**Files:**
- None modified.

- [ ] **Step 1: Confirm scope**

Run: `git log --oneline -4`

Expected: four new commits, one per task. Subjects start with `docs(readme)`, `docs(api)`, `docs(changelog)`, `docs(claude)` (the order depends on which task ran first; all four must be present).

- [ ] **Step 2: Confirm no source/test changes**

Run: `git diff HEAD~4 --stat`

Expected: only `README.md`, `docs/API.md`, `docs/CHANGELOG-2026-05-27.md`, and `CLAUDE.md` appear. No `src/`, no `tests/`, no `pyproject.toml`. (The pre-existing in-flight changes on the branch — listed in the session-start git status — are not touched by these four commits and remain `M` in `git status`.)

- [ ] **Step 3: Confirm tests still pass**

Run: `uv run pytest tests/test_mtp.py -q`

Expected: same result as before this plan. If the MTP-capable model is present at `./models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` (or `LLAMA_MTP_TEST_MODEL`), the full suite passes; otherwise the model-dependent tests skip cleanly. **Either outcome is acceptable** — this plan changes no code, so test results must match the pre-plan baseline. If the suite errors with anything new, stop and investigate.

- [ ] **Step 4: Confirm the warnings are visible**

Run: `grep -n "no throughput benefit\|net regression\|Scaffold only\|No acceleration consumer\|net throughput regression" README.md docs/API.md docs/CHANGELOG-2026-05-27.md CLAUDE.md`

Expected: at least one match per file. (`README.md` matches "no throughput benefit"; `docs/API.md` matches "Scaffold only" and "No acceleration consumer"; `docs/CHANGELOG-2026-05-27.md` matches "net throughput regression"; `CLAUDE.md` matches "Scaffold only" and "no throughput benefit".)

- [ ] **Step 5: No commit needed for this task**

Verification only.

---

## Self-review (run by the implementing agent before declaring done)

1. **Spec coverage.** The spec lists four files. Tasks 1–4 cover them, in the order spec → README, docs/API.md, docs/CHANGELOG-2026-05-27.md, CLAUDE.md. Task 5 verifies the spec's "Out of scope" guarantees (no source/test changes).
2. **Placeholder scan.** The plan contains no "TBD", "TODO", "implement later", or "similar to Task N". Every step has the actual text to insert.
3. **Type consistency.** Not applicable — no code in this plan. The five symbol names referenced verbatim across tasks (`LLAMA_CONTEXT_TYPE_MTP`, `LLAMA_CONTEXT_TYPE_DEFAULT`, `common_speculative_impl_draft_mtp`, `llama_set_embeddings_pre_norm`, `libllama-common`) match between tasks and match upstream.
4. **No emojis introduced.** Confirmed — none of the `new_string` blocks contain emojis.
