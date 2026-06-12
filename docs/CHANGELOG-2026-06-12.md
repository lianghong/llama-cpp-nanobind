# Changelog — 2026-06-12

**Focus:** restore the build against current llama.cpp, fix a spec→nonspec
continuation divergence, and apply code-review findings (probe cost, error
attribution, doc accuracy). No breaking API changes.

---

## Build fix

### Upstream renamed `llama_set_embeddings_pre_norm` → `llama_set_embeddings_nextn`

llama.cpp commit `166fe294` ("qwen35: use post-norm hidden state for MTP",
b9496+, 2026-06-04) renamed the staging API the speculative loop's RAII guard
calls. The bindings failed to compile against current headers, and previously
built extensions failed to import (`undefined symbol:
llama_set_embeddings_pre_norm`). The guard (`PreNormGuard` →
`NextnEmbdGuard`) now calls the new name; signature unchanged.

## Bug fix

### spec→nonspec continuation diverged when `max_tokens` ran out on an accepted draft

`_guard_speculative_mode_switch` healed only the off-by-1 exit shape (KV one
position behind the mirror). When a speculative turn exhausts `max_tokens`
exactly on an *accepted draft*, KV exits **aligned** with the mirror — but the
logits buffer still holds the last verify batch's final *drafted* position,
which is stale whenever any draft was rejected that round. A
`reset_kv_cache=False` non-speculative continuation then decodes nothing
(full LCP) and samples its first token from those stale logits → divergent
output (caught by
`test_speculative_mtp.py::test_speculative_continuation_modes`,
spec=True→False).

The guard now handles both exit shapes in place: KV-behind decodes the
undecoded tail (as before); KV-aligned trims the final token via
`kv_cache_seq_rm` and re-decodes it to refresh logits (bumping the on-device
state epoch), falling back to a full reset when hybrid memory refuses the
trim. Prefix reuse is preserved on both paths.

## Probe correctness & cost (code review)

- **Narrative correction.** The 2026-06-03 claim that "llama.cpp allocates a
  degenerate MTP context for any qwen35-arch model" is true only for
  **pre-b9180** builds; b9180+ (upstream `25558268`, 2026-05-16) rejects the
  allocation when `hparams.n_layer_nextn == 0` (verified empirically against
  all installed libs). Comments/changelogs now say "pre-b9180"; the metadata
  gate remains (correct on all lib versions, and avoids allocating a
  throwaway draft context for a capability query).
- `mtp_predict_layers()` is **memoized** (the value is immutable per model;
  the probe runs on every speculative call) and now **warns on stderr** when
  the metadata key exists but fails to parse, instead of silently disabling
  speculative.
- `supports_speculative_mtp` is bound with
  `nb::call_guard<nb::gil_scoped_release>` — its first call on an MTP model
  allocates the draft context and previously held the GIL throughout.
- `Llama` gains public `supports_speculative_mtp()` / `mtp_predict_layers()`
  wrappers (with `_check_closed`), matching what `docs/API.md` documents —
  previously the methods existed only on the inner `Context`.
- `Llama._validate_speculative` distinguishes "no MTP metadata" from
  "MTP-capable but draft-context allocation failed (likely OOM)" in its
  `ValidationError` message.
- `UnifiedLLM(speculative=False)` no longer probes (the probe's success path
  allocates the draft context that a `False` caller never uses);
  `create_chat_completion` checks `_check_closed()` before
  `_validate_speculative` so a closed instance raises `LlamaError`, not a
  misleading `ValidationError`.

## Tests & docs

- `MTP_MODEL_PATH` / `requires_mtp_model` hoisted into `tests/conftest.py`
  (was copy-pasted in 4 test files).
- The positive probe test loads the 35B checkpoint with `n_gpu_layers=-1`
  (was `0` — full CPU residency for an 18 GB model).
- `docs/API.md`: UnifiedLLM constructor now documents `speculative` /
  `n_draft_max` and the `speculative_enabled` property; exception types
  corrected (`ValueError` from `UnifiedLLM`, `ValidationError` from `Llama`).

Verified: 270/270 (`tests/`, minus the long-running streaming/double-free
files) against libllama b9592 on live 4B / 35B-MoE models.
