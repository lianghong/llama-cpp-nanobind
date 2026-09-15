# Upstream integration and MTP update — 2026-09-15

The bindings now build against llama.cpp **b10972 / `7cf1c54a9`**.
The previous b9592 integration failed to compile against current headers.
The [upstream research](UPSTREAM_RESEARCH-2026-09-15.md) covers 1,381 commits
through `1bc7a5af…`, whose only change beyond the installed revision is web UI
code. Installed upstream libraries were not modified.

## Compatibility

- Translate the existing `use_mmap`/`use_mlock` switches to all four
  corresponding upstream `load_mode` values.
- Add `LlamaConfig.load_mtp=True`. Upstream now defaults to skipping MTP
  tensors; this wrapper retains them by default so speculation can still be
  enabled after model loading. Set `False` to save their weight memory.
- Use `llama_model_n_layer_nextn` instead of parsing metadata. This is a
  **declared** count, so capability checks also retain the weight-loading
  decision and check draft-context creation.
- Update penalty and DRY constructor signatures. Resolve Python's
  `-1 = full context` history windows before passing them to upstream, which
  now clamps negative windows to zero.
- Apply the model's suppression-token list alongside explicit logit biases.
- `UnifiedLLM` sizes target rollback snapshots to its selected draft width.
  Explicit `speculative=False` skips MTP weights and recurrent snapshots.
- CMake checks the required interfaces and respects explicit installation
  prefixes. Headers, libllama, ggml, and libllama-common must match; rebuild
  the extension after upgrading upstream. Older b9592 libraries are not
  supported by this source revision.

## MTP execution and state

The first draft starts with a **sampled target token** at its actual `pos0`.
The old loop started with the last prompt token and used the removed
`n_past` field. The current loop follows upstream's hidden-state pairing and
selects the accepted verification row through `common_speculative_accept`.

`Context` owns the common speculative driver as well as both contexts.
This retains hidden-state carryover and attached draft samplers across
unchanged-prefix continuations. External KV/state mutations and changing
draft width invalidate the driver; the next speculative generation rebuilds
from its complete prompt.

Normal exits decode the final sampled token exactly once. Returned tokens,
KV position, target logits, and pending hidden state are aligned. This
supersedes the old wrapper's deliberately undecoded final token and its
mode-switch repair. An ordinary continuation can reuse the aligned prefix.
Switching into speculative mode rebuilds missing draft state.

Stops and callback cancellation clear uncertain KV rather than retaining a
partially rolled-back hidden state. The Python mirror and on-device snapshot
epoch are invalidated accordingly. Streaming buffers possible stop prefixes,
so matched multi-token stops never reach the consumer.

## Batching and validation

- Prefill is chunked within both contexts' logical batch limits and one
  target micro-batch; each decode immediately feeds its hidden rows to MTP.
- One batch allocation serves prompt chunks, verification rounds, and the
  final decode. Draft and verification token buffers are reused.
- Drafting is bounded by output budget and context space. Near the context
  end, verification falls back to single-token steps because upstream applies
  its per-round `n_max` limit only after drafting.
- Target recurrent snapshots must cover `n_draft_max`; recurrent
  `n_ubatch <= n_rs_seq + 1` is rejected before upstream can assert.
  Draft contexts use `n_rs_seq=0`, resolved target context size, no pooling,
  and `ctx_other` following upstream initialization.
- Every nonzero decode result is treated as failure. Ordinary prompt decode
  also chunks inputs exceeding `n_batch`.
- Boolean, fractional, and string draft-width overrides are rejected rather
  than silently coerced.

## Scope

Live MTP validation uses a **Qwen3.5-4B-Q4_K_M-MTP** checkpoint on an
**NVIDIA RTX 3080**, with matching ordinary Qwen3.5-4B tests. The repository's
default 35B MTP fixture is absent; tests use `LLAMA_MTP_TEST_MODEL`.

The interface supports embedded MTP layers in the loaded model. Separate
assistant/draft GGUF loading, EAGLE3, DFlash, and DSpark are not added.
Step multiple-head, DeepSeek width variants, shared-KV assistants, and
other GPU backends have not received live model validation here.
Target verification retains the existing CPU sampler/grammar path;
backend draft sampling remains managed by upstream with its fallback.

Greedy differential tests compare ordinary and speculative decoding within
the same build. They do not guarantee bit-for-bit equality across upstream
versions, hardware, or floating-point ties. State files are also not promised
portable across the upstream state-format changes.

The benchmark now accepts model, draft-width, run-count, and token-budget
arguments. It retains warmed contexts, counts actual generated token IDs,
and reports end-to-end throughput and greedy-output agreement. It no longer
treats a 1.10× speedup as a universal acceptance threshold.

## Validation and measurement

- Editable Release extension built against the installed b10972 headers and
  libraries; CMake's interface check passed.
- Full suite: **322 passed, 1 failed**. The new invalid-rollback test exposed
  missing cleanup for a native `ValueError` during context initialization.
  After fixing that wrapper path, **all 60 focused MTP, validation,
  partial-initialization, and on-device-state tests passed**.
- After updating `UnifiedLLM` loading and rollback allocation, **all 50
  UnifiedLLM tests passed**, including new live draft-width-4 and width-8
  generation cases. Together these runs cover all 325 tests now present;
  the full suite was not rerun after the focused fixes.
- Ruff, mypy (all five Python source modules), Python 3.14 warning-as-error
  compilation, and diff whitespace checks passed. clang-tidy completed with
  four existing style warnings in unchanged state-restoration code; no
  compiler errors or new warnings remained.

Benchmark command:

```bash
python examples/bench_speculative.py \
  --model /path/to/Qwen3.5-4B-Q4_K_M-MTP.gguf \
  --max-tokens 128 --runs 3 --draft-max 1 2 4
```

RTX 3080, all model layers on GPU, `n_ctx=2048`, `n_rs_seq=4` on both
ordinary and speculative runs, greedy sampling, warmed contexts, median of
three runs. Timing includes prompt processing and generation; these compare
the two modes in the updated binding, not speed against the unbuildable
previous binding.

| Mode | Tokens/s | Speedup | Greedy output matches ordinary |
| --- | ---: | ---: | --- |
| Ordinary | 92.6 | 1.00× | — |
| MTP, draft width 1 | 108.7 | 1.17× | Yes |
| MTP, draft width 2 | **110.3** | **1.19×** | **Yes** |
| MTP, draft width 4 | 106.5 | 1.15× | No |

Width 4 first differed at generated token index 88 (`" each"` versus
`" different"`). The same rank reversal was reproduced by **ordinary
target-only replay of the identical token prefix in batches of five**, with
no draft context or rollback:

| Evaluation of that prefix | Logit for `" each"` | Logit for `" different"` |
| --- | ---: | ---: |
| Ordinary single-token replay | 19.380348 | 19.189846 |
| Ordinary five-token replay | 19.256237 | 19.293266 |
| MTP width-4 verification | 19.223820 | 19.235931 |

Neither candidate was penalized by the last-64-token repetition window.
This demonstrates batching sensitivity independently of MTP state handling;
it does not establish bitwise equivalence for arbitrary prompts. Width 2
remains the default and is the best measured choice for this fixture.
Temporary diagnostic instrumentation was removed after this investigation.
