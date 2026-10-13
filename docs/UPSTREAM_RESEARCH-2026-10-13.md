# Upstream research — 2026-10-13

This update investigates **ggml-org/llama.cpp** changes since this binding's
September integration, concentrating on interfaces actually consumed by the
extension, MTP decoding, state compatibility, and practical performance work.
Claims below come from upstream source and commit history. Recommendations
are engineering judgments; performance gains have not been measured by this
research.

## Revisions and method

| Boundary | Revision and evidence |
| --- | --- |
| Binding baseline | Local commit `cd5936a0db4edcc299e9e03c6477949ce3e0268f`, September 15. Its [system-library documentation](SYSTEM_LIBRARIES.md) records validation against **b10972**, upstream `7cf1c54a96d4e950ffa614b94babf762803a8de7`. |
| Installed-source checkout | `/home/lianghong/Projects/llama_cpp_projects/llama.cpp` was clean at `23b0202a189c44a54625aadcb37a946dd1d6278d`, October 10, 19:25:38 UTC. Generated build info reports **b11552 / `23b0202`**; independently calling the installed `/usr/local/lib/libllama-common.so` build-info export returned **`b11552-23b0202`**. [Upstream commit][installed]. |
| Observed public HEAD | `git ls-remote` and GitHub's commit API returned `cbe28bb3cb0758f0c566a9a3561766821aeb0748`, October 12, 22:23:01 UTC, “model: fix vllm d3 multi seq case.” In Shanghai this is October 13, 06:23:01. [Commit][head]. |
| Comparison | Baseline to observed HEAD contains **605 commits**, with status `ahead` in GitHub's comparison API. All 605 entries were retrieved through pagination. [Exact comparison][comparison]. |
| Upstream version | Both the installed checkout and observed HEAD declare **0.6.0**, normally **0.6.0-dev** for development builds. This is separate from the binding's package version and upstream nightly build numbers. [Pinned CMake source][cmake]. |

The repository uses system libraries, with no vendored upstream dependency or
enforced exact commit pin. A documented tested revision establishes a baseline;
it does not guarantee compatibility with a later system installation.
See [system-library integration](SYSTEM_LIBRARIES.md).

Method: inspect the local binding and previous research; resolve the public
Git ref; retrieve paginated official history; compare `llama.h`,
`llama-ext.h`, `common.h`, and `speculative.h` at the baseline, installed
checkout, and observed HEAD; inspect relevant commit patches and current
implementations. Research did not rebuild or replace system llama.cpp.
Runtime checks for the accompanying implementation belong in its validation
notes, not in this source investigation.

The build-info check used `ctypes` on the C++ export
`_Z16llama_build_infov`; its unqualified C function name is not exported by
this Linux build. This checks reported revision identity, not every installed
header's provenance. [Build-info implementation][build-info].

## Immediate migration requirement

**The extension's old `common_speculative_process(spec, llama_batch)` call
must migrate to `common_batch`.** This is a C++ ABI change, not simply a
new optional API. The September source used the removed overload.
[Legacy overload removal][batch-removal], [current declaration][spec-header].

The evolution has three stages:

1. September 24, `fc343a84bbd925b37dde3219de35ea0bed50d630`,
   PR #24669: introduces the public `llama_batch_ext` API and
   `llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)`.
   The old `llama_decode` entry point remains as an adapter.
   [Introduction][batch-add].
2. September 28, `f1ea206218210afb913ae2f5d2c51faed35915da`,
   PR #29385: migrates speculative/common code to `common_batch`, temporarily
   retaining the legacy overload. [Migration][batch-migration].
3. September 30, `60e9cf7a7be81e4a1be7f15de6e8470d0e5b8b29`,
   PR #29601: removes the legacy overload. Rebuilding unchanged old binding
   source consequently fails; an already-built extension can instead fail
   when resolving the removed symbol. [Removal][batch-removal].

Recommended integration:

- Construct `common_batch(ctx_tgt)`, populate its tokens with explicit
  sequence IDs, positions, and output flags, and call `llama_process` on
  `batch.get()` before `common_speculative_process(spec, batch)`.
- Keep each submitted batch bounded by the configured target and draft
  capacities. `common_batch` can accumulate more than `n_batch`, but
  rendering an oversized sub-batch can abort when adding an entry fails.
- Reserve `batch.tokens` once and reuse it with `clear()` / `add()` through
  prefill and verification. Upstream's `clear()` clears its vector without
  shrinking capacity.
- Construct with a context: the default constructor has no owned extended
  batch, and `get_sub_batch()` asserts that one exists.
- Preserve non-owning embedding/state buffers until processing finishes.
  `common_batch::token` stores views; it does not copy those arrays.

These contracts are visible in [the wrapper declaration][common-header] and
[constructor, clearing, rendering, and processing implementation][common-code].
Normal token-only decode paths may continue using `llama_decode`; upstream
still provides the compatibility adapter. [Public API][header],
[context adapter][context].

The installed revision and observed HEAD have identical `speculative.h`
interfaces. This migration therefore addresses both investigated revisions.
[Installed header][installed-spec], [observed-HEAD header][spec-header].

## Build compatibility and state boundaries

The existing CMake feature check tests September-era members and sampler
signatures but does not call the newly required common batch interfaces.
Extend the configure-time compile/link probe to instantiate `common_batch`,
use `llama_process`, and pass that wrapper to
`common_speculative_process`. This catches missing matching headers or
symbols before the extension is built. See the current
[common interface][spec-header] and [public batch API][header].

Current upstream installs **libllama-common** when `LLAMA_BUILD_COMMON` is
enabled, but its public-header installation lists only `llama.h` and
`llama-cpp.h`. `common.h`, `speculative.h`, their transitive dependencies,
and `src/llama-ext.h` still need to be supplied explicitly. Plain
`cmake --install` does not supply every header required by this project.
[Installation targets][cmake], [common dependencies][common-header].

For a source build installed to a writable custom prefix, run the usual
upstream install first, then add the currently consumed common/staging
headers from the **same checkout**:

```bash
rtk proxy cmake --install build --prefix /opt/llama
rtk proxy install -d /opt/llama/include/llama
rtk proxy install -m 644 common/common.h common/speculative.h /opt/llama/include/llama/
rtk proxy install -m 644 src/llama-ext.h /opt/llama/include/
```

Build upstream with `BUILD_SHARED_LIBS=ON` and `LLAMA_BUILD_COMMON=ON`.
The two copied common headers require `llama-cpp.h`, `llama.h`, `ggml.h`,
and `ggml-opt.h`, which the normal upstream public-header installation
supplies. Use another writable prefix when `/opt/llama` is not available;
system installation may require administrator privileges.
[Common includes][common-header], [speculative includes][spec-header],
[llama installation][cmake], [ggml header installation][ggml-cmake].

Rebuild the binding whenever these common/staging interfaces change, using
headers and libraries from one upstream build. `llama-ext.h` explicitly
labels its contents experimental and permits breaking changes.
[Staging contract][ext]. The observed HEAD additionally adds fields to
`llama_perf_context_data`; compiling against new struct layouts and loading
older libraries remains unsafe even when names or SONAMEs match.
[Performance ABI change][perf].

October 5, `210791069bf5460bef3c09b982ee6f0aa9967404`, PR #28498,
adds exact KV rotation metadata and changes **session version 10 → 11**
and **sequence-state version 3 → 4**. Treat persisted snapshots from older
upstream builds as incompatible and regenerate them after an upgrade.
This includes raw state buffers, whose KV payload layout also changes.
In-session device handles additionally retain this binding's existing
context-owner and epoch restrictions. [State-format change][state-format],
[binding state documentation](API.md).

September 26, PR #27530, also clears affected KV/recurrent state and
discards deferred copies after failed restores. Keeping this upstream fix
is necessary; Python position bookkeeping alone cannot undo partial
native restoration. [Failed-restore fix][restore-fix].

## MTP behavior to preserve

Upstream's October 10 MTP change separates a token embedding from a carried
**state embedding**. Draft batches attach target hidden rows with
`set_embd_state()`, permitting vision token embeddings to remain separate.
Text-only callers that delegate to the common MTP driver inherit this
behavior without implementing their own hidden-state feed.
[MTP embedding fix][mtp-state], [current driver][spec-code].

This also makes a standalone MTP context unsuitable for the binding's legacy
token-only `decode()` interface: `llm_graph_input_embd_h::set_input()` asserts
that `ubatch->embd_state` exists. Without that target-derived state, the
native call can abort the process. Reject such calls before invoking upstream;
the accompanying binding update adds `ValueError` / `ValidationError` guards.
Use the supported speculative driver, which attaches state rows through
`set_embd_state()`. These requirements follow from the source; runtime
validation of the guards is recorded separately in implementation notes.
[Required graph input][graph-state], [driver state input][spec-code].

The driver still uses output embedding width, carries the final teacher-forced
hidden row across batches, and updates carryover to the accepted verification
row. It handles shared target KV and multiple trained heads internally.
Preserve the target prefill/process hook on every bounded batch rather than
reintroducing a custom draft loop. [Current MTP implementation][spec-code].

October 2, PR #27694, adds `result_q`, `temp`, and `seed` to draft parameters.
Providing candidate distributions enables probabilistic draft sampling and
rejection verification. Leaving `result_q` null keeps sample-and-match
behavior; enabling probabilistic drafting requires the corresponding target
accept/reject algorithm and RNG handling. Merely filling the pointer does
not complete that algorithm. [Interface change][probabilistic],
[draft implementation][spec-code].

The observed HEAD also contains October 12 example rejection/sampling fixes
after the installed revision. Treat full probabilistic speculation as a
separate feature with distribution-level validation rather than a necessary
compatibility change. [Example fixes][rejection-fix].

## Performance opportunities

| Priority | Upstream evidence | Recommended action |
| --- | --- | --- |
| Immediate | October 7, PR #29797, selects a greedy terminal sampler for eligible zero-temperature chains, retaining distribution sampling when probabilities are requested. [Change][greedy]. | Adopt the same condition for fast deterministic generation. Preserve logprob computation and adaptive-p behavior; verify identical selected tokens, including ties and grammar constraints. |
| Immediate | September 16, PR #28549, gives output and no-output MTP batches distinct graph arenas for CUDA capture. [Change][cuda-mtp]. | Obtain through the matching upstream runtime and reuse draft contexts/drivers across continuations. Benchmark steady-state decoding after warmup. |
| Investigate | October 6, PR #30020, marks the scheduler for reservation whenever NextN extraction flags change. [Change][nextn-reserve]. | Measure repeated speculative requests and transitions. Keeping extraction flags stable across reusable speculative turns may avoid allocations, but ordinary generation and output lifetimes must stay correct. |
| Optional | October 7–8, PRs #29887 and #30112, introduce `llama_context_params.moe_cache_size`, a device cache for experts kept in host memory, including multiple GPUs. [Initial cache][moe], [multi-GPU support][moe-multi]. | Expose a byte-sized opt-in config for CPU-offloaded MoE workloads. Preserve default zero; mirror upstream's zero cache on derived draft contexts. |
| Backend update | October 8, PR #30087, changes CUDA GDN state processing; October 9, PR #30223, keeps backend sampling graphs static across microbatches. [GDN][gdn], [sampling graph][sampling-graph]. | Inherit the kernels/runtime through upstream. Backend target sampling still requires correct output budgets, sampler rollback, and feature fallback. |
| Future instrumentation | October 12, PR #30329, adds `n_reallocs_same` and `n_reallocs_diff` to performance data. [Change][perf]. | Feature-detect before exposing them: installed b11552 lacks these fields. Use them when validating graph stability against later builds. |

The binding's own aggressive compiler flags do not rebuild or optimize its
external ggml kernels. Measure binding overhead separately from upstream
prompt processing, target verification, draft generation, and sampling.
Upstream builds the common and ggml libraries as separate targets.
[Build configuration][cmake], [common target][common-cmake].

None of these source changes establishes a speedup on this machine. Report
measured tokens per second, prefill latency, acceptance rate, and model/backend
settings instead of adding unrelated upstream benchmark percentages.

### MoE cache placement and capacity

`moe_cache_size` is a budget, not a request to change model placement.
Upstream enables useful caching only when all of the following hold:

- The model contains MoE experts and at least one active expert per token.
- A GPU or integrated GPU backend exists; tensor parallel split mode is
  currently unsupported.
- An eligible layer is assigned to that GPU, while all of that layer's expert
  tensors reside in host weight buffers.
- The cache budget provides enough slots for the active experts of a token,
  after splitting the budget among devices and expert-layout groups.

When no eligible layer exists, upstream logs that the cache is disabled.
It skips groups with insufficient slots and throws if no group can hold the
experts of one token. Cache allocation also accounts for tensor alignment and
one extra expert slot. [Pinned cache implementation][moe-code].

`n_gpu_layers` alone does not keep expert tensors in host buffers for layers
assigned to a GPU. Upstream's `--cpu-moe` explicitly applies
`llm_ffn_exps_cpu_override()`, which maps expert-weight tensor names to the CPU
buffer type while preserving separate layer placement. The accompanying
binding exposes the same helper through `cpu_moe=True`; combine it with
GPU-assigned layers and a nonzero `moe_cache_size` when evaluating the cache.
[Tensor-placement helper][cpu-moe-helper], [official argument handling][arg].

Benchmark an actual eligible MoE model and inspect placement/cache logs.
Successful dense-model inference or successful construction with the default
zero budget does not validate the caching path. These are integration
requirements from upstream source, not a measured speedup or a claim that this
research ran an MoE benchmark. [Cache selection and initialization][moe-code].

## Validation recommended for this update

The following checks follow from the changed contracts above:

- Build and import against installed b11552 with a fresh extension.
- Compare greedy ordinary and MTP generation using the same model and build,
  including a prompt longer than the configured batch and a request that
  exhausts its token budget in an accepted draft.
- Exercise cached speculative continuations, speculative/ordinary mode
  transitions, multi-token stops, cancellation, and grammar.
- Verify same-build state save/restore and malformed-state recovery; document
  the upstream state-version boundary.
- Confirm unsupported installations fail the CMake interface probe.
- Benchmark only after correctness checks, distinguishing warmup and repeated
  requests from model loading.

These are proposed acceptance criteria, not a claim that research executed
them. See [previous upstream research](UPSTREAM_RESEARCH-2026-09-15.md) for
the earlier rollback and MTP contracts that still apply.

[installed]: https://github.com/ggml-org/llama.cpp/commit/23b0202a189c44a54625aadcb37a946dd1d6278d
[head]: https://github.com/ggml-org/llama.cpp/commit/cbe28bb3cb0758f0c566a9a3561766821aeb0748
[comparison]: https://github.com/ggml-org/llama.cpp/compare/7cf1c54a96d4e950ffa614b94babf762803a8de7...cbe28bb3cb0758f0c566a9a3561766821aeb0748
[cmake]: https://github.com/ggml-org/llama.cpp/blob/cbe28bb3cb0758f0c566a9a3561766821aeb0748/CMakeLists.txt
[common-cmake]: https://github.com/ggml-org/llama.cpp/blob/cbe28bb3cb0758f0c566a9a3561766821aeb0748/common/CMakeLists.txt
[ggml-cmake]: https://github.com/ggml-org/llama.cpp/blob/cbe28bb3cb0758f0c566a9a3561766821aeb0748/ggml/CMakeLists.txt
[build-info]: https://github.com/ggml-org/llama.cpp/blob/cbe28bb3cb0758f0c566a9a3561766821aeb0748/common/build-info.cpp.in
[header]: https://github.com/ggml-org/llama.cpp/blob/cbe28bb3cb0758f0c566a9a3561766821aeb0748/include/llama.h
[ext]: https://github.com/ggml-org/llama.cpp/blob/cbe28bb3cb0758f0c566a9a3561766821aeb0748/src/llama-ext.h
[spec-header]: https://github.com/ggml-org/llama.cpp/blob/cbe28bb3cb0758f0c566a9a3561766821aeb0748/common/speculative.h
[installed-spec]: https://github.com/ggml-org/llama.cpp/blob/23b0202a189c44a54625aadcb37a946dd1d6278d/common/speculative.h
[common-header]: https://github.com/ggml-org/llama.cpp/blob/cbe28bb3cb0758f0c566a9a3561766821aeb0748/common/common.h
[cpu-moe-helper]: https://github.com/ggml-org/llama.cpp/blob/cbe28bb3cb0758f0c566a9a3561766821aeb0748/common/common.h#L1239-L1249
[arg]: https://github.com/ggml-org/llama.cpp/blob/cbe28bb3cb0758f0c566a9a3561766821aeb0748/common/arg.cpp#L2762-L2785
[common-code]: https://github.com/ggml-org/llama.cpp/blob/cbe28bb3cb0758f0c566a9a3561766821aeb0748/common/common.cpp
[context]: https://github.com/ggml-org/llama.cpp/blob/cbe28bb3cb0758f0c566a9a3561766821aeb0748/src/llama-context.cpp
[spec-code]: https://github.com/ggml-org/llama.cpp/blob/cbe28bb3cb0758f0c566a9a3561766821aeb0748/common/speculative.cpp
[graph-state]: https://github.com/ggml-org/llama.cpp/blob/cbe28bb3cb0758f0c566a9a3561766821aeb0748/src/llama-graph.cpp#L140-L158
[moe-code]: https://github.com/ggml-org/llama.cpp/blob/cbe28bb3cb0758f0c566a9a3561766821aeb0748/src/llama-moe-cache.cpp#L221-L332
[batch-add]: https://github.com/ggml-org/llama.cpp/commit/fc343a84bbd925b37dde3219de35ea0bed50d630
[batch-migration]: https://github.com/ggml-org/llama.cpp/commit/f1ea206218210afb913ae2f5d2c51faed35915da
[batch-removal]: https://github.com/ggml-org/llama.cpp/commit/60e9cf7a7be81e4a1be7f15de6e8470d0e5b8b29
[state-format]: https://github.com/ggml-org/llama.cpp/commit/210791069bf5460bef3c09b982ee6f0aa9967404
[restore-fix]: https://github.com/ggml-org/llama.cpp/commit/08618ff8e735141d8e4e5be28e6d6af170e4757b
[mtp-state]: https://github.com/ggml-org/llama.cpp/commit/781dbc5ac98921dbdb5e5b2ec5b7a50960e937d4
[probabilistic]: https://github.com/ggml-org/llama.cpp/commit/1fb7ef3e3327f18f1e99d294115b8493d54e196a
[rejection-fix]: https://github.com/ggml-org/llama.cpp/commit/77757a34672a888845a09ab45728b8d34aa475ff
[greedy]: https://github.com/ggml-org/llama.cpp/commit/d0b490f25edec147b4dcd69cce31c7d1099dff1d
[cuda-mtp]: https://github.com/ggml-org/llama.cpp/commit/2f3fd02526682adbd3ba771d929d271e477a35c5
[nextn-reserve]: https://github.com/ggml-org/llama.cpp/commit/1a3011cc0c183f184849b5fa4fa15ab0399f17d8
[moe]: https://github.com/ggml-org/llama.cpp/commit/d6cf9acb25e6c657d6c6a4645ab7b69fce119995
[moe-multi]: https://github.com/ggml-org/llama.cpp/commit/c811cb8f0ac91b8ac72a32f970bdd45037f20da7
[gdn]: https://github.com/ggml-org/llama.cpp/commit/24e41838e0c834f67db214e6d280af51f266056e
[sampling-graph]: https://github.com/ggml-org/llama.cpp/commit/a518119d30cade6260f7494120863428a3fe8ee5
[perf]: https://github.com/ggml-org/llama.cpp/commit/972b5fcc19670e597f8de9bf5cacfb85752a5fff
