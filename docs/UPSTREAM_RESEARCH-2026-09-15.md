# Upstream research — 2026-09-15

Scope: changes in **ggml-org/llama.cpp** since this repository's last documented upstream update, emphasizing MTP/speculative decoding, correctness, performance, and public interfaces consumed by bindings. This report gives integration recommendations, not an audit of local implementation. No local build, inference, or tests were run for this research; the installed upstream was not modified.

## Baseline and research boundary

| Item | Exact revision / evidence |
| --- | --- |
| Local update | `5ad73e8490f690ae8c6ba3203dacdd6ca381f348`, dated 2026-06-12. Its [changelog][local-changelog] reports testing against **libllama b9592**. |
| Upstream baseline | Tag **b9592** resolves to **`ac4cddeb0dbd778f650bf568f6f08344a06abe3a`**, committed **2026-06-10 20:28:03 UTC**: “vendor : update LibreSSL to 4.3.2” ([commit][baseline], [tag API][baseline-tag]). |
| Reachable upstream head | Public Git `refs/heads/master` and GitHub's commits API both returned **`1bc7a5af0d14b1fb72f266abbd1237b394187115`**, committed **2026-09-14 23:11:26 UTC**: “webui: stop re-probing disabled /tools endpoint on every message” ([commit][head], [revision API][head-api]). In Asia/Shanghai, this is September 15 at 07:11:26. |
| Comparison | `ac4cddeb0dbd778f650bf568f6f08344a06abe3a..1bc7a5af0d14b1fb72f266abbd1237b394187115`: **1,381 commits**, with baseline ancestry verified in fetched Git history. [Exact comparison][comparison]. |
| Version at head | CMake declares **0.4.1**; development builds default to **0.4.1-dev**. This is distinct from a commit or nightly build number. [Pinned CMake source][head-cmake]. |

The baseline is the changelog's **tested upstream revision**, not a vendored dependency pin or a claim about today's installed library. The older rename commit cited there is **`166fe29492abb4093ec889b5c6f6fdb4e3b8ba98`**, committed June 3 at 17:29:09 UTC / June 4 in Shanghai. Its `pre_norm` → `nextn` rename already preceded b9592; it is **not a new migration requirement**. [Rename commit][rename].

During research, the implementation owner reported that installed libraries correspond to local upstream source `/home/lianghong/Projects/llama_cpp_projects/llama.cpp` at **`7cf1c54a96d4e950ffa614b94babf762803a8de7`**, September 14, with local `build-info.cpp` reporting **b10972 / `7cf1c54a9`**. This is the direct parent of the observed head, whose only changes are three web UI files. Consequently, the core/speculative/header findings below also apply to that parent revision. The installation-to-source correspondence is owner-provided, not independently audited here. [Head commit, parent and changed files][head-api].

Method: read local update history and the changelog; resolve public upstream refs using Git and GitHub's API; fetch history into a separate temporary bare repository; inspect the complete baseline-to-head diffs of `include/llama.h`, `src/llama-ext.h`, and `common/speculative.h`; inspect relevant commits, merged PR descriptions, and pinned implementation source. The report freezes the observed head instead of inferring a release from the calendar. It covers merged changes reachable from that head, not unmerged proposals. PR performance figures are attributed measurements, not measurements of this binding.

## Highest-priority integration decisions

1. **Choose MTP weight loading before model creation.** New upstream defaults skip MTP tensors. Positive metadata afterward does not make skipped tensors available. Preserve any intended ability to enable speculation later through an explicit load policy. [MTP loading change][mtp-load].
2. **Adapt the public API before evaluating runtime behavior.** Model-loading booleans were replaced with `load_mode`; penalty and DRY sampler signatures changed; context/sampler struct layouts changed. Rebuild against matching headers and libraries. [Loading change][load-mode], [penalties][penalty-vocab], [history samplers][history-samplers], [multi-output sampling][multi-output].
3. **Treat rollback, hidden carryover, and sampler state as one transaction.** Current upstream still needs bounded rollback and now makes pending recurrent rollback explicitly single-use. Backend sampling rollback copies state into the existing attached sampler. [Rollback fix][rollback-multiseq], [server verification][head-server], [sampler copy][head-sampling].
4. **Use output embedding width and distinguish MTP modes.** Current upstream uses `llama_model_n_embd_out`, supports trained-head chaining, and separately handles shared-KV assistants. One repeated-head loop does not implement all three. [DeepSeek V4 change][dsv4-mtp], [multi-head change][multi-head], [current driver][head-spec].
5. **Validate backend sampling separately.** Target verification can now use it, but needs sufficient per-sequence outputs, contiguous acceptance order, compatible samplers, and CPU fallback. [Merged PR][multi-output-pr], [public header][head-header], [sampling implementation][head-sampling].

## MTP interfaces and behavioral contracts

### Weight loading and capability detection

**July 31 — `82dbc4f017a7b005f993ac2e7af9c048ad686c04`, PR #26296:** adds `llama_model_params.load_mtp`, defaults it to `false`, and skips unused MTP tensors, including Qwen3.5/Qwen3.5-MoE. `common_model_params_to_llama` enables it when `draft-mtp` is selected. This saves memory for ordinary inference but changes what default-loaded models can do. [Commit][mtp-load], [PR][mtp-load-pr].

Recommendation: track separate facts for **checkpoint metadata**, **whether MTP weights were requested**, and **whether a usable draft context exists**. To permit enabling MTP after loading, either load its weights up front or require an explicit reload when they were omitted. The inspected public API has no call that retroactively loads skipped MTP tensors. Avoid probing by speculative context allocation when weights were deliberately skipped. [Current model defaults][head-model], [header][head-header].

`llama_model_n_layer_nextn(model)` is now a public typed query of the **declared count stored in model hyperparameters**, introduced June 21. It is **not a loaded-layer count**: Qwen loaded with `load_mtp=false` can still return a positive count after skipping MTP tensors. Prefer the getter over parsing metadata strings on supported versions, retaining metadata fallback only if older versions remain supported. It is **not a universal capability predicate**: current context initialization rejects MTP when the count is zero **or** `router_layer >= 0`, because some architectures repurpose NextN count for a router. Track the load decision separately and gate draft creation/probing on it. A failed context allocation must not automatically be diagnosed as OOM. [Multi-head commit][multi-head], [tensor-skipping change][mtp-load], [context initialization][head-context], [accessor][head-model].

**August 13 — `1d2869c6e54d5003f3927a79efbca0fefa034a6d`, PR #27005:** adds draft-GGUF MTP detection. The current detector checks for the last block's `nextn.eh_proj.weight`; DFlash/DSpark use different tensor markers. File structure matters, but this `common/` helper is not a stable exported `libllama` capability API. [Commit][mtp-detect], [detector][head-spec].

### Hidden states, row width, and carryover

The existing `llama_set_embeddings_nextn`, `llama_get_embeddings_nextn`, and `llama_get_embeddings_nextn_ith` names remain. They are still in **`src/llama-ext.h`**, whose explicit contract permits breaking changes and C++ and calls everything in it WIP. These are not stable public `llama.h` ABI. [Pinned staging header][head-ext].

Current MTP allocates rows using **`llama_model_n_embd_out`** on target and draft, asserts matching output widths, and uses output width for MTP input embeddings and NextN output reordering. DeepSeek V4 provides a concrete reason ordinary embedding width cannot be assumed to equal NextN width. Use output width consistently for allocations, strides, bounds, and copies; validate target/draft compatibility. [August 2 commit `596a5795bdd6da317ea103fc06c0a71c296e3669`, #25784][dsv4-mtp], [context source][head-context], [MTP driver][head-spec].

These are **current invariants to preserve**, not all new since b9592:

- Target NextN extraction is **unmasked**, covering every token even when ordinary logits are selected sparsely. Draft NextN extraction is **masked** to requested outputs.
- MTP batches carry **both token IDs and hidden embeddings**. The driver compensates for `llama_batch_init` allocating only one array.
- Teacher-forced catch-up pairs `x[p+1]` with target `h[p]`. The last hidden row crosses the batch boundary through `pending_h`.
- After verification, carryover is **`verify_h[n_accepted]`**, where `n_accepted` excludes the final replacement/bonus token. Unconditionally retaining the final verification row would preserve a rejected suffix.
- The current driver assumes one sequence per token and contiguous rows for each sequence. It skips batches containing input embeddings and leaves a TODO for vision-token support.

Source: constructor, `process`, `draft`, and `accept` in the [pinned driver][head-spec]. Preserve output-buffer lifetimes across decode calls: the exported NextN getters synchronize before returning internal context storage. [Accessor implementation][head-context].

### Single head, chained heads, and shared KV

**June 21 — `d789527482d925156d7c4adfecebf5fb8481e0ee`, PR #24340:** supports Step3.5/3.7 multiple trained heads, adding public `llama_model_n_layer_nextn` and staging `llama_set_nextn_layer_offset`. Graph reuse now checks the head offset and NextN extraction flags. [Commit][multi-head], [PR data-flow trace][multi-head-pr].

| Current mode | Required behavior | Compatibility constraint |
| --- | --- | --- |
| Single trained head, e.g. Qwen3.5 | Reuse the head autoregressively; draft positions advance. | A layer count of one is **not** a draft-length limit of one. |
| Multiple trained heads, e.g. Step3.5 | Prime every head with teacher-forced target features; select head `i`; rebuild the growing draft prefix under it; restore offset zero afterward. | Cap effective draft length by trained-head count for this mode. Switching heads and decoding only the newest token leaves required per-layer KV missing. |
| Shared target KV, e.g. Gemma4 assistants | Detect `llama_get_ctx_other(ctx_dft) == ctx_tgt`; skip independent catch-up; reuse the same position for draft steps. | Do not trim shared memory as though it belonged only to the draft. This path already existed at baseline. |

Source: [current driver][head-spec], [multi-head diff][multi-head]. Step's loader now requires every declared MTP block on that path; previously pruned files that still claim the full head count need corrected metadata or compatible exports. [Loader change][multi-head].

Recommendation: declare supported model families and execution modes explicitly. Expand beyond existing Qwen use cases only after implementing the corresponding width, memory, position, and head-selection contracts.

### Context configuration and allocation

**August 22 — `2c6b141efb3b0868fd39d3cae73f69606e1d654c`, PR #27400:** draft parameters no longer inherit target embedding/pooling requests. Upstream sets `embedding=false` and `pooling_type=LLAMA_POOLING_TYPE_UNSPECIFIED`; NextN extraction remains separately enabled. Mirror this separation to avoid the reported null-tensor pooling failure. [Commit][mtp-pooling], [PR][mtp-pooling-pr].

Current draft initialization uses resolved target context size, `ctx_other=ctx_tgt`, and `n_rs_seq=0`. Backend MTP drafting normally needs one output per sequence per step; block drafters need different budgets. Preserve these deliberate differences when deriving draft parameters. [Initialization and output limits][head-spec].

**September 1 — `9d817213a0975020775efe6c458822616826f376`, #28159:** loads `n_layer_nextn` before calculations using `n_layer()`. **September 11 — `5cdd3d1dad5cbb7107b3e9f6d23239ba88ac0123`, #28630:** generalizes filtering of trunk versus NextN KV layers, fixing allocation for DeepSeek2, GLM4-MoE, and others while excluding router-only/all-NextN special cases. Include both before widening model support. [Layer-count fix][nextn-load-order], [KV fix][mtp-kv].

## Correctness and performance changes worth adopting

| Date / exact change | Source finding | Integration consequence |
| --- | --- | --- |
| July 3, `5a460dea9f961cdb508d58a6e7b0f9e259b4c19f`, #23940 | CUDA writes GDN snapshots directly into recurrent cache when safe, eliminating intermediate copies. | Relevant to Qwen hybrid verification. Obtain through the backend update; no duplicate binding kernel is needed. [Commit][gdn-copies]. |
| July 8, `230ea9d214320c5e79cc8166ed708ac60514c71e`, #25278 | Recurrent equal splitting preserves the last `1 + n_rs_seq` tokens together to keep rollback snapshots valid. | Size batches/microbatches coherently. This protected-tail path asserts `n_ubatch > n_keep_tail`; arbitrary large snapshot settings are inappropriate. [Commit][recurrent-split]. |
| July 16, `32e789fdfd598e9a1872da55ac941e4d94f030bd`, #25758 | Starts actually exercising recurrent rollback with a small generated Qwen3.5 model. | Prior test existence alone did not prove coverage ran. [Commit][rollback-tests], [PR][rollback-tests-pr]. |
| July 30 UTC, `432d7ffe2c3b4e539f3d0d4ae0a4893090a018d6`, #25676 | Synchronizes asynchronous output copies before freeing the context or clearing pooled embeddings. | Relevant to embedding requests and close/reuse lifetimes; decode return does not imply every asynchronous operation completed. [Commit][async-embeddings]. |
| August 14, `1692f9e50bb20fd96b963af38a282daf78feea64`, #26623 | Adds snapshots to `ggml_ssm_scan`, expanding rollback support including Nemotron-related paths. | Support remains architecture dependent; query resolved `llama_n_rs_seq` and keep fallback behavior. [Commit][ssm-rollback]. |
| August 23, `b0539c43ed13b16bf0d8a0840646faea65469702`, #26756 | Fixes DeepSeek V4 multi-sequence rollback; makes pending recurrent rollback single-use in shared recurrent-memory code. | A second partial removal before the pending restore is consumed can fail. Check every removal result; use checkpoint/replay or controlled reset when needed. [Commit][rollback-multiseq]. |
| August 28, `86632248188c106d749fad34a1dcd237c95863d4`, #27877 | Removes non-fused GDN/Lightning Indexer selection paths; unsupported backend operations can fall back to CPU. | Rebenchmark actual placement; enabling a GPU backend does not prove every operation stays there. [Commit][fused-ops], [PR][fused-ops-pr]. |
| August 31, `2d8d612e4c68d3801e556a1b4a028f55ec33ecbb`, #27991 | Restores fragmented KV cells by contiguous runs; fixes the on-device reader's assumption that equal tensor counts imply equal chunk sizes; adds host/device tests. | Valuable for checkpoint/prefix reuse. Do not preserve obsolete one-copy-per-cell or one-to-one chunk assumptions. [Commit][restore-runs]. |
| September 6, `5fdfa6282936576d2f352d4b97f397a109f207a6`, #28068 | GDN Q/K normalization changes from `x / max(sqrt(sum(x*x)), eps)` to `x * rsqrt(sum(x*x) + eps)` for Qwen3.5/MoE and others. | Logits/output can legitimately differ from b9592. Compare speculative and ordinary decoding within the **same new build**, separately from cross-version checks. [Commit][gdn-norm], [PR][gdn-norm-pr]. |

Performance evidence has limits. PR #23940 reports approximately **4% MTP improvement** on DGX Spark with Qwen3.6-35B-A3B Q4_K_M. PR #25532 reports approximately **8% improvement** from backend sampling on RTX 5090 with Qwen3.6-35B Q4_K_M and unchanged acceptance ratios in that test. These prioritize local benchmarking; they are not additive gains or guarantees for this repository. [GDN benchmark][gdn-copies-pr], [sampling benchmark][multi-output-pr].

### Rollback contracts and an upstream discrepancy

The current server computes `n_rollback = draft_count + 1 - accepted_count`, where `accepted_count` includes the replacement/bonus token. It uses a checkpoint if partial removal is unsupported or rollback exceeds resolved `llama_n_rs_seq`; it restores sampler state before replay and informs the drafter with `accepted_count - 1`. A successful verification decode does not commit the entire batch. [Server verification][head-server].

**Source discrepancy:** `llama.h` documents negative sequence IDs in `llama_memory_seq_rm` as matching any sequence. At this head, recurrent `seq_rm` casts the ID to `uint32_t` and rejects out-of-range values, including `-1`; hybrid memory forwards to that implementation and the public wrapper forwards to the memory object. This is a source-level inconsistency, not a runtime reproduction here. Prefer explicit valid IDs for per-sequence removal and `llama_memory_clear` for a whole-memory clear; do not ignore a failed negative-ID removal on hybrid models. [Public contract][head-header], [recurrent implementation][head-recurrent], [hybrid forwarding][head-hybrid], [public wrapper][head-context].

## Backend sampling and verification

**August 10 — `dd1ea524333b1e697489067d7a4c39c60d32beee`, PR #25532:** supports multiple backend-sampled outputs per sequence, enabling target speculative verification sampling. The final interface is **`n_outputs_max_per_seq`**; the PR description retains an earlier working name. Use the merged header. [Commit][multi-output], [PR][multi-output-pr], [header][head-header].

Required behavior:

- Configure total and per-sequence budgets before context creation. Default per-sequence budget is **1**; zero inherits total outputs. The helper budgets `draft_length + 1`, bounded by `n_batch`. Decode returns `-1` when an attached backend sampler exceeds its per-sequence limit; this is not a general restriction on CPU-side verification. [Context][head-context], [output-limit helper][head-spec].
- `llama_get_sampled_token_ith` reads without committing sampler state. Accept a **contiguous prefix in output order**. `llama_sampler_sample` already samples **and accepts**; do not accept twice. [Header][head-header], [implementation][head-sampler].
- New `llama_sampler_copy(src, dst)` and `copy_state` preserve destination sampling-graph references. Upstream rollback copies into the existing attached sampler rather than replacing it. Source and destination must match in type/configuration. [Header][head-header], [server][head-server], [copy implementation][head-sampling].
- Advance randomness only through committed outputs. The ordinary upstream verifier samples the target sequentially, stops at the first mismatch, and samples a bonus token only after all drafts match. It does not blindly accept high-confidence draft tokens. [PR][multi-output-pr], [verifier][head-sampling].
- Current common sampling disables backend sampling with grammar or reasoning-budget samplers. Capability-check attachment and keep CPU fallback, as the MTP draft constructor does. [Sampling checks][head-sampling], [MTP constructor][head-spec].

Linking `libllama` does not automatically supply these `common/` policies. A binding with its own sampler chain/speculative loop must implement the necessary contracts or deliberately depend on pinned upstream common code.

## Other speculative interfaces: separate integration work

| Milestone | Integration relevance |
| --- | --- |
| EAGLE3: June 12, `88a39274ecf88ba11686acd357b59685b1cbf03d`, #18039 | Adds target-layer extraction and a distinct draft model/feature path. Layer-input APIs remain in the staging header. [Commit][eagle3], [header][head-ext]. |
| DFlash: June 28, `d1b34251bc57b696a5c91968069f8a0e6be13ef4`, #22105 | Block drafting consumes multiple target feature layers rather than single MTP carryover. [Commit][dflash]. |
| DSpark: July 28, `84075273c82f7681d43436b692073cbd4ab15fe9`, #25173 | Adds another implementation and model-specific draft machinery. [Commit][dspark]. |
| DFlash2: August 27, `b10f9ca58c89ccfc3653ac01e979dd085d582b76`, #27816 | Local convolution/candidate selector; current driver reads a selector lattice through NextN output instead of ordinary draft logits. [Commit][dflash2], [driver][head-spec]. |
| DFlash encoder fusion: August 31, `662a0b0121a53c23b825a71e64ab6eff59b7f4d8`, #27310 | Fuses encoder work into draft KV injection. Follow current flow when adopting this type. [Commit][dflash-fusion]. |
| Position fix: September 11, `b0dcb8192b201e402ec3eff524e55450f8070e3e`, #28715 | Renames `n_past` to **`pos0`** and supplies actual position rather than token count after images, affecting all drafters. It does not resolve MTP's embedded-input TODO. [Commit][spec-position], [driver][head-spec]. |

For MTP, `id_last` is the **already sampled target token at `pos0`**, not the last prompt token; `pending_h` belongs to the preceding target position. After verifying `[id_last, drafts...]`, pass the accepted-draft count to `accept` and carry the replacement/bonus token into the next round. The [multi-head PR's explicit position trace][multi-head-pr] and [current driver][head-spec] demonstrate this distinction. This is particularly important when replacing a hand-written loop with a persistent common driver.

These invariants do not require a wrapper to expose an off-by-one KV state at generation exit. A wrapper can finish with aligned KV by evaluating its final corrected token and updating the persistent driver consistently; continuation logic should follow that chosen invariant rather than importing assumptions from the previous local loop. This is an integration recommendation, not an audit of the owner's new exit implementation.

The current `common/speculative.h` removes `common_speculative_need_embd` and `common_speculative_need_embd_nextn`; adds initialization helpers, a resolved `common_speculative_n_max` overload, output-limit calculation, and optional get/set-state; and uses `pos0`. These are C++ common-library interfaces, **not public `llama.h` additions**. [Header][head-spec-header], [comparison][comparison].

Optional speculative-state APIs do not prove MTP state is serialized: the current MTP class does not override the base get/set-state methods. Maintain an explicit lifecycle for carryover and draft KV across prefix reuse, reset, and session restore. [Implementation][head-spec].

Keep a persistent driver alive across all required target prefill/verification hooks. The MTP implementation sizes its draft batch from draft `n_batch` and consumes the supplied batch's rows without splitting that copy itself. Choose hook chunks within both contexts' logical batch capacities; respecting target `n_ubatch` also makes physical prefill boundaries explicit. This is a recommended integration discipline, not a claim that the core can never split a larger logical batch. Preserve carryover at every boundary. [Driver allocation/process][head-spec], [core decoding][head-context].

**Synthetic acceptance is benchmark-only:** August 27 `2bb9bddafad44ecbb50889644ca47537ec11841b`, #27711, adds synthetic lengths/rates and selects a separate server acceptance routine. Exclude it from production generation and correctness comparisons. [Commit][synth], [server][head-server].

Upstream added mean acceptance length and acceptance by position on June 16 (`635b65ad7a194cdb7fdbe21681683d5cb4b5188e`, #24536). Record them with wall-clock throughput, prompt time, draft/verify/replay cost, and memory. Acceptance ratio alone does not establish speedup. [Commit][metrics], [current calculations][head-server].

## Public `llama.h` migration checklist

This table is based on the direct baseline-to-head header diff. It identifies binding-relevant surfaces without claiming they are all used locally.

| Surface | Change / compatibility action | Primary source |
| --- | --- | --- |
| **Model loading: breaking fields** | Removes `use_mmap`, `use_mlock`, `use_direct_io`; adds `load_mode`. Preserve old explicit choices: neither → `NONE`, mmap only → `MMAP`, mlock only → `MLOCK`, both → `MMAP_MLOCK`. `DIRECT_IO` is a separate mode. | July 23 `e6dd0e29a6751d4859abaa8899959f5ddf756f4e` [#20834][load-mode]; July 27 `ad256ded30b5e9dbf43c146b452673a1471b62cd` [#26135][mlock-mode]. |
| **Load defaults / model layout** | `load_mode` defaults to `AUTO`, which can avoid mmap based on device capabilities. This differs from an explicit legacy mmap choice. New `lazy_mode` enables on-demand reads of marked tensors, requiring mmap for active lazy reading. | August 11 `153d324bcf86d220b235ca010eeb11213f32b5d1` [#26081][load-auto]; August 27 `fac889fb38fd0e267636bd95bf096555e45b2270` [#27794][lazy-load]; [defaults][head-model]. |
| **MTP loading** | Adds `load_mtp=false`; a successful rebuild alone can leave MTP weights unavailable. | [#26296][mtp-load]. |
| **Context layout** | Adds `n_outputs_max_per_seq`, default 1; zero inherits total outputs. Configure for multi-output backend sampling. | [#25532][multi-output], [defaults][head-context]. |
| **Penalty sampler: breaking signature** | `llama_sampler_init_penalties` gains leading `int32_t n_vocab`. Use `llama_vocab_n_tokens(llama_model_get_vocab(model))`. | August 4 `935cad6497e8d1569b3302c1d12d03472b556401` [#26520][penalty-vocab]. |
| **DRY: breaking signature / history semantics** | Removes `n_ctx_train` from `llama_sampler_init_dry`; removes full-context `-1` semantics. Current penalty and DRY constructors clamp negative windows to zero. Resolve a Python `-1 = context length` promise to an explicit positive window or migrate that contract. | August 4 `a6aa6f5450eaad18b3c86631b5c3fff330f5a46e` [#26524][history-samplers], [constructors][head-sampler]. |
| **Custom sampler: breaking interface** | `backend_init` gains per-sequence output budget; adds `backend_reset` and `copy_state`; adds `llama_sampler_copy`. Recompile and update custom interface initializers. | [#25532][multi-output], [header][head-header]. |
| **Suppressed tokens: behavior** | Adds `llama_vocab_get_suppress_tokens`; common sampling applies model suppression through `-INFINITY` logit biases. Custom sampling must supply that policy when needed rather than assuming raw logits already do so. | July 29 `afeebe103bd99cda8f5dfaefcabadf890db7fda7` [#26276][suppress-tokens], [implementation][head-sampling]. |
| **Saved-state format: breaking persistence** | Session version **9 → 10**; sequence-state version **2 → 3**, with token IDs in KV cells. Do not promise old saved sessions or raw blobs are portable; key caches by model/config/upstream identity. | August 26 `925e1179947ea0c0ebfb0032df18af3a729822be` [#27762][state-format]. |
| **State-file query** | `llama_state_seq_load_file` with `tokens_out=NULL` reports token count without loading state; this call is not a completed restore. | August 12 `5d9e5ac30e469d44c0a5a52556de0ead03aaa5b0` [#26640][state-media], [header][head-header]. |
| **Metadata/version additions** | Adds `llama_model_n_layer_nextn`, `llama_ftype_name`, `llama_model_ftype`, `llama_version`, and `LLAMA_FTYPE_MOSTLY_Q2_0`. Older libraries lack the new symbols. | [#24340][multi-head]; July 2 `fdb1db877c526ec90f668eca1b858da5dba85560` [#25134][ftype]; July 7 `bec4772f6a2527d371557b5d2032641e5ff7619c` [#24448][q2]; [#26839][semver]. |
| **Quantization struct layout** | Adds `size_t max_buf_size`, zero selecting an 8 GiB working-buffer default. Relevant if quantization parameters are exposed/constructed. | August 27 `732707dff265be513347b02911e1dd18f7e1f386` [#27795][quant-buffer]. |
| **Sampler count declaration** | `llama_sampler_chain_n`: `int` → `int32_t`. Usually equivalent on conventional platforms, but update exact generated/function-pointer declarations. | September 9 `4850c7727fa73bbe3098e10ee369fbc3467c445f` [#28631][chain-n]. |

Semver support arrived August 12 (`680a9ae63d60d35c21a0dcd7d3fabdb9c6bfc963`, #26839), including `llama_version`, CMake package versions, and library version properties. At head, `SOVERSION` remains major version **0**. The shared-library major name therefore does not establish compatibility across this interval's demonstrated layout/signature changes. `llama_version()` returns a version string, not the commit; record the commit separately. [Change][semver], [CMake][head-cmake], [library properties][head-lib-cmake], [implementation][head-llama].

## Integration sequence and validation recommendations

These are recommendations for the implementation owner, not completed local checks.

1. **Choose a reproducible upstream boundary.** The reported installed `7cf1c54a…` includes these core changes. Match headers, `libllama`, ggml libraries, and any compiled common code to the same source revision. For continued b9592 support, use compile-time feature/signature checks and explicit branches; runtime version strings cannot repair struct ABI mismatches or missing import-time symbols.
2. **Finish API migration and MTP load policy first.** Adapt load modes and sampler signatures; preserve or migrate negative history semantics; distinguish absent metadata, unloaded weights, unsupported model behavior, and allocation failure.
3. **Validate existing single-head correctness.** Include first sampled token/`pos0`, prefill chunk carryover, rejected suffixes, accepted-token budget exits, EOS/stop/cancellation, prefix reuse, and speculative-to-ordinary continuation. Check every trim/restore result and explicitly handle pending recurrent rollback.
4. **Compare with ordinary decoding in the same new build.** Fix model, quantization, sampler settings, seed, devices, and batching. Check greedy agreement and sampled RNG/continuation behavior through rejection/replay; analyze differences from b9592 separately. The owner identified `/home/lianghong/Projects/llama_cpp_projects/models/Qwen3.5-4B-Q4_K_M-MTP.gguf` as an available fixture; its contents and runtime behavior were not independently inspected here.
5. **Benchmark optimizations after correctness.** Prioritize CUDA GDN improvements, bounded recurrent snapshots, and then backend draft/target sampling with fallback and grammar coverage. Sweep modest draft lengths using real acceptance metrics; report memory and replay cost.
6. **Expand modes separately.** Step multi-head, shared-KV assistants, DeepSeek width variants, and EAGLE3/DFlash/DSpark each require dedicated fixtures and support boundaries. Do not infer general MTP media support from the server position fix.

Useful upstream validation references are the current [recurrent rollback test][head-rollback-test] and host/on-device fragmented state-restore tests added by [#27991][restore-runs]. Their existence guides coverage; this report does not claim they pass locally.

## Primary sources

Source links are pinned to exact commits. PR links provide merged rationale and attributed benchmarks; final interfaces were checked against code.

[local-changelog]: https://github.com/lianghong/llama-cpp-nanobind/blob/5ad73e8490f690ae8c6ba3203dacdd6ca381f348/docs/CHANGELOG-2026-06-12.md
[baseline]: https://github.com/ggml-org/llama.cpp/commit/ac4cddeb0dbd778f650bf568f6f08344a06abe3a
[baseline-tag]: https://api.github.com/repos/ggml-org/llama.cpp/git/ref/tags/b9592
[head]: https://github.com/ggml-org/llama.cpp/commit/1bc7a5af0d14b1fb72f266abbd1237b394187115
[head-api]: https://api.github.com/repos/ggml-org/llama.cpp/commits/1bc7a5af0d14b1fb72f266abbd1237b394187115
[comparison]: https://github.com/ggml-org/llama.cpp/compare/ac4cddeb0dbd778f650bf568f6f08344a06abe3a...1bc7a5af0d14b1fb72f266abbd1237b394187115
[rename]: https://github.com/ggml-org/llama.cpp/commit/166fe29492abb4093ec889b5c6f6fdb4e3b8ba98
[head-cmake]: https://github.com/ggml-org/llama.cpp/blob/1bc7a5af0d14b1fb72f266abbd1237b394187115/CMakeLists.txt#L5-L25
[head-lib-cmake]: https://github.com/ggml-org/llama.cpp/blob/1bc7a5af0d14b1fb72f266abbd1237b394187115/src/CMakeLists.txt#L47-L58
[head-header]: https://github.com/ggml-org/llama.cpp/blob/1bc7a5af0d14b1fb72f266abbd1237b394187115/include/llama.h
[head-ext]: https://github.com/ggml-org/llama.cpp/blob/1bc7a5af0d14b1fb72f266abbd1237b394187115/src/llama-ext.h
[head-context]: https://github.com/ggml-org/llama.cpp/blob/1bc7a5af0d14b1fb72f266abbd1237b394187115/src/llama-context.cpp
[head-model]: https://github.com/ggml-org/llama.cpp/blob/1bc7a5af0d14b1fb72f266abbd1237b394187115/src/llama-model.cpp
[head-spec]: https://github.com/ggml-org/llama.cpp/blob/1bc7a5af0d14b1fb72f266abbd1237b394187115/common/speculative.cpp
[head-spec-header]: https://github.com/ggml-org/llama.cpp/blob/1bc7a5af0d14b1fb72f266abbd1237b394187115/common/speculative.h
[head-sampling]: https://github.com/ggml-org/llama.cpp/blob/1bc7a5af0d14b1fb72f266abbd1237b394187115/common/sampling.cpp
[head-sampler]: https://github.com/ggml-org/llama.cpp/blob/1bc7a5af0d14b1fb72f266abbd1237b394187115/src/llama-sampler.cpp
[head-server]: https://github.com/ggml-org/llama.cpp/blob/1bc7a5af0d14b1fb72f266abbd1237b394187115/tools/server/server-context.cpp
[head-recurrent]: https://github.com/ggml-org/llama.cpp/blob/1bc7a5af0d14b1fb72f266abbd1237b394187115/src/llama-memory-recurrent.cpp#L161-L214
[head-hybrid]: https://github.com/ggml-org/llama.cpp/blob/1bc7a5af0d14b1fb72f266abbd1237b394187115/src/llama-memory-hybrid.cpp#L143-L150
[head-llama]: https://github.com/ggml-org/llama.cpp/blob/1bc7a5af0d14b1fb72f266abbd1237b394187115/src/llama.cpp#L118-L120
[head-rollback-test]: https://github.com/ggml-org/llama.cpp/blob/1bc7a5af0d14b1fb72f266abbd1237b394187115/tests/test-recurrent-state-rollback.cpp
[mtp-load]: https://github.com/ggml-org/llama.cpp/commit/82dbc4f017a7b005f993ac2e7af9c048ad686c04
[mtp-load-pr]: https://github.com/ggml-org/llama.cpp/pull/26296
[multi-head]: https://github.com/ggml-org/llama.cpp/commit/d789527482d925156d7c4adfecebf5fb8481e0ee
[multi-head-pr]: https://github.com/ggml-org/llama.cpp/pull/24340
[dsv4-mtp]: https://github.com/ggml-org/llama.cpp/commit/596a5795bdd6da317ea103fc06c0a71c296e3669
[mtp-detect]: https://github.com/ggml-org/llama.cpp/commit/1d2869c6e54d5003f3927a79efbca0fefa034a6d
[mtp-pooling]: https://github.com/ggml-org/llama.cpp/commit/2c6b141efb3b0868fd39d3cae73f69606e1d654c
[mtp-pooling-pr]: https://github.com/ggml-org/llama.cpp/pull/27400
[nextn-load-order]: https://github.com/ggml-org/llama.cpp/commit/9d817213a0975020775efe6c458822616826f376
[mtp-kv]: https://github.com/ggml-org/llama.cpp/commit/5cdd3d1dad5cbb7107b3e9f6d23239ba88ac0123
[gdn-copies]: https://github.com/ggml-org/llama.cpp/commit/5a460dea9f961cdb508d58a6e7b0f9e259b4c19f
[gdn-copies-pr]: https://github.com/ggml-org/llama.cpp/pull/23940
[recurrent-split]: https://github.com/ggml-org/llama.cpp/commit/230ea9d214320c5e79cc8166ed708ac60514c71e
[rollback-tests]: https://github.com/ggml-org/llama.cpp/commit/32e789fdfd598e9a1872da55ac941e4d94f030bd
[rollback-tests-pr]: https://github.com/ggml-org/llama.cpp/pull/25758
[async-embeddings]: https://github.com/ggml-org/llama.cpp/commit/432d7ffe2c3b4e539f3d0d4ae0a4893090a018d6
[ssm-rollback]: https://github.com/ggml-org/llama.cpp/commit/1692f9e50bb20fd96b963af38a282daf78feea64
[rollback-multiseq]: https://github.com/ggml-org/llama.cpp/commit/b0539c43ed13b16bf0d8a0840646faea65469702
[fused-ops]: https://github.com/ggml-org/llama.cpp/commit/86632248188c106d749fad34a1dcd237c95863d4
[fused-ops-pr]: https://github.com/ggml-org/llama.cpp/pull/27877
[restore-runs]: https://github.com/ggml-org/llama.cpp/commit/2d8d612e4c68d3801e556a1b4a028f55ec33ecbb
[gdn-norm]: https://github.com/ggml-org/llama.cpp/commit/5fdfa6282936576d2f352d4b97f397a109f207a6
[gdn-norm-pr]: https://github.com/ggml-org/llama.cpp/pull/28068
[multi-output]: https://github.com/ggml-org/llama.cpp/commit/dd1ea524333b1e697489067d7a4c39c60d32beee
[multi-output-pr]: https://github.com/ggml-org/llama.cpp/pull/25532
[eagle3]: https://github.com/ggml-org/llama.cpp/commit/88a39274ecf88ba11686acd357b59685b1cbf03d
[dflash]: https://github.com/ggml-org/llama.cpp/commit/d1b34251bc57b696a5c91968069f8a0e6be13ef4
[dspark]: https://github.com/ggml-org/llama.cpp/commit/84075273c82f7681d43436b692073cbd4ab15fe9
[dflash2]: https://github.com/ggml-org/llama.cpp/commit/b10f9ca58c89ccfc3653ac01e979dd085d582b76
[dflash-fusion]: https://github.com/ggml-org/llama.cpp/commit/662a0b0121a53c23b825a71e64ab6eff59b7f4d8
[spec-position]: https://github.com/ggml-org/llama.cpp/commit/b0dcb8192b201e402ec3eff524e55450f8070e3e
[synth]: https://github.com/ggml-org/llama.cpp/commit/2bb9bddafad44ecbb50889644ca47537ec11841b
[metrics]: https://github.com/ggml-org/llama.cpp/commit/635b65ad7a194cdb7fdbe21681683d5cb4b5188e
[load-mode]: https://github.com/ggml-org/llama.cpp/commit/e6dd0e29a6751d4859abaa8899959f5ddf756f4e
[mlock-mode]: https://github.com/ggml-org/llama.cpp/commit/ad256ded30b5e9dbf43c146b452673a1471b62cd
[load-auto]: https://github.com/ggml-org/llama.cpp/commit/153d324bcf86d220b235ca010eeb11213f32b5d1
[lazy-load]: https://github.com/ggml-org/llama.cpp/commit/fac889fb38fd0e267636bd95bf096555e45b2270
[penalty-vocab]: https://github.com/ggml-org/llama.cpp/commit/935cad6497e8d1569b3302c1d12d03472b556401
[history-samplers]: https://github.com/ggml-org/llama.cpp/commit/a6aa6f5450eaad18b3c86631b5c3fff330f5a46e
[suppress-tokens]: https://github.com/ggml-org/llama.cpp/commit/afeebe103bd99cda8f5dfaefcabadf890db7fda7
[state-format]: https://github.com/ggml-org/llama.cpp/commit/925e1179947ea0c0ebfb0032df18af3a729822be
[state-media]: https://github.com/ggml-org/llama.cpp/commit/5d9e5ac30e469d44c0a5a52556de0ead03aaa5b0
[ftype]: https://github.com/ggml-org/llama.cpp/commit/fdb1db877c526ec90f668eca1b858da5dba85560
[q2]: https://github.com/ggml-org/llama.cpp/commit/bec4772f6a2527d371557b5d2032641e5ff7619c
[semver]: https://github.com/ggml-org/llama.cpp/commit/680a9ae63d60d35c21a0dcd7d3fabdb9c6bfc963
[quant-buffer]: https://github.com/ggml-org/llama.cpp/commit/732707dff265be513347b02911e1dd18f7e1f386
[chain-n]: https://github.com/ggml-org/llama.cpp/commit/4850c7727fa73bbe3098e10ee369fbc3467c445f
