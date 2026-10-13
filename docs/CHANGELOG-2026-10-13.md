# Changes and validation — 2026-10-13

This update restores import/build compatibility after the installed llama.cpp
upgrade and reduces binding overhead. The upstream investigation is in
[the research report](UPSTREAM_RESEARCH-2026-10-13.md).

## Compatibility and correctness

- Migrated speculative prefill and verification to owned `common_batch` and
  `llama_process`. The removed `llama_batch` overload had caused an undefined
  symbol when importing the previous extension.
- The CMake compile/link probe exercises the current common batch interface
  and reruns on every configure, including reused build directories.
- Standalone MTP graphs now require target state embeddings. Token-only
  generation raises `ValidationError`; native `decode`, `decode_one`, and
  `decode_multi` raise `ValueError` before upstream can abort. Use a default
  context with `speculative=True` for supported MTP generation.
- Stop handling checks the actual rollback result before changing position.
  LFM2 can report shift support while refusing a requested recurrent rewind.
  A refused rewind clears KV; the Python mirror and device-state epoch are
  invalidated, so subsequent requests re-prime safely.
- Native parameter dictionaries serialize unregistered C enums as integers,
  fixing `std::bad_cast` from model split mode and context flash-attention mode.
- Documented the supplemental common/staging headers needed after upstream
  installation. Previous serialized states require regeneration: upstream
  session version is now 11 and sequence-state version is 4.

## Optimizations and controls

- Reuse prompt batch storage across decode calls and resets, growing it only
  when needed. Context closure frees the retained allocation.
- Fixed zero-temperature chains use a greedy terminal sampler, avoiding
  distribution softmax and RNG work. The preceding temperature mask retains
  existing sampler-adjusted logprobs; dynamic temperature and adaptive-p keep
  their existing paths.
- Added `cpu_moe=True`, matching upstream host expert placement, and
  experimental `moe_cache_size` in bytes for the target GPU expert cache.
  Its default remains zero; internal MTP draft contexts disable the cache.
  The cache requires compatible host experts, a GPU backend, and enough space
  for active experts. See [configuration documentation](API.md#host-expert-cache).
- CMake honors `CC` / `CXX` and uses its verified, compiler-specific IPO/LTO
  flags instead of unconditionally adding GCC-specific LTO options.
- Successful MTP requests retain NextN graph mode, avoiding repeated target
  reservations when KV clearing requires a new speculative driver. Ordinary
  decoding restores normal mode; cancellation and exceptions restore it through
  a guard. Context reset clears the tracked mode. KV and hidden-state
  invalidation retain their existing rules. See
  [the latency investigation](MTP_LATENCY-2026-10-13.md).
- Final draft prefill follows the first target-token callback. Earlier chunks
  still process before the next target decode; the last processes before
  `begin`, drafting, or final-token catch-up. First-token stops, cancellation,
  and callback exceptions avoid the deferred work and clear KV.
  See [streaming and multi-turn measurements](MTP_FIRST_TOKEN-2026-10-13.md).
- Added `tools.benchmark_mtp_sweep` for widths 1–4 at 512/2048/4096 prompt
  tokens, native and three-turn chat timing, exact IDs, draft acceptance,
  actual prefix reuse, and sampled peak GPU process memory through NVML.
  The default remains 2. See [tuning results](MTP_TUNING-2026-10-13.md).
  Stop/cancellation coverage now spans all 12 width/long-prompt combinations;
  71 unique MTP checks passed across compatible fixtures.

## Validation scope

Runtime validation for the initial update uses installed **b11552 / `23b0202`**,
independently queried from libllama-common, with matching headers and libraries.
The public
HEAD examined by research is **`cbe28bb3`**; its headers receive a compile-only
check. Runtime compatibility with that HEAD is not claimed because its
performance-data struct differs from the installed library.

The follow-up built matching HEAD libraries in an isolated prefix and
validated their runtime. It also added compatibility CI, native provenance
checks, acceptance counters, and reproducible benchmarks. See
[the current validation guide and results](UPSTREAM_VALIDATION.md).

The supplied model directory provides ordinary LFM2.5-350M and
Qwen3.5-4B-MTP coverage. MoE cache throughput and positive allocation require
a genuine MoE checkpoint; the optional integration test uses
`LLAMA_MOE_TEST_MODEL`. Dense models do not validate that feature.

All **344 collected tests** were accounted for across compatible fixtures:
**343 passed, 1 skipped**. This is the aggregate of separate runs:

- LFM2.5 for ordinary inference and Qwen3.5-4B-MTP for MTP tests:
  **282 passed, 10 skipped, 2 deselected**. Nine skips require a model
  supported by UnifiedLLM; the remaining skip lacks a MoE cache checkpoint.
- Qwen3.5-4B-MTP for UnifiedLLM, speculative wrapper, and the two streaming
  early-break tests: **50 passed**. Two negative MTP tests in that run
  failed because their fixture requires a non-MTP checkpoint.
- Those two unchanged negative test bodies passed using the supplied
  **MiMo-V2.6-Distill-Qwen-9B** checkpoint, which declares `qwen35` and no MTP
  layers. A temporary pytest adapter passed the real constructor an explicit
  `family="qwen3.5-small"` and `n_ctx=512`, because filename auto-detection
  does not recognize the MiMo name: **2 passed**.
- The nine previously skipped UnifiedLLM lifecycle tests passed with
  Qwen3.5-4B-MTP, covering repeated closure, deletion, mixed ownership, and
  rapid creation/closure: **9 passed**.

The LFM model ended two early-break prompts after only two chunks, so those
tests used Qwen to exercise an actual early break. The initial broad run
also exposed the stop-rollback bug and enum serialization defect described
above; their focused regressions and the final broad run passed after fixes.

Additional checks passed:

- GCC 15 editable Release build/import and Clang 21 portable wheel build.
- Compile-only validation of the final C++ source against pinned public HEAD.
- Older libraries rejected at configure time, including after a successful
  configure in the same CMake build directory.
- Ruff lint/format, mypy across 25 source files, clang-format, and Python 3.14
  warning-as-error compilation.
- Existing positional `LlamaConfig` argument order and native enum dictionary
  serialization checks.
- New Markdown files passed lint; all edited document local links passed.
  Older documents retain pre-existing style findings, with no increase in
  their total counts. Diff whitespace checks passed.

## Warmed throughput

Command:

```bash
.venv/bin/python examples/bench_speculative.py \
  --model /path/to/Qwen3.5-4B-Q4_K_M-MTP.gguf \
  --max-tokens 128 --runs 3 --draft-max 1 2
```

NVIDIA RTX 3080, full GPU offload, `n_ctx=2048`, `n_rs_seq=2`,
greedy sampling, warmed contexts, median of three runs. Timing includes
tokenization, prompt processing, generation, and detokenization.

| Mode | Tokens/s | Relative throughput | Greedy output matches |
| --- | ---: | ---: | --- |
| Ordinary | 96.1 | 1.00× | — |
| MTP, draft width 1 | 109.7 | 1.14× | Yes |
| MTP, draft width 2 | 114.6 | 1.19× | Yes |

These compare modes within the updated binding. They do not measure the
incremental improvement over the previous binding, which could not import
against these libraries, or guarantee a speedup for other workloads.
