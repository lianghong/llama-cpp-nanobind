# MTP first-token streaming and chat — 2026-10-13

The final draft-prefill chunk now follows initial target sampling and streaming
handling. A safe first token can reach the callback before draft catch-up.
Normal generation still completes that work before drafting or another target
decode. This builds on [MTP graph reuse](MTP_LATENCY-2026-10-13.md).

## Measured cause and change

A fresh narrowed benchmark reproduced 6.1 ms of warmed MTP overhead:
25.3 ms ordinary first-token latency versus 31.4 ms MTP. It used 64 prompt
tokens, one output token, draft width 2, seven measured requests after two
warmups. A local diagnostic requiring at most 3 ms of MTP overhead failed.

After deferral, ordinary latency remained 25.3 ms and MTP fell to **23.9 ms**,
a **24% reduction** from the previous binding. The diagnostic passed.
Total one-token request time was nearly unchanged: this moves required work
after the callback rather than eliminating it.

The earlier phase profile attributed about 7.6 ms to draft prefill on this
prompt length. The measured 7.5 ms latency reduction supports draft prefill
as the remaining blocking work.

## Native throughput and memory

Fixed greedy prompts, 128 output tokens, five timed repetitions after two
warmups, four CPU threads, full GPU offload, `n_ctx=2048`,
`n_batch=n_ubatch=512`, and `n_rs_seq=2`. TTFT is in milliseconds.

| Prompt | Mode | TTFT before | TTFT after | Tokens/s before | Tokens/s after |
| ---: | --- | ---: | ---: | ---: | ---: |
| 64 | Ordinary | 28.2 | 28.3 | 92.6 | 92.7 |
| 64 | MTP width 1 | 32.1 | 24.2 | 106.1 | 108.6 |
| 64 | MTP width 2 | 32.7 | 24.1 | 118.3 | 118.8 |
| 256 | Ordinary | 55.9 | 56.3 | 90.9 | 91.0 |
| 256 | MTP width 1 | 65.6 | 53.9 | 105.8 | 106.1 |
| 256 | MTP width 2 | 66.0 | 54.8 | 112.5 | 112.4 |

At width 2, first-token latency decreased **26.2%** and **17.0%** for 64 and
256 prompt tokens. Total throughput remained essentially steady. Every token
ID and proposed/accepted draft count matched before and after, including
warmup requests.

GPU process-memory samples were identical for each paired workload: 3694–3696
MiB ordinary and 3838–3842 MiB MTP. Host RSS varied by at most 4 MiB between
paired samples. This comparison does not measure peak memory.

Raw native reports:

- [Before](benchmarks/2026-10-13-mtp-first-token-before.json)
- [After](benchmarks/2026-10-13-mtp-first-token-after.json)
- [Narrowed workload before](benchmarks/2026-10-13-mtp-first-token-single-before.json)
- [Narrowed workload after](benchmarks/2026-10-13-mtp-first-token-single-after.json)

## Three-turn streaming chat

These historical chat measurements used the then-current tokenizer, which
encoded template role markers as ordinary text by default. The native results
above remain independent of that issue. The
[cache study](MTP_CACHE-2026-10-13.md) fixes chat tokenization and records
comparisons with the corrected template protocol on both versions.

The public chat API used three related questions, greedy sampling, and a
64-token output budget per turn. Each version ran five measured conversations
after two warmups. At draft width 2, time to first nonempty text was:

| Turn | P50 before (ms) | P50 after (ms) | P95 before (ms) | P95 after (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 35.3 | 26.7 | 36.4 | 27.6 |
| 2 | 26.5 | 25.6 | 31.2 | 26.5 |
| 3 | 27.1 | 25.6 | 27.4 | 26.1 |

The first turn improved **24.4%** at P50. Later turns already reused 126 and
231 cached tokens from prompts of 167 and 273 tokens; the draft driver
processed only the remaining 41 and 42 prompt tokens. Their P50 gains were
smaller, 0.9 and 1.5 ms. Width 1 improved first-turn P50 from 33.5 to 25.8 ms,
with little change in later turns. The ordinary control varied from
25.1/24.0/25.8 to 26.2/24.7/24.9 ms across the three turns.

Every complete response, prompt hash, retained cache size, and proposed/accepted
draft count matched before and after, including warmups. Each version also
matched ordinary greedy responses. This establishes unchanged behavior for
the measured conversations.

At width 2, whole-request median durations were 705.9/618.2/696.4 ms before
and 696.4/615.4/693.4 ms after. Exact UTF-8 throughput was
466.1/433.5/438.0 bytes/s before and 472.4/435.5/439.9 bytes/s after.
These small differences do not establish a general completion-speed gain.

GPU memory samples were unchanged: 3692 MiB ordinary, 3836–3840 MiB width 1,
and 3842 MiB width 2. Across measured turns, width-2 RSS ranged from
1242–1253 MiB before and 1237–1251 MiB after. These are samples after completed
turns, rather than peak measurements.

Raw chat reports:

- [Before](benchmarks/2026-10-13-mtp-chat-before.json)
- [After](benchmarks/2026-10-13-mtp-chat-after.json)

## Sequencing and exits

Upstream's
[MTP process hook](https://github.com/ggml-org/llama.cpp/blob/cbe28bb3cb0758f0c566a9a3561766821aeb0748/common/speculative.cpp#L1539)
reads hidden rows from the most recent target batch. Earlier prefill chunks
still process immediately; their rows must be consumed before the next target
decode overwrites them.

The final batch remains intact through sampling and the initial emission
attempt. It is then processed before
[`common_speculative_begin`](https://github.com/ggml-org/llama.cpp/blob/cbe28bb3cb0758f0c566a9a3561766821aeb0748/common/speculative.cpp#L1518),
drafting, or final-token decode. Target sampling uses its own sampler, so
resetting the draft sampler in `begin` can remain after prefill.

- A normal one-token budget still processes the deferred chunk and final
  sampled token. KV, current logits, and pending hidden rows remain aligned.
- First-token stop/EOG exits clear KV and skip the deferred chunk.
- A callback refusing the first token or throwing clears KV and restores
  ordinary graph mode, without processing the deferred chunk.
- Multi-token stop prefixes remain buffered. If the first token cannot safely
  be delivered yet, draft processing proceeds before the eventual callback.
  This optimization does not bypass stop buffering.
- Fully cached continuation has no pending prefill chunk. It uses the retained
  hidden row and current target logits.

`perf()["n_mtp_prefill"]` counts successfully forwarded target prompt tokens,
excluding verification and final-token catch-up. It accumulates until
`perf_reset()` and survives KV clearing. This provides a deterministic
callback-ordering check without wall-clock assertions in CI.

## Regression evidence

The new callback-ordering test failed on the old sequence: 12 prompt tokens
had already reached the draft driver when the first callback ran, versus the
expected zero. It now covers one and several prefill chunks, one-token and
longer budgets, and cached ordinary/MTP continuation.

The continuation oracle replays identical MTP prefill before comparing modes,
so both begin with the same KV and current logits. One truncated repeated-text
fixture already differed between cached continuation and fresh whole-prefix
prefill in the previous binding; the before and after continuations matched
exactly. Existing fresh-prefill differential tests remain unchanged.
Upstream state serialization restores memory without current output logits,
so a state-only restore is not a suitable oracle for zero-decode sampling.

Additional checks cover first-token stops, callback refusal/exceptions,
prefill counters across clear/reset, grammar, rollback, near-context limits,
stop buffering, and device-state handles.

Against the isolated upstream HEAD, compatible fixtures account for all
**365 unique tests: 364 passed, 1 genuine-MoE skip**. The ordinary LFM fixture
completed an early-break chat test after two chunks, so Qwen exercised that
test's required three-chunk break. Qwen also covers UnifiedLLM and lifecycle
tests; MiMo supplies the non-MTP Qwen fixture through the same explicit-family
adapter used in earlier validation.

JUnit and merged observations are in `build/mtp-first-token/`. The initial
LFM fixture failure remains visible in `head-main.xml`; a passing compatible
run takes precedence in the merged results. The installed b11552 baseline's
44 MTP cases are accounted for with 42 passes and two ordinary-model skips,
including the focused rerun after correcting the new continuation oracle.

GCC 15 and Clang 21 portable HEAD wheels built and passed native-provenance
checks. Ruff lint/format, mypy across 29 files, warning-as-error Python
compilation, clang-format, and whitespace checks passed. Clang-tidy completed
with the same four existing state-position warnings documented in the prior
latency report. Temporary continuation probes are under the ignored
`build/mtp-first-token/debug/` directory.

## Reproduction

Use matching upstream headers/libraries and native-runtime provenance checks
from [the compatibility guide](UPSTREAM_VALIDATION.md). Both binding versions
use the same isolated upstream commit
`cbe28bb3cb0758f0c566a9a3561766821aeb0748`, GCC 15 portable builds, CUDA 13.3,
and the supplied Qwen3.5-4B-Q4_K_M-MTP checkpoint on RTX 3080.

Native fixed-prompt measurements:

```bash
rtk proxy env LD_LIBRARY_PATH=/path/to/prefix/lib \
  /path/to/venv/bin/python -m tools.benchmark_upstream \
  --model /path/to/Qwen3.5-4B-Q4_K_M-MTP.gguf \
  --prompt-tokens 64 256 --max-tokens 128 \
  --draft-max 1 2 --runs 5 --warmup 2 \
  --output /path/to/native-report.json
```

Public streaming chat measurements:

```bash
rtk proxy env LD_LIBRARY_PATH=/path/to/prefix/lib \
  /path/to/venv/bin/python -m tools.benchmark_mtp_chat \
  --model /path/to/Qwen3.5-4B-Q4_K_M-MTP.gguf \
  --max-tokens 64 --runs 5 --warmup 2 \
  --output /path/to/chat-report.json
```

The chat benchmark uses three related questions and includes template
processing, tokenization, sampler/thread creation, queue delivery, and text
decoding. Later turns use `reset_kv_cache=False, cache_prompt=True`; reports
record matching prefixes and retained cache size, rather than assuming reuse
occurred. Stops or hybrid prefix edits can require full re-prefill.

First-text latency ignores role-only and empty chunks. Byte throughput uses
exact UTF-8 output size and whole-request duration; native token throughput is
measured separately. P50/P95 describe the five measured conversations at each
turn. Model loading and initial capability probing are outside timing.
Memory readings are samples after each completed request, not peak memory.

Run before and after sequentially with no concurrent builds or model tests.
Source snapshots and extension hashes identify the actual loaded versions.
