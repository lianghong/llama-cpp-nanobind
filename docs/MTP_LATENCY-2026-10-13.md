# MTP first-token latency — 2026-10-13

Successful MTP requests now retain NextN graph mode across KV clearing and
speculative-driver reconstruction. Ordinary decoding restores normal mode.
Cancellation and exceptions during MTP setup or decoding also restore it.
Context reset recreates the contexts and clears the tracked mode.

## Diagnosis

The original narrowed workload used Qwen3.5-4B-Q4_K_M-MTP on an RTX 3080:
64 prompt tokens, one output token, draft width 2, seven timed requests after
two warmups. Ordinary first-token latency was 25.7 ms; MTP was 55.4 ms.
An exploratory check requiring less than 10 ms of warmed MTP overhead failed
with 29.7 ms. This threshold is a local diagnostic, not a timing assertion in CI.

Temporary phase timers synchronized target and draft work before attributing
GPU time. Median measurements from five requests after two warmups are
preserved in [the phase profile](benchmarks/2026-10-13-mtp-phase-profile.json):

| Phase, 64-token prompt | Previous lifecycle | Retained graph mode experiment |
| --- | ---: | ---: |
| Speculative driver creation | Under 0.1 ms | Under 0.1 ms |
| Target prefill | 42.4 ms | 19.5 ms |
| Draft prompt processing | About 7.6 ms | About 7.6 ms |
| First-token sampling | About 4.2 ms | About 4.2 ms |
| First-token latency | 54.4 ms | 31.9 ms |

Enabling or disabling NextN
[marks upstream's scheduler reservation as stale](https://github.com/ggml-org/llama.cpp/blob/cbe28bb3cb0758f0c566a9a3561766821aeb0748/src/llama-context.cpp#L1330);
the reservation is deferred until decoding. The old guard disabled NextN at
every request exit, so the next request repeated target graph reservation
during prefill. Driver creation was not the main contributor.

The one-variable experiment retained NextN flags at request exit. For a
256-token prompt, first-token latency fell from 87.1 ms to 64.7 ms, while draft
processing remained about 11–12 ms. The first cold MTP request still reserved
graphs. All diagnostic timers and experimental environment switches were
removed from production source.

## Lifecycle and correctness

Graph shape and speculative hidden state have separate lifetimes:

- KV clears, state changes, and prefix edits still invalidate the driver.
  A new MTP driver rebuilds from the complete prompt.
- Successful MTP exits retain NextN flags, including successful stop handling
  that clears KV. This preserves graph reservation without preserving invalid
  pending hidden rows.
- Ordinary `decode`, `decode_one`, and `decode_multi` restore normal mode.
  Ordinary generation does so before sampling even with a fully cached prompt.
- Cancellation and exceptions during setup or generation restore normal mode
  through a guard. Existing cancellation and decode-failure paths clear KV.
- Reset and close discard the tracked mode with their contexts.

`perf()["n_mtp_graph_switches"]` counts transitions of the binding's graph mode,
not scheduler reservations. `perf_reset()` resets the counter; KV clearing
preserves it. A series of successful MTP requests switches on once. Switching
to ordinary generation switches off once.

The integration regression failed before the optimization with two mode
changes after the first MTP request, then passed with one. It compares greedy
output across repeated KV-cleared requests and ordinary/MTP transitions.
Additional tests check direct ordinary decode, context recreation, refused
callbacks, and callback exceptions. Existing differential tests cover cached
continuation, stop buffering, rollback, draft bounds, and device-state handles.

## Final measurements

Production builds before and after the change used identical upstream CUDA
libraries, GCC 15 portable extension builds, and the supplied 4B MTP checkpoint.
Each case used 128 output tokens, five timed repetitions after two full
warmups, four CPU threads, full GPU offload, `n_ctx=2048`,
`n_batch=n_ubatch=512`, and `n_rs_seq=2`.
TTFT is in milliseconds; total tokens/s includes prompt processing.

| Prompt | Mode | TTFT before | TTFT after | Tokens/s before | Tokens/s after |
| ---: | --- | ---: | ---: | ---: | ---: |
| 64 | Ordinary | 28.4 | 28.4 | 91.5 | 92.2 |
| 64 | MTP width 1 | 59.9 | 32.1 | 105.2 | 109.0 |
| 64 | MTP width 2 | 55.6 | 31.8 | 115.2 | 118.1 |
| 256 | Ordinary | 56.5 | 56.7 | 89.4 | 90.8 |
| 256 | MTP width 1 | 95.0 | 65.6 | 102.2 | 106.6 |
| 256 | MTP width 2 | 89.7 | 67.4 | 108.4 | 107.9 |

At width 2, warmed first-token latency decreased **42.7%** for the 64-token
prompt and **24.8%** for the 256-token prompt. Width 1 decreased 46.4% and
31.0%, respectively. All output token IDs and proposed/accepted draft counts
matched before and after, including warmup outputs.

The first MTP request at width 2 still took 61.6 ms for the 64-token prompt;
subsequent requests retained graph mode and recorded zero mode changes.
These are setup savings. Subsequent-token throughput did not consistently
improve: at width 2 it changed from 120.4 to 120.7 tokens/s for 64 prompt
tokens, and from 116.6 to 113.4 tokens/s for 256 prompt tokens.

Raw reports:

- [Before](benchmarks/2026-10-13-mtp-latency-before.json)
- [After](benchmarks/2026-10-13-mtp-latency-after.json)
- [Original narrowed workload](benchmarks/2026-10-13-mtp-latency-single-token-before.json)
- [Final narrowed one-token workload](benchmarks/2026-10-13-mtp-latency-single-token.json)

The original one-token diagnostic also turned green: ordinary TTFT was
26.6 ms and optimized MTP TTFT was 33.2 ms, for **6.6 ms** overhead instead of
29.7 ms. It passed the exploratory 10 ms limit with the same greedy token.

## Validation

Against the isolated upstream HEAD, compatible fixtures account for all
**358 unique tests: 357 passed, 1 skipped**. The remaining skip requires a
genuine MoE checkpoint. LFM2 ordinary inference plus Qwen MTP passed 334
tests; UnifiedLLM cases that rejected LFM2 passed with Qwen or the non-MTP
MiMo checkpoint. The MiMo adapter supplies its explicit `qwen3.5-small`
family and a 512-token context without changing the test bodies.

JUnit and the merged per-test observations are under `build/mtp-latency/`.
The initial incompatible-fixture failures remain in `head-main.xml`.
The merge gives precedence to a passing compatible-fixture run.

The installed b11552 baseline also passed **35 MTP tests** (two ordinary-model
cases skipped without that fixture) and **145 model-free tests**. GCC 15 and
Clang 21 HEAD wheels built and passed import/native-provenance checks.
Ruff lint/format, mypy across 28 files, warning-as-error Python compilation,
clang-format, and whitespace checks passed. Clang-tidy completed with four
existing warnings in state-position code outside this change: two pointer
typedef const warnings and two suggestions to use `std::max`.
New documentation passed Markdown lint and local-link checks; existing
README/API documents received local-link checks.

## Reproduction and scope

The benchmark uses the same isolated CUDA libraries at upstream commit
`cbe28bb3cb0758f0c566a9a3561766821aeb0748` for both binding versions.
See [the upstream validation guide](UPSTREAM_VALIDATION.md) for building that
prefix and checking native-library provenance.

```bash
rtk proxy env LD_LIBRARY_PATH=/path/to/prefix/lib \
  /path/to/venv/bin/python -m tools.benchmark_upstream \
  --model /path/to/Qwen3.5-4B-Q4_K_M-MTP.gguf \
  --prompt-tokens 64 256 --max-tokens 128 \
  --draft-max 1 2 --runs 5 --warmup 2 \
  --output /path/to/report.json
```

Schema version 2 retains warmup requests and records the native extension hash
and graph-mode counter. The first warmup in each MTP mode records its initial
setup; later requests show reservation reuse. Source hashes must refer to the
source used to build the loaded wheel. For the before measurements, the
benchmark runs from a source snapshot with the original C++ file and the same
benchmark tool as the after measurements.

Timing covers native streaming from fixed token IDs. It excludes loading,
tokenization, sampler creation, detokenization, and the capability probe that
creates the draft context. These measurements describe warmed consecutive MTP
requests on this checkpoint and hardware. The first request after startup,
context reset, or an ordinary/MTP transition can still pay reservation costs.
Alternating modes does not receive the consecutive-request benefit.
