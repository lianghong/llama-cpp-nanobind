# MTP draft-width tuning — 2026-10-13

Keep **draft width 2** as the starting point on this checkpoint and hardware.
Wider drafts do not give a consistent enough advantage to change the default.
Use ordinary decoding when its exact greedy output is required.
Cache-prefix stability matters more to later-turn latency than small width
differences.

This study follows [the first-token optimization](MTP_FIRST_TOKEN-2026-10-13.md).
It compares ordinary generation and MTP draft widths 1–4 with the supplied
Qwen3.5-4B-Q4_K_M-MTP checkpoint, including longer cached conversations.

## Native results

Median tokens/s includes fresh prefill and complete 256-token generation:

| Draft width | 512 prompt tokens | 2048 prompt tokens | 4096 prompt tokens |
| ---: | ---: | ---: | ---: |
| Ordinary | 92.1 | 82.6 | 73.2 |
| 1 | 112.8 | 99.1 | 84.5 |
| 2 | 53.7* | 108.9 | 89.6 |
| 3 | 127.0 | 110.4 | 91.9 |
| 4 | 116.6† | 112.9 | 93.6 |

The width-2 512-token block (*) was unusually slow despite unchanged token
IDs and 100% draft acceptance. Its five timed requests ranged from 48.5 to
69.6 tokens/s. Follow-up controls below investigate this variation; the
original measurements remain in the raw report.

The width-4 512-token fixture (†) differed from ordinary greedy output at
token index 120 on every warmup and timed run. Widths 1–3 matched ordinary
output on every native fixture; width 4 matched at 2048 and 4096 tokens.

Widths 1–3 reached 100% draft acceptance in these repeated-text native
fixtures. Width 4 reached 70.7% at 512 tokens and 100% on longer prompts.
These unusually predictable outputs favor wider drafting and should not be
treated as representative chat acceptance.

Fresh-prompt first-token P50/P95, in milliseconds:

| Prompt tokens | Ordinary P50/P95 | Width 2 P50/P95 | Width 3 P50/P95 |
| ---: | ---: | ---: | ---: |
| 512 | 94.2 / 95.6 | 124.5 / 132.8* | 97.0 / 97.1 |
| 2048 | 365.6 / 366.7 | 423.1 / 424.5 | 420.8 / 422.6 |
| 4096 | 731.0 / 737.6 | 854.7 / 858.9 | 856.9 / 859.9 |

At 2048/4096 tokens, width 2 raised throughput 31.8%/22.4%, while adding
57.5/123.7 ms to first-token latency. Deferring the last draft-prefill chunk
still leaves earlier chunks on the first-token path.

## Follow-up controls

The anomalous 512-token width-2 block did not reproduce. Separate native
controls used the same 512-token prompt, 256 output tokens, 8K allocation,
model, runtime, and extension. Each received two warmups, followed by five
timed runs for unmonitored width 2 and four-snapshot monitored width 2;
the ordinary, width-1, width-3, and two-snapshot monitored controls had
three timed runs.

| Mode | Recurrent snapshots | NVML polling | Median tokens/s |
| --- | ---: | --- | ---: |
| Ordinary | 4 | Off | 91.9 |
| Width 2 | 4 | Off | 125.7 |
| Width 2 | 4 | On | 124.6 |
| Width 1 | 4 | On | 113.0 |
| Width 3 | 4 | On | 126.9 |
| Width 2 | 2 | Off | 125.0 |
| Width 2 | 2 | On | 126.8 |

Every warmup and timed output matched ordinary token IDs. The four-snapshot
width-2 runs ranged from 122.8–126.6 tokens/s without polling and
123.6–125.5 with polling. First-token medians were 96.2/96.8 ms.
These controls do not support a large persistent polling penalty or a
snapshot-capacity explanation for the original slowdown. Its precise cause
remains unproven; the initial slow block is retained rather than replaced.

In the monitored controls, width 3 improved over width 2 by only 1.8%.
At 2048/4096 prompt tokens, the main sweep's width-3 advantage was
1.4%/2.6%. Those differences do not justify a new global default.

The native width-2 memory control observed **4068 MiB with two snapshots
versus 4168 MiB with four**, a 100 MiB reduction. Throughput remained
similar, and exact outputs matched. This capacity comparison was measured
on the 512-token native fixture; the main long-context/chat matrix used four.

[Control measurements and complete experiment source](benchmarks/2026-10-13-mtp-width-controls.json)
are archived separately, including GPU state before and after each block.

## Streaming chat results

These historical chat measurements encoded template role markers as ordinary
text under the then-current default. The
[cache study](MTP_CACHE-2026-10-13.md) corrects that behavior and records new
paired measurements. Native width results above are unaffected; the chat table
below describes the archived protocol.

Conversation throughput is the median of exact total generated IDs divided
by summed duration of all three requests. Output lengths vary because EOG
can occur before the 256-token budget.

| Draft width | 512 initial tokens | 2048 initial tokens | 4096 initial tokens |
| ---: | ---: | ---: | ---: |
| Ordinary | 92.0 | 86.1 | 76.0 |
| 1 | 103.1 | 88.0 | 82.0 |
| 2 | 112.9 | 98.5 | 82.8 |
| 3 | 103.6 | 81.1 | 82.1 |
| 4 | 97.9 | 83.5 | 78.0 |

Every setting was stable across its own warmup and timed runs, but **all
12 MTP chat settings differed from ordinary output**. The table describes
observed trajectories; it does not isolate a speedup for identical work.
Widths 1 and 2 did produce identical token IDs, response strings, and prompt
hashes across the 512-token conversations. Within that matching pair,
width 2 was 9.5% faster over the complete conversation.

Width-2 first-text P50/P95 for turns 1, 2, and 3:

| Initial tokens | Turn 1 (ms) | Turn 2 (ms) | Turn 3 (ms) |
| ---: | ---: | ---: | ---: |
| 512 | 107.6 / 108.4 | 177.9 / 178.6 | 36.5 / 37.3 |
| 2048 | 448.9 / 456.7 | 512.3 / 513.3 | 54.3 / 55.3 |
| 4096 | 903.5 / 905.2 | 959.2 / 960.4 | 73.4 / 74.9 |

For width 2, turn 2 fully re-prefilled all 737/2269/4277 prompt tokens;
turn 3 processed only its 42-token suffix at every initial length.
The initial chat allocation remains 8K throughout.

Cache reuse changed the observed latency more than adjacent draft widths.
At 2048 initial tokens, width 2 reused 2525 cached tokens on turn 3 and
processed 42 new tokens, reaching first text in 54.3 ms. Width 3 matched
2520 prefix tokens but could not retain that prefix; it re-prefilled all
2561 tokens and took 582.2 ms. Different outputs changed the formatted
history, so this is a measured cache-path contrast rather than an isolated
causal comparison of widths.

Chat acceptance was lower than native acceptance. Across contexts and
turns, proposed-token acceptance ranged from 62.8–88.1% at width 1,
43.3–82.3% at width 2, 35.5–57.1% at width 3, and 28.9–54.4% at width 4.
A lower accepted fraction alone does not imply lower throughput: a wider
round can still accept more total tokens.

## Memory results

With the fixed 8K allocation and four recurrent snapshots, timed native
requests observed 3990–4006 MiB ordinary GPU process memory and
4174–4178 MiB MTP memory. Chat observed 4002–4006 MiB ordinary and
4176–4178 MiB MTP. Wider drafts did not produce a meaningful memory
separation in this matrix.

Across all 420 warmup and timed requests, the largest observation gap was
50.2 ms despite the requested 10 ms interval. These are sampled peaks,
with the limitations described below.

Raw measurements: [complete width sweep](benchmarks/2026-10-13-mtp-width-sweep.json).

## Method

Both native fixed-token prompts and three-turn public streaming chat use
512, 2,048, and 4,096 initial prompt tokens. The first chat prompt contains
synthetic deployment notes and a memory/throughput question; subsequent turns
ask about quantization and conversation length. Reports include the complete
first message and every generated token ID and response.

All modes use the same model hash and frozen upstream runtime:
`cbe28bb3cb0758f0c566a9a3561766821aeb0748`, b11577, GCC 15 portable binding,
CUDA 13.3, RTX 3080 with 20 GiB VRAM. Runtime provenance and loaded library
paths appear in the report. The production extension is unchanged from the
first-token phase.

Configuration is constant: `n_ctx=8192`, `n_batch=n_ubatch=512`, `n_rs_seq=4`,
four CPU threads, full GPU offload, F16 KV caches, greedy sampling, and a
256-token output budget. Prompt length varies within the fixed 8K allocation;
these results do not compare differently sized context allocations.

Each mode receives two warmup runs and five measured runs per workload.
Model loading and initial MTP capability probing are outside timing. Mode
order rotates across prompt lengths to reduce consistent ordering bias.
Runs are sequential, without concurrent builds or model tests. Desktop GPU
activity remains present, so close timing differences should be treated
cautiously.
P95 is an interpolated percentile from five observations per turn, suitable
for this comparison rather than a production tail-latency guarantee.

Native throughput counts exact callback token IDs divided by complete request
duration, including prefill. Native first-token timing excludes Python text
processing and sampler creation. Chat uses the public streamed API, including
formatting, tokenization, sampler/thread creation, queue delivery, and text
decoding. Exact chat IDs are observed at the existing cache-commit method,
including when EOG clears retained KV. First-text timing excludes empty chunks.

Chat turns 2 and 3 request `reset_kv_cache=False, cache_prompt=True`. Their
actual prompt lengths, matching prefixes, retained cache sizes, and MTP
prefill counters establish whether reuse occurred. Responses that end in
EOG can require full prefill on the next turn. Greedy equality checks include
warmups, exact token IDs, complete responses, and conversation prompt hashes.

Peak GPU process memory is sampled through NVML at a requested 10 ms interval
during each inference request. The sampler sums this process's compute memory
across physical GPUs and records observation counts and actual maximum gaps.
These observed peaks are lower bounds: allocations shorter than the sampling
gap can be missed. They exclude model-loading peaks and do not include other
desktop processes. A live smoke check matched `nvidia-smi` process memory.

## Exact-output limits

A 32-token pilot exposed a repeatable width-3 difference on a cached second
turn. Its first turn's generated IDs and second turn's prompt were identical
to the ordinary control, so constructed chat history did not explain it.

Replaying the same first MTP turn and using ordinary cached continuation
matched the ordinary control. MTP width-3 continuation chose a different
fifth token. Fresh full-prefix ordinary and MTP generation instead matched
that latter trajectory. The same five configurations produced identical
results in the previous binding, before first-token deferral.

A four-token continuation followed by a single-token decode produced
identical next-token logits across modes; the leading two candidates were
only 0.043 logit units apart. The observed difference is sensitive to prefill
and verification batching. These probes do not establish universal greedy
equality or identify a corrected upstream defect. Raw diagnostics remain in
the ignored `build/mtp-sweep/debug/` directory.

The benchmark preserves any exact-output mismatch and exits nonzero after
saving the complete report. Performance figures from divergent trajectories
cannot qualify a setting as interchangeable with ordinary greedy decoding.

## Applying settings

Keep a draft width fixed through a cached conversation. Changing
`n_draft_max` rebuilds the common driver and clears target KV in
`Context::prepare_speculative`; the next generation processes the full prompt.
That cost can outweigh small throughput differences between adjacent widths.
The data do not justify adding automatic width changes.

For hybrid/recurrent targets, set `n_rs_seq >= n_draft_max`. The sweep uses
four snapshots for every mode to isolate draft width; memory figures therefore
apply to that allocation. Reducing snapshot capacity requires its own
measurement rather than assuming the same memory usage. Width 2 can use the
existing `n_rs_seq=2` default; the native control above measures that reduction.

Use ordinary decoding when reproducing its exact greedy trajectory is a
requirement. The speculative target still samples greedily, but batched
evaluation can choose differently near a tie. This study measures equality
on particular fixtures; it cannot guarantee it on arbitrary inputs.

## Validation

All **71 unique MTP regression cases passed** across supplied compatible
fixtures. The main Qwen run passed 69 cases and skipped two tests requiring
an ordinary model; those two passed with LFM2.5-350M.

Coverage now includes callback refusal and exceptions, first-token stops,
multi-token stop buffering, and public-stream cancellation at widths 1–4.
A further 12 cases exercise multi-token stops and callback cancellation at
every width and each long prompt length, checking cleared KV and repeatable
recovery. Existing grammar, continuation, rollback, one-token budgets,
context boundaries, and device-state checks also passed.

Evidence is in `build/mtp-sweep/mtp-regressions.xml` and
`build/mtp-sweep/ordinary-fixture.xml`. This phase changes benchmark tooling
and test coverage; it does not change the production generation algorithm
or the default draft width.

The original fixed-prompt benchmark underestimated the number of repetitions
needed for 512 tokens because repeated text tokenized more compactly than
one standalone sentence. It now expands until it can return the requested
exact length; the real 512-token pilot reproduced and verified that fix.

## Reproduction

Use matching headers, native libraries, and a previously built wheel as
described in [upstream validation](UPSTREAM_VALIDATION.md).
The memory sampler requires Linux and NVIDIA NVML; no Python NVML package
is needed.

```bash
rtk proxy env LD_LIBRARY_PATH=/path/to/prefix/lib \
  /path/to/venv/bin/python -m tools.benchmark_mtp_sweep \
  --model /path/to/Qwen3.5-4B-Q4_K_M-MTP.gguf \
  --prompt-tokens 512 2048 4096 --draft-max 1 2 3 4 \
  --runs 5 --warmup 2 --max-tokens 256 --n-ctx 8192 \
  --output /path/to/mtp-width-sweep.json
```

Use `--scope native` or `--scope chat` to isolate a workload.
Reports are saved after each completed mode, with `completed=false` during
the sweep and `completed=true` only after all requested cases.
`exact_greedy_pass` describes the final equality verdict, separately from
whether the measurement completed.

Repeat on the deployment model and hardware before treating the results as
recommended settings for another application.
