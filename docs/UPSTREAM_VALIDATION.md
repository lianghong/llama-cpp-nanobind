# Upstream compatibility and benchmarks

The validated baseline is llama.cpp
[`23b0202a189c44a54625aadcb37a946dd1d6278d`](https://github.com/ggml-org/llama.cpp/commit/23b0202a189c44a54625aadcb37a946dd1d6278d)
(b11552). Keep it as the reference while checking newer commits.

The subsequent [MTP latency investigation](MTP_LATENCY-2026-10-13.md)
compares binding versions on identical upstream HEAD libraries and documents
graph reservation reuse, additional regressions, and updated measurements.

## Continuous compatibility checks

[The workflow](../.github/workflows/upstream-compatibility.yml) runs on pushes,
pull requests, Monday schedules, and manual dispatch. Its independent lanes
build the baseline and upstream `master`; manual dispatch can replace `master`
with a specific upstream ref.

Each lane builds shared CPU libraries, installs the required supplemental
headers, builds a portable Python 3.14 wheel, and checks the loaded native
runtime before running tests. It does not reuse native build caches.
Native-runtime JSON, JUnit results, and upstream CMake configuration are
retained as workflow artifacts.
The commit is authoritative; shallow checkouts can report build number 1.
CMake locates nanobind through the selected Python interpreter, including
isolated environments created by current build backends.

The runtime gate verifies the commit exported by the loaded libllama-common
and that every loaded llama/ggml library belongs to the requested prefix.
This catches runtime loader mistakes that a successful compile cannot detect.
Optional CUDA/Metal/BLAS libraries are searched beside the selected ggml
library, preventing fallback to another system installation.

Hosted CI deliberately supplies no model. Model-dependent tests skip;
import and model-free tests run against real native libraries. This checks
compatibility, while local model runs check inference correctness.

## Isolated local validation

Use a fresh build directory, installation prefix, and virtual environment
for each upstream revision. Headers and libraries must come from the same
build; rebuilding only the Python extension is insufficient.

The following Linux example installs under the ignored project `build/`
directory. It does not require a system-wide installation.

```bash
git clone https://github.com/ggml-org/llama.cpp.git build/upstream-source
git -C build/upstream-source checkout cbe28bb3cb0758f0c566a9a3561766821aeb0748

cmake -S build/upstream-source -B build/upstream-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/build/upstream-prefix" \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_COMMON=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=OFF \
  -DLLAMA_OPENSSL=OFF -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build/upstream-build --parallel 4
cmake --install build/upstream-build
install -m 644 build/upstream-source/src/llama-ext.h \
  build/upstream-prefix/include/
install -d build/upstream-prefix/include/llama
install -m 644 build/upstream-source/common/{common,speculative}.h \
  build/upstream-prefix/include/llama/

uv venv --python 3.14 build/upstream-venv
validation_prefix="$PWD/build/upstream-prefix"
CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$validation_prefix -DLLAMA_PORTABLE=ON" \
  uv pip install --python build/upstream-venv/bin/python . \
  pytest pytest-asyncio fastapi gguf

LD_LIBRARY_PATH="$validation_prefix/lib" \
  build/upstream-venv/bin/python tools/check_runtime.py \
  --prefix build/upstream-prefix \
  --expected-commit "$(git -C build/upstream-source rev-parse HEAD)" \
  --output build/runtime.json
LD_LIBRARY_PATH="$validation_prefix/lib" \
  build/upstream-venv/bin/python -m pytest -q --junitxml=build/model-free.xml
```

The final pytest command uses the fixture variables, or their default paths,
if provided. For a strictly model-free run, set `LLAMA_TEST_MODEL`,
`LLAMA_MTP_TEST_MODEL`, and `LLAMA_MOE_TEST_MODEL` to nonexistent paths.

For model-backed runs, `LLAMA_TEST_MODEL` should point to a non-MTP checkpoint
supported by UnifiedLLM; `LLAMA_MTP_TEST_MODEL` should point to a checkpoint
declaring MTP layers. With those compatible fixtures, run the entire suite.
LFM2 is suitable for low-level inference tests but is not a supported
UnifiedLLM family. See the [fixture-specific validation notes](CHANGELOG-2026-10-13.md#validation-scope)
when using the prepared LFM2/Qwen/MiMo models.

## Reproducible benchmarks

[The benchmark tool](../tools/benchmark_upstream.py) accepts repeated model
arguments, exact prompt lengths, generation limits, draft widths, and thread
counts. Run it once per isolated native installation with identical arguments:

```bash
LD_LIBRARY_PATH="$PWD/build/upstream-prefix/lib" \
  build/upstream-venv/bin/python -m tools.benchmark_upstream \
  --model /path/to/LFM2.5-350M-Q4_K_M.gguf \
  --model /path/to/Qwen3.5-4B-Q4_K_M-MTP.gguf \
  --prompt-tokens 64 256 --max-tokens 128 \
  --draft-max 1 2 --runs 3 --warmup 1 \
  --n-ctx 2048 --n-batch 512 --threads 4 \
  --output build/benchmark-head.json
```

The JSON report records:

- Native commit, compiler, loaded library paths, Python, GPU, and system info.
- Model and binding-source SHA-256 hashes, prompt-token hashes, and config.
- Native extension SHA-256 and warmup request observations in schema version 2.
- Every generated token ID and repetition's wall time, plus median rates.
- Standalone ordinary prefill rate, first-token latency, and subsequent
  generation rate.
- Proposed/accepted draft counts, verification rounds, and acceptance rate.
- MTP graph-mode changes when the loaded binding exposes the counter.
- Host RSS and NVIDIA process memory sampled after a request.

Each mode warms up with the full workload before timed repetitions.
Warmup requests are retained separately; medians use only timed repetitions.
Prompt KV is cleared before every request. Contexts close between ordinary
and draft modes, so ordinary memory samples exclude the MTP draft context.
Models without MTP still receive ordinary benchmarks.

Timing covers native streaming from fixed token IDs. It excludes model load,
tokenization, sampler construction, and detokenization. First-token latency
includes prefill and MTP preparation; subsequent generation time includes
verification and final decode. Standalone prefill includes logits retrieval
to synchronize the computation. Memory readings are point samples, not peaks.

The tool checks exact greedy token agreement across modes and repetitions.
It saves the report and exits with an error on mismatch. Compare token arrays
between revision reports too: upstream numerical changes may affect greedy
output. Native `t_p_eval_ms` is not used to estimate MTP prompt throughput,
because upstream counts multi-token verification batches as prompt evaluation.

`Llama.perf()` now includes `n_drafted`, `n_accepted`, and `n_verify`.
These counters accumulate until `perf_reset()`; clearing KV does not reset
them. Acceptance rate is `n_accepted / n_drafted`, with no rate when no tokens
were drafted.
`n_mtp_graph_switches` similarly counts changes to the binding's MTP graph mode,
rather than actual scheduler reservations.

Run performance comparisons after builds/tests finish, on an otherwise idle
GPU, using the same backend settings. A different CUDA/BLAS configuration is
a configuration comparison, not a controlled revision comparison.

## Local results on 2026-10-13

The baseline CPU lane was reproduced locally with a fresh upstream build,
prefix, virtual environment, and portable wheel: **145 passed, 206 skipped**.
The skips are model-dependent tests, as intended for hosted CI. Actionlint
validated the workflow; no hosted workflow was dispatched during this work.

Pinned public HEAD **`cbe28bb3cb0758f0c566a9a3561766821aeb0748`** was built
with GCC 15, CUDA 13.3, and architecture `86-real`, then tested through its
own wheel and virtual environment. The runtime gate confirmed all llama/ggml
libraries came from its private prefix. The system baseline installation
was retained.
The final extension also passed a Clang 21 portable wheel build and import
against those GCC-built HEAD libraries. Model-backed tests used the GCC wheel.
Ruff, mypy across 28 files, clang-format, warning-as-error Python compilation,
Markdown QA, and whitespace checks passed.

Across all **351 unique tests**, compatible fixture runs yielded
**350 passed, 1 skipped**:

- LFM2 ordinary inference plus Qwen MTP: 288 passed, 10 skipped, 2 deselected.
  One additional streaming test received a naturally complete two-chunk LFM2
  response and could not exercise its three-chunk early-break assertion.
- Qwen for UnifiedLLM, lifecycle checks, and the three early-break cases:
  60 passed. This resolved the nine family-related skips and streaming failure.
- MiMo/Qwen 9B for two tests requiring a non-MTP model: 2 passed, using an
  explicit `qwen3.5-small` preset and 512-token context through a temporary
  pytest adapter. The test bodies were unchanged.
- The remaining skip requires a genuine MoE checkpoint.

JUnit files and the merged per-test observations are in
`build/validation-head/`. A passing rerun takes precedence in the aggregate;
the initial failure remains visible in `main.xml`.

### Qwen warmed benchmark

RTX 3080, full GPU offload, 128 output tokens, exact 64/256-token prompts,
three timed repetitions after full-workload warmup, four CPU threads,
`n_ctx=2048`, `n_batch=n_ubatch=512`, `n_rs_seq=2`, fixed greedy sampling.

| Revision | Prompt | Draft | Tok/s | First ms | Accept % |
| --- | ---: | ---: | ---: | ---: | ---: |
| `23b0202` | 64 | 0 | 93.5 | 27.8 | n/a |
| `23b0202` | 64 | 1 | 108.5 | 60.4 | 84.1 |
| `23b0202` | 64 | 2 | 121.1 | 55.2 | 86.0 |
| `23b0202` | 256 | 0 | 92.6 | 55.3 | n/a |
| `23b0202` | 256 | 1 | 103.2 | 92.5 | 84.1 |
| `23b0202` | 256 | 2 | 112.7 | 88.2 | 75.2 |
| `cbe28bb3` | 64 | 0 | 93.0 | 28.5 | n/a |
| `cbe28bb3` | 64 | 1 | 106.0 | 59.6 | 84.1 |
| `cbe28bb3` | 64 | 2 | 116.5 | 55.0 | 86.0 |
| `cbe28bb3` | 256 | 0 | 91.4 | 56.1 | n/a |
| `cbe28bb3` | 256 | 1 | 103.5 | 90.8 | 84.1 |
| `cbe28bb3` | 256 | 2 | 108.7 | 88.4 | 75.2 |

Draft width 2 improved HEAD throughput by about 25% for the 64-token prompt
and 19% for the 256-token prompt relative to ordinary decoding. First-token
latency increased, and process GPU memory rose from about 3696 MiB to
3890 MiB. All generated token IDs matched across repetitions, modes, and
both revisions.

Ordinary prefill on HEAD measured about 2970/4911 tokens/s for Qwen's
64/256-token prompts. LFM2 prefill measured about 16040/29995 tokens/s;
generation produced 128 tokens for its 64-token prompt but naturally ended
after 7 tokens for the 256-token prompt. Those generation rates are different
workloads and should not be compared as sustained throughput.

The installed baseline includes BLAS and `NO_PEER_COPY`; the isolated HEAD
build uses upstream defaults without BLAS. These runs establish reference
measurements, not a controlled causal measurement of the revision change.
They do not justify replacing the validated baseline or enabling MTP for
every request.

Raw records with model/source hashes, per-run tokens, latency, counters,
and memory samples are archived in
[baseline JSON](benchmarks/2026-10-13-baseline.json) and
[HEAD JSON](benchmarks/2026-10-13-head.json).
