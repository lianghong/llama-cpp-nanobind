"""Ad-hoc benchmark: speculative=False vs speculative=True on Qwen3.6-MoE.

Usage:
    LLAMA_MTP_TEST_MODEL=models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
        uv run python examples/bench_speculative.py

Reports tok/s for each path and the speedup ratio. Acceptance floor for
the implementation is >= 1.10x.

Each path is run N times and the **median** wall-clock is reported, so a
single warm-cache spike doesn't skew the result. The speculative path is
also swept across a few ``n_draft_max`` values so you can see where the
sweet spot sits for your model + hardware.
"""

from __future__ import annotations

import os
import statistics
import time

from llama_cpp import (
    Llama,
    LlamaConfig,
    SamplingParams,
)


PROMPT = (
    "Write a paragraph explaining how mixture-of-experts language models "
    "balance throughput and parameter count."
)
MAX_TOKENS = 256
RUNS = 3
DRAFT_MAX_SWEEP = (2, 3, 4, 6)


def _llm() -> Llama:
    path = os.environ.get(
        "LLAMA_MTP_TEST_MODEL",
        os.path.join("models", "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"),
    )
    cfg = LlamaConfig(
        model_path=path,
        n_ctx=2048,
        n_gpu_layers=-1,
        # On hybrid MTP models the speculative loop rolls back rejected drafts
        # via the draft context's recurrent-state slots, so n_rs_seq must be
        # >= the largest n_draft_max we sweep — otherwise the reject trim fails
        # ("kv_cache_seq_rm (tgt reject trim) failed"). Default is 2; the sweep
        # goes up to max(DRAFT_MAX_SWEEP), so size it accordingly.
        n_rs_seq=max(2, *DRAFT_MAX_SWEEP),
        verbose=False,
    )
    return Llama(path, config=cfg)


def _time_run(llm: Llama, *, speculative: bool, n_draft_max: int = 2) -> tuple[int, float]:
    sp = SamplingParams(seed=0, temperature=0.0, n_draft_max=n_draft_max)
    # Reset KV + perf counters between runs so each run starts cold-ish.
    llm.reset()
    llm.perf_reset()
    t0 = time.perf_counter()
    out = llm.generate(
        PROMPT,
        max_tokens=MAX_TOKENS,
        sampling=sp,
        speculative=speculative,
    )
    elapsed = time.perf_counter() - t0
    if not isinstance(out, str):
        raise RuntimeError(f"expected str from generate(), got {type(out).__name__}")
    n_tok = len(llm.tokenize(out, add_special=False))
    return n_tok, elapsed


def _median_tps(
    llm: Llama, *, speculative: bool, n_draft_max: int = 2, runs: int = RUNS
) -> tuple[int, float, float]:
    """Return (tokens, median_elapsed, median_tps)."""
    results = [
        _time_run(llm, speculative=speculative, n_draft_max=n_draft_max)
        for _ in range(runs)
    ]
    n_tok = results[-1][0]  # token count is deterministic at temperature=0
    median_elapsed = statistics.median(t for _, t in results)
    return n_tok, median_elapsed, n_tok / median_elapsed


def main() -> None:
    llm = _llm()
    try:
        # Warmup once to pay JIT/CUDA-graph cost outside timed runs.
        llm.generate("warmup", max_tokens=4, sampling=SamplingParams(seed=0, temperature=0.0))

        n_base, t_base, base_tps = _median_tps(llm, speculative=False)
        print(f"baseline:    {n_base:4d} tok in {t_base:6.2f}s = {base_tps:6.1f} tok/s  (median of {RUNS})")

        for nd in DRAFT_MAX_SWEEP:
            n_spec, t_spec, spec_tps = _median_tps(llm, speculative=True, n_draft_max=nd)
            print(
                f"spec n_dft={nd}: {n_spec:4d} tok in {t_spec:6.2f}s = {spec_tps:6.1f} tok/s  "
                f"(speedup {spec_tps / base_tps:.2f}x; floor 1.10x)"
            )
    finally:
        llm.close()


if __name__ == "__main__":
    main()
