"""Ad-hoc benchmark: speculative=False vs speculative=True on Qwen3.6-MoE.

Usage:
    LLAMA_MTP_TEST_MODEL=models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
        uv run python examples/bench_speculative.py

Reports end-to-end tok/s (including prefill), speedup, and greedy-output
agreement. Speedup depends on the model, prompt, draft width, and hardware.

Each path is run N times and the **median** wall-clock is reported, so a
single warm-cache spike doesn't skew the result. The speculative path is
also swept across a few ``n_draft_max`` values so you can see where the
sweet spot sits for your model + hardware.
"""

from __future__ import annotations

import argparse
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


def _llm(path: str, draft_max: tuple[int, ...]) -> Llama:
    cfg = LlamaConfig(
        model_path=path,
        n_ctx=2048,
        n_gpu_layers=-1,
        # On hybrid MTP models the speculative loop rolls back rejected drafts
        # via the target context's recurrent-state slots, so n_rs_seq must be
        # >= the largest n_draft_max we sweep — otherwise the reject trim fails
        # ("kv_cache_seq_rm (tgt reject trim) failed"). Default is 2; the sweep
        # goes up to max(DRAFT_MAX_SWEEP), so size it accordingly.
        n_rs_seq=max(2, *draft_max),
        verbose=False,
    )
    return Llama(path, config=cfg)


def _time_run(
    llm: Llama, *, speculative: bool, n_draft_max: int = 2, max_tokens: int = MAX_TOKENS
) -> tuple[int, float, str]:
    sp = SamplingParams(seed=0, temperature=0.0, n_draft_max=n_draft_max)
    # Clear KV while retaining warmed contexts and CUDA kernels.
    llm.kv_cache_clear()
    llm.perf_reset()
    t0 = time.perf_counter()
    # Use the token generation helper to count actual outputs, including when
    # an EOG clears the cache. Include tokenization and detokenization in time.
    tokens = llm._generate_from_tokens(
        llm.tokenize(PROMPT, add_special=False),
        max_tokens=max_tokens,
        sampler=llm._build_sampler(sp),
        speculative=speculative,
        n_draft_max=n_draft_max,
    )
    out = llm.detokenize(tokens)
    elapsed = time.perf_counter() - t0
    if not isinstance(out, str):
        raise RuntimeError(f"expected str from generate(), got {type(out).__name__}")
    n_tok = len(tokens)
    return n_tok, elapsed, out


def _median_tps(
    llm: Llama,
    *,
    speculative: bool,
    n_draft_max: int = 2,
    runs: int = RUNS,
    max_tokens: int = MAX_TOKENS,
) -> tuple[int, float, float, str]:
    """Return (tokens, median_elapsed, median_tps)."""
    results = [
        _time_run(
            llm, speculative=speculative, n_draft_max=n_draft_max, max_tokens=max_tokens
        )
        for _ in range(runs)
    ]
    n_tok = results[-1][0]  # token count is deterministic at temperature=0
    if any(text != results[0][2] for _, _, text in results):
        raise RuntimeError("greedy output changed between benchmark repetitions")
    median_elapsed = statistics.median(t for _, t, _ in results)
    return n_tok, median_elapsed, n_tok / median_elapsed, results[0][2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "LLAMA_MTP_TEST_MODEL", "models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--draft-max", type=int, nargs="+", default=DRAFT_MAX_SWEEP)
    args = parser.parse_args()
    if (
        args.runs < 1
        or args.max_tokens < 1
        or any(not 1 <= n <= 8 for n in args.draft_max)
    ):
        parser.error(
            "runs and max-tokens must be positive; draft-max must be in [1, 8]"
        )
    llm = _llm(args.model, tuple(args.draft_max))
    try:
        # Warmup once to pay JIT/CUDA-graph cost outside timed runs.
        _time_run(llm, speculative=False, max_tokens=8)

        n_base, t_base, base_tps, baseline = _median_tps(
            llm, speculative=False, runs=args.runs, max_tokens=args.max_tokens
        )
        print(
            f"baseline:    {n_base:4d} tok in {t_base:6.2f}s = {base_tps:6.1f} tok/s  (median of {args.runs})"
        )

        for nd in args.draft_max:
            _time_run(llm, speculative=True, n_draft_max=nd, max_tokens=8)
            n_spec, t_spec, spec_tps, text = _median_tps(
                llm,
                speculative=True,
                n_draft_max=nd,
                runs=args.runs,
                max_tokens=args.max_tokens,
            )
            print(
                f"spec n_dft={nd}: {n_spec:4d} tok in {t_spec:6.2f}s = {spec_tps:6.1f} tok/s  "
                f"(speedup {spec_tps / base_tps:.2f}x; greedy match: {text == baseline})"
            )
    finally:
        llm.close()


if __name__ == "__main__":
    main()
