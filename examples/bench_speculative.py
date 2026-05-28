"""Ad-hoc benchmark: speculative=False vs speculative=True on Qwen3.6-MoE.

Usage:
    LLAMA_MTP_TEST_MODEL=models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
        uv run python examples/bench_speculative.py

Reports tok/s for each path and the speedup ratio. Acceptance floor for
the implementation is >= 1.10x.
"""

from __future__ import annotations

import os
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


def _llm() -> Llama:
    path = os.environ.get(
        "LLAMA_MTP_TEST_MODEL",
        os.path.join("models", "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"),
    )
    cfg = LlamaConfig(
        model_path=path,
        n_ctx=2048,
        n_gpu_layers=-1,
        verbose=False,
    )
    return Llama(path, config=cfg)


def _time_run(llm: Llama, *, speculative: bool) -> tuple[int, float]:
    sp = SamplingParams(seed=0, temperature=0.0, n_draft_max=2)
    # Warmup
    llm.generate("warmup", max_tokens=4, sampling=sp, speculative=speculative)
    t0 = time.perf_counter()
    out = llm.generate(
        PROMPT,
        max_tokens=MAX_TOKENS,
        sampling=sp,
        speculative=speculative,
    )
    elapsed = time.perf_counter() - t0
    assert isinstance(out, str)
    n_tok = len(llm.tokenize(out, add_special=False))
    return n_tok, elapsed


def main() -> None:
    llm = _llm()
    try:
        n_base, t_base = _time_run(llm, speculative=False)
        n_spec, t_spec = _time_run(llm, speculative=True)
    finally:
        llm.close()

    base_tps = n_base / t_base
    spec_tps = n_spec / t_spec
    print(f"baseline:    {n_base:4d} tok in {t_base:6.2f}s = {base_tps:6.1f} tok/s")
    print(f"speculative: {n_spec:4d} tok in {t_spec:6.2f}s = {spec_tps:6.1f} tok/s")
    print(f"speedup:     {spec_tps / base_tps:.2f}x  (floor: 1.10x)")


if __name__ == "__main__":
    main()
