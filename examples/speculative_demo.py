#!/usr/bin/env python3
"""Draft-MTP speculative decoding via UnifiedLLM.

Speculative decoding uses the model's own Multi-Token Prediction (MTP) graph as
an internal *draft* context to propose several tokens per step, which the
user-facing context then verifies in one batch. On MTP-capable checkpoints
(e.g. Qwen3.6-MoE) this is a measurable tok/s win with **identical** greedy
output to the per-token path.

With ``UnifiedLLM`` it is a single constructor argument:

    UnifiedLLM(model_path, speculative="auto")   # on iff model exposes MTP
    UnifiedLLM(model_path, speculative=True)      # force on (raises if no MTP)
    UnifiedLLM(model_path, speculative=False)     # off

``"auto"`` (the default) probes ``Context.supports_speculative_mtp()`` after the
model loads, so a non-MTP model silently runs the normal path — the same code
works everywhere. ``n_draft_max`` (range [1, 8]) tunes how many tokens are
drafted per verify step.

Usage:
    # MTP model: speculative auto-enables
    python examples/speculative_demo.py --model models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf

    # Any model: "auto" just runs the normal path if there's no MTP graph
    python examples/speculative_demo.py --model models/Qwen3.5-9B-Q4_K_M.gguf

Preconditions for speculative to engage (see docs/CHANGELOG-v0.6.0.md):
    - The checkpoint exposes an MTP graph (nextn_predict_layers > 0).
    - Embeddings mode is off (UnifiedLLM satisfies this).
Greedy (temperature=0.0) output is bit-exact vs. the non-speculative path.
"""

from __future__ import annotations

import argparse
import os
import time

from llama_cpp.unified import UnifiedLLM
from llama_cpp.unified import UnsupportedModelError


SEPARATOR = "=" * 70


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draft-MTP speculative decoding via UnifiedLLM"
    )
    parser.add_argument(
        "--model",
        default="models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf",
        help="Path to a GGUF model (MTP-capable to engage speculative)",
    )
    parser.add_argument(
        "--prompt",
        default="Explain mixture-of-experts language models in two sentences.",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument(
        "--n-draft-max",
        type=int,
        default=2,
        help="Draft tokens proposed per verify step (range [1, 8])",
    )
    parser.add_argument(
        "--speculative",
        choices=("auto", "on", "off"),
        default="auto",
        help="auto (default) = on iff MTP graph present; on = force; off = disable",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(
            f"Model file not found: {args.model}\n"
            "Pass --model with an MTP-capable GGUF you have, e.g. "
            "--model models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
        )

    # Map the CLI choice to UnifiedLLM's bool | "auto" contract.
    spec_arg: bool | str = {"auto": "auto", "on": True, "off": False}[args.speculative]

    print(f"Loading model: {args.model}")
    try:
        with UnifiedLLM(
            args.model,
            n_ctx=args.ctx,
            speculative=spec_arg,
            n_draft_max=args.n_draft_max,
            verbose=args.verbose,
        ) as llm:
            print(f"Model family:        {llm.family.name}")
            print(f"Speculative engaged: {llm.speculative_enabled}")
            if not llm.speculative_enabled:
                print(
                    "  (model has no MTP graph, or speculative was disabled — "
                    "running the normal per-token path)"
                )
            print(f"n_draft_max:         {args.n_draft_max}")
            print(SEPARATOR)
            print(f"Prompt: {args.prompt}")

            start = time.perf_counter()
            answer = llm.generate(args.prompt, max_tokens=args.max_tokens)
            elapsed = time.perf_counter() - start

            print(f"\nAnswer:\n{answer}")
            print(SEPARATOR)
            print(f"Elapsed: {elapsed:.2f}s")
            print(
                "Tip: examples/bench_speculative.py compares speculative=True vs "
                "False tok/s and sweeps n_draft_max."
            )
    except UnsupportedModelError as e:
        raise SystemExit(
            f"{e}\n\nUnifiedLLM only supports a curated set of families; "
            "drop to the lower-level Llama API for others (see examples/basic.py)."
        ) from None


if __name__ == "__main__":
    main()
