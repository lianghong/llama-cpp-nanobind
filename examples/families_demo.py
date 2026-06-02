"""Demo of the four UnifiedLLM-supported model families.

UnifiedLLM auto-detects the family from the GGUF filename and applies
the family's recommended sampling defaults, stop sequences, and chat
template. The four currently supported families are:

  * Qwen 3.5 (small + thinking variants)
  * Qwen 3.6 (dense + MoE; thinking + instruct)
  * Gemma 4 (dense + small MoE; thinking via system prefix)
  * IBM Granite 4.1 (deterministic by default)

Other GGUFs raise ``UnsupportedModelError`` at construction. For those,
drop to the lower-level ``Llama`` class (see ``basic.py``).

Usage:
    uv run python examples/families_demo.py models/Qwen3.5-4B-Q4_K_M.gguf
    uv run python examples/families_demo.py models/Gemma-4-E4B-Q8_0.gguf --thinking
    uv run python examples/families_demo.py models/granite-4-3b.gguf --prompt "Sum 7+8"
"""

from __future__ import annotations

import argparse
import os
import time

from llama_cpp.unified import UnifiedLLM
from llama_cpp.unified import UnsupportedModelError


SEPARATOR = "=" * 70


def _print_header(llm: UnifiedLLM) -> None:
    cfg = llm.model_config
    print(f"Model family:      {llm.family.name}")
    print(f"Supports thinking: {llm.supports_thinking}")
    print(f"Max context:       {cfg.max_ctx}")
    print(f"Default sampling:  T={cfg.temperature} top_p={cfg.top_p} top_k={cfg.top_k}")
    print(f"Stop sequences:    {cfg.stop_sequences}")
    print()


def _run(
    llm: UnifiedLLM,
    prompt: str,
    *,
    system_prompt: str | None,
    thinking: bool,
    max_tokens: int,
) -> None:
    print(f"{SEPARATOR}\nPROMPT: {prompt}\n{SEPARATOR}")
    start = time.perf_counter()

    if thinking and llm.supports_thinking:
        thoughts, answer = llm.generate_with_thinking(
            prompt, system_prompt=system_prompt, max_tokens=max_tokens
        )
        elapsed = time.perf_counter() - start
        if thoughts:
            preview = thoughts if len(thoughts) <= 400 else thoughts[:400] + "..."
            print(f"\nTHINKING:\n{preview}")
        print(f"\nANSWER:\n{answer}")
    else:
        if thinking and not llm.supports_thinking:
            print("(--thinking ignored: this family has no thinking mode)\n")
        answer = llm.generate(
            prompt, system_prompt=system_prompt, max_tokens=max_tokens
        )
        elapsed = time.perf_counter() - start
        print(f"\nOUTPUT:\n{answer}")

    print(f"\n[elapsed: {elapsed:.2f}s]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo of the four UnifiedLLM-supported model families."
    )
    parser.add_argument("model", help="Path to a GGUF model file")
    parser.add_argument("--prompt", default="What is the capital of France?")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Use generate_with_thinking when the family supports it",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(
            f"Model file not found: {args.model}\n"
            "Pass a path to a GGUF you have, e.g. "
            "--model models/Qwen3.5-9B-Q4_K_M.gguf"
        )

    try:
        with UnifiedLLM(args.model, n_ctx=args.ctx, verbose=args.verbose) as llm:
            _print_header(llm)
            _run(
                llm,
                args.prompt,
                system_prompt=args.system_prompt,
                thinking=args.thinking,
                max_tokens=args.max_tokens,
            )
    except UnsupportedModelError as e:
        raise SystemExit(
            f"{e}\n\nSee examples/basic.py for the lower-level Llama API."
        ) from None


if __name__ == "__main__":
    main()
