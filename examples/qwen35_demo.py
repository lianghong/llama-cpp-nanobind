#!/usr/bin/env python3
"""Demo of Qwen3.5-27B model using UnifiedLLM wrapper.

Qwen3.5 is a large language model from Alibaba Cloud featuring a hybrid
attention architecture that combines Gated Delta Networks (linear attention)
with full attention layers. This design enables efficient processing of very
long contexts while maintaining high-quality reasoning.

Key characteristics:
    - Hybrid architecture: Gated Delta Networks + full attention layers
    - 262K native context window (262144 tokens)
    - 201 language support (vs 119 in Qwen3)
    - Thinking mode ON by default (controlled by chat template, NOT by
      /think or /nothink suffixes -- Qwen3.5 does NOT support the soft
      switch used in Qwen3)
    - Recommended presence_penalty=1.5 to prevent repetition (set
      automatically by UnifiedLLM's QWEN3_5 model config)
    - Sampling defaults: temperature=1.0, top_p=0.95, top_k=20
    - Thinking mode sampling: temperature=0.6, top_p=0.95, top_k=20

Usage:
    # Run built-in demo examples
    python examples/qwen35_demo.py

    # Single prompt
    python examples/qwen35_demo.py --prompt "Explain quantum entanglement"

    # Read prompt from file, show internal thinking
    python examples/qwen35_demo.py --input-file prompt.txt --show-thinking

    # Custom context size and system prompt
    python examples/qwen35_demo.py --ctx 32768 --system-prompt "You are a math tutor."

Reference: https://huggingface.co/Qwen/Qwen3.5-27B-GGUF
"""

import argparse
import time

from llama_cpp.unified import UnifiedLLM

DEFAULT_MODEL = "models/Qwen3.5-27B-Q4_K_M.gguf"
SEPARATOR = "=" * 60


def run_example(
    llm: UnifiedLLM,
    title: str,
    prompt: str,
    system_prompt: str | None = None,
    max_tokens: int = 512,
    show_thinking: bool = False,
) -> None:
    """Run a single generation example with timing.

    Qwen3.5 always uses thinking mode -- there is no toggle. The
    show_thinking flag controls whether the internal reasoning is printed.
    """
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)
    print(f"Prompt: {prompt}")
    if system_prompt:
        print(f"System: {system_prompt}")

    llm.kv_cache_clear()
    start = time.perf_counter()

    thinking_text, answer = llm.generate_with_thinking(
        prompt, system_prompt=system_prompt, max_tokens=max_tokens
    )
    elapsed = time.perf_counter() - start

    if show_thinking and thinking_text:
        print(f"\nThinking:\n{thinking_text}")
    print(f"\nAnswer:\n{answer}")

    print(f"\n  [thinking | {elapsed:.2f}s]")


def run_builtin_demos(llm: UnifiedLLM, show_thinking: bool = False) -> None:
    """Run built-in demo examples showcasing Qwen3.5 capabilities."""
    # -- Example 1: Basic generation -------------------------------------------
    run_example(
        llm,
        "1. Basic Generation",
        "What are the key advantages of hybrid attention architectures "
        "that combine linear and full attention mechanisms?",
        show_thinking=show_thinking,
    )

    # -- Example 2: Math reasoning (thinking shines here) ----------------------
    run_example(
        llm,
        "2. Math Reasoning",
        "A train travels from city A to city B at 60 km/h and returns at "
        "90 km/h. What is the average speed for the round trip?",
        show_thinking=show_thinking,
        max_tokens=1024,
    )

    # -- Example 3: Code generation --------------------------------------------
    run_example(
        llm,
        "3. Code Generation",
        "Write a Python function that implements binary search on a sorted "
        "list. Include type hints, a docstring, and handle edge cases.",
        show_thinking=show_thinking,
        max_tokens=1024,
    )

    # -- Example 4: Multilingual generation (Chinese) --------------------------
    run_example(
        llm,
        "4. Multilingual Generation (Chinese)",
        "请用中文简要介绍门控增量网络（Gated Delta Networks）的工作原理，"
        "以及它与传统注意力机制相比有什么优势。",
        show_thinking=show_thinking,
        max_tokens=1024,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen3.5-27B demo (hybrid attention, 262K context, 201 languages)"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to GGUF model")
    parser.add_argument(
        "--ctx", type=int, default=16384, help="Context size (default: 16384)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument("--prompt", help="User prompt text")
    parser.add_argument(
        "--input-file", type=argparse.FileType("r"), help="Read prompt from file"
    )
    parser.add_argument("--system-prompt", help="Custom system prompt")
    parser.add_argument(
        "--show-thinking",
        action="store_true",
        help="Show internal thinking/reasoning process",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    print(f"Context size:  {args.ctx}")

    with UnifiedLLM(args.model, n_ctx=args.ctx, verbose=args.verbose) as llm:
        print(f"Model family:       {llm.family.name}")
        print(f"Supports thinking:  {llm.supports_thinking}")
        print(f"Context window:     {llm.n_ctx()} tokens")
        print(f"Temperature:        {llm.model_config.temperature}")
        print(f"Presence penalty:   {llm.model_config.presence_penalty}")
        print(
            "Note: Qwen3.5 thinks by default -- no /think suffix needed. "
            "Thinking is controlled by the chat template (enable_thinking), "
            "not by appending text to the prompt."
        )

        if args.input_file:
            user_prompt = args.input_file.read()
        elif args.prompt:
            user_prompt = args.prompt
        else:
            user_prompt = None

        if user_prompt is not None:
            # Single prompt mode
            print("\nGenerating (thinking mode is always on)")
            start = time.perf_counter()

            thinking_text, final = llm.generate_with_thinking(
                user_prompt, system_prompt=args.system_prompt
            )
            elapsed = time.perf_counter() - start

            if args.show_thinking and thinking_text:
                print(f"\n{SEPARATOR}\nTHINKING PROCESS:\n{SEPARATOR}\n{thinking_text}")
            print(f"\n{SEPARATOR}\nFINAL ANSWER:\n{SEPARATOR}\n{final}")
            print(f"\n{SEPARATOR}\nElapsed: {elapsed:.2f}s")
        else:
            # Built-in demo mode
            print("\nNo prompt provided -- running built-in demo examples.")
            run_builtin_demos(llm, show_thinking=args.show_thinking)

            print(f"\n{SEPARATOR}")
            print("Qwen3.5-27B demo completed.")
            print(SEPARATOR)


if __name__ == "__main__":
    main()
