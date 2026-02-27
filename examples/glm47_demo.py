#!/usr/bin/env python3
"""Demo of GLM-4.7-Flash-REAP model using UnifiedLLM wrapper.

GLM-4.7-Flash is a Mixture-of-Experts (MoE) language model from Z.AI
(Zhipu AI) with 23B total parameters and 3B active parameters per token,
making it highly efficient for its capability level.

The REAP (Router-weighted Expert Activation Pruning) variant is from
Cerebras, which prunes 25% of MoE experts while preserving quality.

Key features:
    - MoE architecture: 23B total / 3B active parameters
    - 202K context window (202752 tokens)
    - Thinking mode via <think> tags (supports reasoning/chain-of-thought)
    - Bilingual: English + Chinese
    - Repeat penalty must be disabled (repeat_penalty=1.0) for correct
      generation -- UnifiedLLM handles this automatically via the GLM-4.7
      model config which uses presence_penalty instead

Reference: https://huggingface.co/cerebras/GLM-4.7-Flash-REAP-23B-A3B
"""

import argparse
import time

from llama_cpp.unified import UnifiedLLM

DEFAULT_MODEL = "models/GLM-4.7-Flash-REAP-23B-A3B-Q4_K_M.gguf"
SEPARATOR = "=" * 60


def run_example(
    llm: UnifiedLLM,
    title: str,
    prompt: str,
    system_prompt: str | None = None,
    max_tokens: int = 512,
    thinking: bool = False,
    show_thinking: bool = False,
) -> None:
    """Run a single generation example with timing."""
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)
    print(f"Prompt: {prompt}")
    if system_prompt:
        print(f"System: {system_prompt}")

    llm.kv_cache_clear()
    start = time.perf_counter()

    if thinking:
        thinking_text, answer = llm.generate_with_thinking(
            prompt, system_prompt=system_prompt, max_tokens=max_tokens
        )
        elapsed = time.perf_counter() - start

        if show_thinking and thinking_text:
            print(f"\nThinking:\n{thinking_text}")
        print(f"\nAnswer:\n{answer}")
    else:
        response = llm.generate(
            prompt, system_prompt=system_prompt, max_tokens=max_tokens
        )
        elapsed = time.perf_counter() - start
        print(f"\nResponse:\n{response}")

    mode = "thinking" if thinking else "standard"
    print(f"\n  [{mode} | {elapsed:.2f}s]")


def run_builtin_demos(llm: UnifiedLLM) -> None:
    """Run built-in demo examples showcasing GLM-4.7 capabilities."""
    # -- Example 1: Basic English generation -----------------------------------
    run_example(
        llm,
        "1. Basic Generation (English)",
        "Explain the difference between compiled and interpreted "
        "programming languages in three sentences.",
    )

    # -- Example 2: Chinese language generation --------------------------------
    run_example(
        llm,
        "2. Chinese Language Generation",
        "请用中文简要介绍混合专家模型（MoE）的工作原理及其优势。",
    )

    # -- Example 3: Thinking mode -- math problem ------------------------------
    run_example(
        llm,
        "3. Thinking Mode: Math Problem",
        "Solve: what is the sum of the first 20 prime numbers?",
        thinking=True,
        show_thinking=True,
        max_tokens=1024,
    )

    # -- Example 4: Thinking mode -- code generation ---------------------------
    run_example(
        llm,
        "4. Thinking Mode: Code Generation",
        "Write a Python function to check if a string is a palindrome. "
        "Include docstring and handle edge cases.",
        thinking=True,
        show_thinking=True,
        max_tokens=1024,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GLM-4.7-Flash-REAP demo (23B MoE, 3B active)"
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
    parser.add_argument("--think", action="store_true", help="Enable thinking mode")
    parser.add_argument(
        "--show-thinking",
        action="store_true",
        help="Show thinking process (implies --think)",
    )
    args = parser.parse_args()

    if args.show_thinking:
        args.think = True

    print(f"Loading model: {args.model}")
    print(f"Context size:  {args.ctx}")

    with UnifiedLLM(args.model, n_ctx=args.ctx, verbose=args.verbose) as llm:
        print(f"Model family:       {llm.family.name}")
        print(f"Supports thinking:  {llm.supports_thinking}")
        print(f"Context window:     {llm.n_ctx()} tokens")
        print(f"Temperature:        {llm.model_config.temperature}")
        print(
            "Note: repeat penalty is disabled for GLM-4.7 "
            "(handled automatically by UnifiedLLM config)"
        )

        if args.input_file:
            user_prompt = args.input_file.read()
        elif args.prompt:
            user_prompt = args.prompt
        else:
            user_prompt = None

        if user_prompt is not None:
            # Single prompt mode
            mode = "thinking" if args.think else "non-thinking"
            print(f"\nGenerating with {mode} mode")
            start = time.perf_counter()

            if args.think:
                thinking_text, final = llm.generate_with_thinking(
                    user_prompt, system_prompt=args.system_prompt
                )
                elapsed = time.perf_counter() - start

                if args.show_thinking and thinking_text:
                    print(
                        f"\n{SEPARATOR}\nTHINKING PROCESS:\n"
                        f"{SEPARATOR}\n{thinking_text}"
                    )
                print(f"\n{SEPARATOR}\nFINAL ANSWER:\n{SEPARATOR}\n{final}")
            else:
                response = llm.generate(user_prompt, system_prompt=args.system_prompt)
                elapsed = time.perf_counter() - start
                print(f"\n{SEPARATOR}\nOUTPUT:\n{SEPARATOR}\n{response}")

            print(f"\n{SEPARATOR}\nMode: {mode} | Elapsed: {elapsed:.2f}s")
        else:
            # Built-in demo mode
            print("\nNo prompt provided -- running built-in demo examples.")
            run_builtin_demos(llm)

            print(f"\n{SEPARATOR}")
            print("GLM-4.7-Flash-REAP demo completed.")
            print(SEPARATOR)


if __name__ == "__main__":
    main()
