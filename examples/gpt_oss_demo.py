#!/usr/bin/env python3
"""Demo of GPT-OSS model using UnifiedLLM wrapper."""

import argparse
import time

from llama_cpp.unified import UnifiedLLM

DEFAULT_MODEL = "models/gpt-oss-20b-Q4_K_M.gguf"
SEPARATOR = "=" * 60


def main():
    parser = argparse.ArgumentParser(description="GPT-OSS demo")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to GGUF model")
    parser.add_argument("--prompt", help="User prompt text")
    parser.add_argument(
        "--input-file", type=argparse.FileType("r"), help="Read prompt from file"
    )
    parser.add_argument("--system-prompt", help="Custom system prompt")
    parser.add_argument(
        "--reasoning",
        default="medium",
        choices=["low", "medium", "high"],
        help="Reasoning level",
    )
    parser.add_argument(
        "--show-analysis", action="store_true", help="Show analysis channel"
    )
    args = parser.parse_args()

    if args.input_file:
        user_prompt = args.input_file.read()
    elif args.prompt:
        user_prompt = args.prompt
    else:
        parser.error("Either --prompt or --input-file is required")

    print(f"Loading model: {args.model}")

    # Use context manager for proper cleanup
    with UnifiedLLM(args.model, verbose=True) as model:
        model.set_reasoning_level(args.reasoning)

        print(f"\nGenerating with reasoning={args.reasoning}")
        start = time.perf_counter()

        analysis, final = model.generate_with_thinking(
            user_prompt, system_prompt=args.system_prompt
        )
        elapsed = time.perf_counter() - start

        if args.show_analysis and analysis:
            print(f"\n{SEPARATOR}\nANALYSIS:\n{SEPARATOR}\n{analysis}")
        print(f"\n{SEPARATOR}\nFINAL:\n{SEPARATOR}\n{final}")
        print(f"\n{SEPARATOR}\nReasoning: {args.reasoning} | Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
