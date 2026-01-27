#!/usr/bin/env python3
"""Demo of Qwen3 model using UnifiedLLM wrapper."""

import argparse
import time

from llama_cpp.unified import UnifiedLLM

DEFAULT_MODEL = "models/Qwen3-30B-A3B-Instruct-2507-Q4_K_S.gguf"
SEPARATOR = "=" * 60


def main():
    parser = argparse.ArgumentParser(description="Qwen3 demo")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to GGUF model")
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

    if args.input_file:
        user_prompt = args.input_file.read()
    elif args.prompt:
        user_prompt = args.prompt
    else:
        parser.error("Either --prompt or --input-file is required")

    print(f"Loading model: {args.model}")

    # Use context manager for proper cleanup
    with UnifiedLLM(args.model, verbose=False) as model:
        mode = "thinking" if args.think else "non-thinking"
        print(f"\nGenerating with {mode} mode")
        start = time.perf_counter()

        if args.think:
            thinking, final = model.generate_with_thinking(
                user_prompt, system_prompt=args.system_prompt
            )
            elapsed = time.perf_counter() - start

            if args.show_thinking and thinking:
                print(f"\n{SEPARATOR}\nTHINKING PROCESS:\n{SEPARATOR}\n{thinking}")
            print(f"\n{SEPARATOR}\nFINAL ANSWER:\n{SEPARATOR}\n{final}")
        else:
            response = model.generate(user_prompt, system_prompt=args.system_prompt)
            elapsed = time.perf_counter() - start
            print(f"\n{SEPARATOR}\nOUTPUT:\n{SEPARATOR}\n{response}")

        print(f"\n{SEPARATOR}\nMode: {mode} | Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
