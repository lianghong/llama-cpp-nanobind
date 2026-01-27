#!/usr/bin/env python3
"""Demo of Mistral model using UnifiedLLM wrapper."""

import argparse
import time

from llama_cpp.unified import UnifiedLLM

DEFAULT_MODEL = "models/Mistral-Instruct.gguf"
SEPARATOR = "=" * 60


def main():
    parser = argparse.ArgumentParser(description="Mistral demo")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to GGUF model")
    parser.add_argument("--prompt", help="User prompt text")
    parser.add_argument(
        "--input-file", type=argparse.FileType("r"), help="Read prompt from file"
    )
    parser.add_argument("--system-prompt", help="Custom system prompt")
    args = parser.parse_args()

    if args.input_file:
        user_prompt = args.input_file.read()
    elif args.prompt:
        user_prompt = args.prompt
    else:
        parser.error("Either --prompt or --input-file is required")

    print(f"Loading model: {args.model}")

    # Use context manager for proper cleanup
    with UnifiedLLM(args.model, verbose=False) as model:
        print(f"Model family: {model.family.name}")

        start = time.perf_counter()
        response = model.generate(user_prompt, system_prompt=args.system_prompt)
        elapsed = time.perf_counter() - start

        print(f"\n{SEPARATOR}\nOUTPUT:\n{SEPARATOR}\n{response}")
        print(f"\n{SEPARATOR}\nElapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
