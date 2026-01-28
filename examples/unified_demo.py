#!/usr/bin/env python3
"""Demo script for UnifiedLLM wrapper with context manager."""

import argparse
import time

from llama_cpp.unified import GPTOSSBackend, UnifiedLLM


def main() -> None:
    parser = argparse.ArgumentParser(description="UnifiedLLM demo")
    parser.add_argument(
        "--model", default="models/Qwen3-8B-Q6_K.gguf", help="Path to GGUF model"
    )
    parser.add_argument("--ctx", type=int, default=8192, help="Context size")
    parser.add_argument(
        "--reasoning", default="medium", choices=["low", "medium", "high"]
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    args = parser.parse_args()

    start_time = time.perf_counter()

    # Use context manager for automatic cleanup
    with UnifiedLLM(args.model, n_ctx=args.ctx, verbose=args.verbose) as llm:
        # Set reasoning level for GPT-OSS
        if isinstance(llm.backend, GPTOSSBackend):
            llm.set_reasoning_level(args.reasoning)

        print(f"Model family: {llm.family.name}")
        print(f"Supports thinking: {llm.supports_thinking}")
        print()

        default_tokens = 512 if llm.family.name == "GPT_OSS" else 64
        total_generated = 0
        total_gen_time = 0.0

        # Basic generation
        print("=== Basic Generation ===")
        llm.llm.perf_reset()
        t0 = time.perf_counter()
        response = llm.generate("What is 2 + 2?", max_tokens=default_tokens)
        t1 = time.perf_counter()
        perf = llm.llm.perf()
        n_tokens = perf.get("n_eval", 0)
        gen_time = t1 - t0
        total_generated += n_tokens
        total_gen_time += gen_time
        print(response)
        if n_tokens > 0 and gen_time > 0:
            print(f"  [{n_tokens} tokens, {n_tokens / gen_time:.1f} tok/s]")

        # With thinking mode
        if llm.supports_thinking:
            print("\n=== Generation with Thinking ===")
            llm.llm.perf_reset()
            t0 = time.perf_counter()
            thinking, answer = llm.generate_with_thinking(
                "Solve: x^2 - 4 = 0", max_tokens=1024
            )
            t1 = time.perf_counter()
            perf = llm.llm.perf()
            n_tokens = perf.get("n_eval", 0)
            gen_time = t1 - t0
            total_generated += n_tokens
            total_gen_time += gen_time
            if thinking:
                print(
                    f"Thinking: {thinking[:300]}..."
                    if len(thinking) > 300
                    else f"Thinking: {thinking}"
                )
            print(f"Answer: {answer}")
            if n_tokens > 0 and gen_time > 0:
                print(f"  [{n_tokens} tokens, {n_tokens / gen_time:.1f} tok/s]")

        # With system prompt
        print("\n=== With System Prompt ===")
        llm.llm.perf_reset()
        t0 = time.perf_counter()
        response = llm.generate(
            "What is the capital of France?",
            system_prompt="Answer in one sentence.",
            max_tokens=default_tokens,
        )
        t1 = time.perf_counter()
        perf = llm.llm.perf()
        n_tokens = perf.get("n_eval", 0)
        gen_time = t1 - t0
        total_generated += n_tokens
        total_gen_time += gen_time
        print(response)
        if n_tokens > 0 and gen_time > 0:
            print(f"  [{n_tokens} tokens, {n_tokens / gen_time:.1f} tok/s]")

        elapsed = time.perf_counter() - start_time
        print("\n=== Summary ===")
        print(f"Total generated: {total_generated} tokens in {total_gen_time:.2f}s")
        if total_gen_time > 0:
            print(f"Average speed: {total_generated / total_gen_time:.1f} tok/s")
        print(f"Total time (incl. loading): {elapsed:.2f}s")


if __name__ == "__main__":
    main()
