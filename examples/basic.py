#!/usr/bin/env python3
"""Minimal inference example with context manager."""
import time

from llama_cpp import Llama, ModelLoadError, SamplingParams


def main() -> None:
    start = time.perf_counter()

    try:
        # Use context manager for automatic resource cleanup
        with Llama("models/Qwen3-8B-Q6_K.gguf") as llm:
            print("=== Single-call generation ===")
            text = llm.generate("Write a short greeting from a GPU.", max_tokens=48)
            print(text)

            print("\n=== True streaming (yields as tokens decode) ===")
            sampling = SamplingParams(temperature=0.7, top_p=0.9, repeat_penalty=1.05)
            for chunk in llm.generate_stream(
                "Name three oceans:", max_tokens=32, sampling=sampling
            ):
                print(chunk, end="", flush=True)
            print()

            print("\n=== Session continuation (reuse KV cache) ===")
            llm.generate("Hello", max_tokens=10, reset_kv_cache=True)
            text = llm.generate("Continue the greeting:", max_tokens=20, reset_kv_cache=False)
            print(text)

    except ModelLoadError as e:
        print(f"Failed to load model: {e}")
        return

    elapsed = time.perf_counter() - start
    print(f"\nExecution time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
