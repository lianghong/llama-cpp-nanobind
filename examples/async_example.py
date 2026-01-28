#!/usr/bin/env python3
"""Async API example with thread-safe concurrent requests."""

import asyncio
import time

from llama_cpp import Llama, LlamaConfig


async def main() -> None:
    config = LlamaConfig(
        model_path="models/Qwen3-8B-Q6_K.gguf",
        n_ctx=4096,
        verbose=True,
    )

    # Use context manager for automatic cleanup
    with Llama("models/Qwen3-8B-Q6_K.gguf", config=config) as llm:
        print("=== Async Generation ===")
        start = time.perf_counter()

        # Single async request (thread-safe)
        text = await llm.generate_async("What is Python?", max_tokens=32)
        print(f"Response: {text}\n")

        # Async streaming
        print("=== Async Streaming ===")
        async for chunk in await llm.generate_async(
            "Name three colors:", max_tokens=24, stream=True
        ):
            print(chunk, end="", flush=True)
        print("\n")

        # Async chat completion
        print("=== Async Chat Completion ===")
        response = await llm.create_chat_completion_async(
            [{"role": "user", "content": "Say hello in one word"}],
            max_tokens=8,
        )
        print(f"Assistant: {response['choices'][0]['message']['content']}\n")

        # Concurrent requests (thread-safe with internal locking)
        print("=== Concurrent Requests ===")
        tasks = [
            llm.generate_async("Count to 3:", max_tokens=16),
            llm.generate_async("Name a fruit:", max_tokens=16),
        ]
        results = await asyncio.gather(*tasks)
        for i, result in enumerate(results, 1):
            print(f"Task {i}: {result.strip()}")

        elapsed = time.perf_counter() - start
        print(f"\nTotal time: {elapsed:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
