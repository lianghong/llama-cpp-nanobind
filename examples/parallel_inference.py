#!/usr/bin/env python3
"""Example: True parallel inference with LlamaPool.

This example demonstrates the performance difference between:
1. Serial processing with a single Llama instance
2. Parallel processing with LlamaPool

Run this to see the speedup from parallel inference.
"""

import asyncio
import time
from llama_cpp import Llama, LlamaPool, LlamaConfig

# Model path - adjust to your model location
MODEL_PATH = "models/Qwen3-8B-Q6_K.gguf"

# Test queries
QUERIES = [
    "What is artificial intelligence?",
    "Explain quantum computing in simple terms.",
    "What are the benefits of Python programming?",
    "How does machine learning work?",
]


async def serial_inference_baseline():
    """Baseline: Serial processing with single Llama instance.

    This is what happens when you use a single Llama instance with async
    methods - requests are serialized due to the thread-safety lock.
    """
    print("=" * 60)
    print("SERIAL INFERENCE (Single Instance)")
    print("=" * 60)

    config = LlamaConfig(model_path=MODEL_PATH, n_ctx=2048)
    llm = Llama(MODEL_PATH, config=config)

    start_time = time.time()
    results = []

    # Process queries one by one
    for i, query in enumerate(QUERIES, 1):
        print(f"[{i}/{len(QUERIES)}] Processing: {query[:50]}...")
        result = await llm.generate_async(query, max_tokens=32)
        results.append(result)

    elapsed = time.time() - start_time
    llm.close()

    print(f"\n✓ Completed in {elapsed:.2f}s")
    print(f"  Average: {elapsed / len(QUERIES):.2f}s per query")
    print(f"  Throughput: {len(QUERIES) / elapsed:.2f} queries/sec\n")

    return results, elapsed


async def parallel_inference_pool():
    """Improved: True parallel processing with LlamaPool.

    Multiple instances can process requests concurrently, achieving
    significant speedup with available GPU memory.
    """
    print("=" * 60)
    print("PARALLEL INFERENCE (Pool with 4 instances)")
    print("=" * 60)

    config = LlamaConfig(model_path=MODEL_PATH, n_ctx=2048)

    # Create pool with 4 workers
    # Note: Each worker loads the full model, so adjust pool_size based on GPU memory
    async with LlamaPool(MODEL_PATH, pool_size=4, config=config) as pool:
        start_time = time.time()

        # All queries start at the same time and run in parallel
        print(f"Processing {len(QUERIES)} queries in parallel...")
        results = await pool.generate_batch(QUERIES, max_tokens=32)

        elapsed = time.time() - start_time

        print(f"\n✓ Completed in {elapsed:.2f}s")
        print(f"  Average: {elapsed / len(QUERIES):.2f}s per query")
        print(f"  Throughput: {len(QUERIES) / elapsed:.2f} queries/sec\n")

        return results, elapsed


async def demonstrate_concurrent_requests():
    """Show that pool handles concurrent requests correctly."""
    print("=" * 60)
    print("CONCURRENT REQUEST HANDLING")
    print("=" * 60)

    config = LlamaConfig(model_path=MODEL_PATH, n_ctx=2048)

    async with LlamaPool(MODEL_PATH, pool_size=2, config=config) as pool:
        # Submit 4 requests concurrently to a pool of 2 workers
        # First 2 run immediately, next 2 wait for workers to become available
        queries = [
            "Short query 1",
            "Short query 2",
            "Short query 3",
            "Short query 4",
        ]

        print(f"Submitting {len(queries)} requests to pool of 2 workers...")
        print("(First 2 run immediately, next 2 wait for available workers)\n")

        start_time = time.time()
        results = await asyncio.gather(
            *[pool.generate(q, max_tokens=16) for q in queries]
        )
        elapsed = time.time() - start_time

        print(f"✓ All {len(queries)} requests completed in {elapsed:.2f}s\n")

        for i, (query, result) in enumerate(zip(queries, results, strict=True), 1):
            print(f"[{i}] {query}")
            print(f"    → {result[:50]}...\n")


async def chat_completion_example():
    """Example: Parallel chat completions."""
    print("=" * 60)
    print("PARALLEL CHAT COMPLETIONS")
    print("=" * 60)

    config = LlamaConfig(model_path=MODEL_PATH, n_ctx=2048, chat_format="gemma")

    async with LlamaPool(MODEL_PATH, pool_size=3, config=config) as pool:
        conversations = [
            [{"role": "user", "content": "Say hello in French"}],
            [{"role": "user", "content": "Count from 1 to 5"}],
            [{"role": "user", "content": "What color is the sky?"}],
        ]

        print(f"Processing {len(conversations)} chat conversations in parallel...\n")

        start_time = time.time()
        responses = await pool.create_chat_completion_batch(
            conversations, max_tokens=32
        )
        elapsed = time.time() - start_time

        print(f"✓ Completed in {elapsed:.2f}s\n")

        for i, response in enumerate(responses, 1):
            content = response["choices"][0]["message"]["content"]
            print(f"[{i}] {content}\n")


async def demonstrate_warmup():
    """Show warmup feature for production deployments."""
    print("=" * 60)
    print("MODEL WARMUP (Optional)")
    print("=" * 60)
    print("Pre-loading GPU caches for consistent first-request latency.\n")

    config = LlamaConfig(model_path=MODEL_PATH, n_ctx=2048)

    # Test without warmup
    print("Without warmup:")
    start_time = time.time()
    pool_no_warmup = LlamaPool(MODEL_PATH, pool_size=2, config=config, warmup=False)
    init_time_no_warmup = time.time() - start_time
    print(f"  Initialization: {init_time_no_warmup:.3f}s")

    first_request_start = time.time()
    await pool_no_warmup.generate("Test", max_tokens=8)
    first_request_time = time.time() - first_request_start
    print(f"  First request: {first_request_time:.3f}s")
    pool_no_warmup.close()

    # Test with warmup
    print("\nWith warmup:")
    start_time = time.time()
    pool_with_warmup = LlamaPool(MODEL_PATH, pool_size=2, config=config, warmup=True)
    init_time_with_warmup = time.time() - start_time
    print(f"  Initialization: {init_time_with_warmup:.3f}s (includes warmup)")

    first_request_start = time.time()
    await pool_with_warmup.generate("Test", max_tokens=8)
    first_request_time_warmed = time.time() - first_request_start
    print(f"  First request: {first_request_time_warmed:.3f}s")
    pool_with_warmup.close()

    warmup_overhead = init_time_with_warmup - init_time_no_warmup
    print(f"\nWarmup overhead: {warmup_overhead:.3f}s")
    print("ℹ️  Warmup recommended for production APIs with strict SLA requirements\n")


async def main():
    """Run all examples and show performance comparison."""
    print("\n" + "=" * 60)
    print("PARALLEL INFERENCE DEMO")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Queries: {len(QUERIES)}")
    print()

    # Run serial baseline
    serial_results, serial_time = await serial_inference_baseline()

    # Run parallel version
    parallel_results, parallel_time = await parallel_inference_pool()

    # Calculate speedup
    speedup = serial_time / parallel_time
    print("=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    print(f"Serial time:     {serial_time:.2f}s")
    print(f"Parallel time:   {parallel_time:.2f}s")
    print(f"Speedup:         {speedup:.2f}x faster")
    print(
        f"Time saved:      {serial_time - parallel_time:.2f}s ({(1 - parallel_time/serial_time) * 100:.1f}%)\n"
    )

    # Verify results match
    if serial_results == parallel_results:
        print("✓ Results are identical (correctness verified)\n")
    else:
        print("Note: Results may differ due to parallel execution order\n")

    # Show concurrent request handling
    await demonstrate_concurrent_requests()

    # Show chat completion example
    await chat_completion_example()

    # Show warmup feature
    await demonstrate_warmup()

    print("=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("✓ LlamaPool enables true parallel inference")
    print("✓ Speedup scales with pool_size (up to GPU memory limit)")
    print("✓ Each worker needs full model in memory")
    print("✓ Perfect for multi-user APIs and batch processing")
    print("✓ Optional warmup for production deployments (warmup=True)")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FileNotFoundError:
        print(f"\nError: Model not found at {MODEL_PATH}")
        print("Please update MODEL_PATH to point to your model file.\n")
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
