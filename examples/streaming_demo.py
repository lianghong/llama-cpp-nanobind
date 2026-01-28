#!/usr/bin/env python3
"""Demonstration of true incremental streaming.

This example shows the difference between:
1. generate(..., stream=True) - buffers all tokens, then yields
2. generate_stream() - yields tokens as they're generated (TRUE streaming)
"""

import sys
import time
from llama_cpp import Llama

MODEL_PATH = "models/Qwen3-8B-Q6_K.gguf"


def demo_buffered_streaming():
    """Demo: generate(..., stream=True) - appears to stream but is buffered."""
    print("=" * 70)
    print("BUFFERED STREAMING: generate(..., stream=True)")
    print("=" * 70)
    print("This buffers all tokens first, then yields them quickly.")
    print()

    llm = Llama(MODEL_PATH)

    prompt = "Count from 1 to 10 slowly"
    print(f"Prompt: {prompt}")
    print("Output: ", end="", flush=True)

    start_time = time.time()
    first_chunk_time = None

    for chunk in llm.generate(prompt, max_tokens=48, stream=True):
        if first_chunk_time is None:
            first_chunk_time = time.time()
            print(
                f"\n[First chunk after {first_chunk_time - start_time:.3f}s]",
                flush=True,
            )
            print("Output: ", end="", flush=True)
        print(chunk, end="", flush=True)
        time.sleep(0.01)  # Simulate processing time

    end_time = time.time()
    llm.close()

    print("\n\nTiming:")
    print(f"  Time to first chunk: {first_chunk_time - start_time:.3f}s")
    print(f"  Total time: {end_time - start_time:.3f}s")
    print(
        f"  → First chunk arrived at {(first_chunk_time - start_time) / (end_time - start_time) * 100:.0f}% of total time"
    )
    print()


def demo_true_streaming():
    """Demo: generate_stream() - true incremental streaming."""
    print("=" * 70)
    print("TRUE STREAMING: generate_stream()")
    print("=" * 70)
    print("This yields tokens as they're generated (incremental).")
    print()

    llm = Llama(MODEL_PATH)

    prompt = "Count from 1 to 10 slowly"
    print(f"Prompt: {prompt}")
    print("Output: ", end="", flush=True)

    start_time = time.time()
    first_chunk_time = None
    chunk_times = []

    for chunk in llm.generate_stream(prompt, max_tokens=48):
        current_time = time.time()
        if first_chunk_time is None:
            first_chunk_time = current_time
            print(
                f"\n[First chunk after {first_chunk_time - start_time:.3f}s]",
                flush=True,
            )
            print("Output: ", end="", flush=True)
        chunk_times.append(current_time - start_time)
        print(chunk, end="", flush=True)
        time.sleep(0.01)  # Simulate processing time

    end_time = time.time()
    llm.close()

    print("\n\nTiming:")
    print(f"  Time to first chunk: {first_chunk_time - start_time:.3f}s")
    print(f"  Total time: {end_time - start_time:.3f}s")
    print(
        f"  → First chunk arrived at {(first_chunk_time - start_time) / (end_time - start_time) * 100:.0f}% of total time"
    )
    print(f"  → Received {len(chunk_times)} chunks incrementally")
    print()


def demo_streaming_with_stop():
    """Demo: Streaming with stop sequences."""
    print("=" * 70)
    print("STREAMING WITH STOP SEQUENCES")
    print("=" * 70)
    print()

    llm = Llama(MODEL_PATH)

    prompt = "List three colors: red"
    print(f"Prompt: {prompt}")
    print("Stop sequence: [',']")
    print("Output: ", end="", flush=True)

    for chunk in llm.generate_stream(prompt, max_tokens=32, stop=[","]):
        print(chunk, end="", flush=True)

    print("\n[Stopped at comma]")
    llm.close()
    print()


def demo_early_termination():
    """Demo: Early termination of streaming."""
    print("=" * 70)
    print("EARLY TERMINATION")
    print("=" * 70)
    print("Taking only first 5 chunks, then stopping.\n")

    llm = Llama(MODEL_PATH)

    prompt = "Write a long story"
    print(f"Prompt: {prompt}")
    print("Output: ", end="", flush=True)

    for i, chunk in enumerate(llm.generate_stream(prompt, max_tokens=100)):
        print(chunk, end="", flush=True)
        if i >= 4:  # Take only first 5 chunks
            print("\n[Stopped early after 5 chunks]")
            break

    llm.close()
    print()


def main():
    """Run all demonstrations."""
    try:
        print("\n" + "=" * 70)
        print("TRUE INCREMENTAL STREAMING DEMONSTRATION")
        print("=" * 70)
        print(f"Model: {MODEL_PATH}")
        print()

        # Demo 1: Buffered (old behavior)
        demo_buffered_streaming()

        # Demo 2: True streaming (new behavior)
        demo_true_streaming()

        # Demo 3: Stop sequences
        demo_streaming_with_stop()

        # Demo 4: Early termination
        demo_early_termination()

        print("=" * 70)
        print("KEY DIFFERENCES")
        print("=" * 70)
        print("✓ generate_stream() yields tokens as generated (true streaming)")
        print("✓ First token arrives much sooner (low latency)")
        print("✓ Perfect for SSE endpoints, WebSocket streaming, live UIs")
        print("✓ Can be terminated early without waiting for completion")
        print()
        print("✗ generate(..., stream=True) buffers then yields (pseudo-streaming)")
        print("✗ First token arrives only after ALL tokens generated")
        print("✗ High latency - user waits for entire generation")
        print("=" * 70)

    except FileNotFoundError:
        print(f"\nError: Model not found at {MODEL_PATH}")
        print("Please update MODEL_PATH to point to your model file.\n")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
