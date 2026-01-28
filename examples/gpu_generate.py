#!/usr/bin/env python3
"""GPU-optimized text generation example with context manager."""

import time

from llama_cpp import Llama, LlamaConfig, SamplingParams

MODEL = "models/Qwen3-8B-Q6_K.gguf"


def main() -> None:
    # Force full GPU offload (n_gpu_layers = -1). main_gpu=0 selects your first CUDA device.
    config = LlamaConfig(
        model_path=MODEL,
        n_ctx=4096,
        n_batch=512,
        n_ubatch=512,
        n_gpu_layers=-1,  # offload all layers to GPU if memory allows
        main_gpu=0,
        offload_kqv=True,  # keep K/Q/V on GPU
        flash_attn=1,  # enable flash attention
        n_seq_max=1,  # single sequence (default)
    )

    sampling = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.05,
    )

    # set_log_level("warn")  # Commented out for verbose debugging
    start = time.perf_counter()

    # Use context manager for proper cleanup
    with Llama(model_path=MODEL, config=config, sampling=sampling) as llm:
        prompt = "Write a short greeting from a GPU to its user."
        text = llm.generate(prompt, max_tokens=64)
        print("=== Single-shot ===")
        print(text)

        print("\n=== True Streaming ===")
        for chunk in llm.generate_stream(
            "List three GPU programming tips:", max_tokens=64
        ):
            print(chunk, end="", flush=True)
        print()

    elapsed = time.perf_counter() - start
    print(f"\nExecution time: {elapsed:.2f} s")


if __name__ == "__main__":
    main()
