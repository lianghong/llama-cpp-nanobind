#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : sample_code.py
# Author            : Lianghong Fei <feilianghong@gmail.com>
# Date              : 2025-12-31
# Last Modified Date: 2025-12-31
# Last Modified By  : Lianghong Fei <feilianghong@gmail.com>
from __future__ import annotations

import gc

from llama_cpp import Llama, LlamaConfig, SamplingParams, shutdown
from llama_cpp.unified import UnifiedLLM


def llama_example() -> None:
    model_path = "models/Qwen3-30B-A3B-Instruct-2507-Q4_K_S.gguf"

    config = LlamaConfig(
        model_path=model_path,
        n_ctx=4096,
        n_batch=1024,
        n_ubatch=256,
        n_gpu_layers=-1,  # set 0 for CPU-only
        offload_kqv=True,
        flash_attn=1,
        use_mmap=True,
        use_mlock=False,
        verbose=False,
    )
    sampling = SamplingParams(temperature=0.7, top_p=0.8, top_k=20)

    # Context manager guarantees close() even on error
    with Llama(model_path, config=config, sampling=sampling) as llm:
        text = llm.generate("Hello! Summarize the benefits of GPUs:",
                            max_tokens=1024)
        print(f"\n*** llama_example output:\n {text}")

    # Explicit cleanup of any remaining objects (optional but safe)
    gc.collect()


def unified_example() -> None:
    model_path = "models/Qwen3-30B-A3B-Instruct-2507-Q4_K_S.gguf"

    # Context manager guarantees close() even on error
    with UnifiedLLM(
        model_path,
        n_ctx=4096,
        n_batch=1024,
        n_ubatch=256,
        n_gpu_layers=-1,  # force GPU (all layers)
    ) as llm:
        llm.model_config.temperature = 0.7
        llm.model_config.top_p = 0.8
        llm.model_config.top_k = 20
        text = llm.generate("Hello! Summarize the benefits of CUDA:",
                            max_tokens=1024)
        print(f"\n*** UnifiledLLM_example output:\n {text}")

    # Explicit cleanup of any remaining objects (optional but safe)
    gc.collect()


def main() -> None:
    llama_example()
    print("\n==================\n")
    unified_example()

    # Cleanly free backend resources (no-op if nothing loaded)
    shutdown()


if __name__ == "__main__":
    main()
