#!/usr/bin/env python3
"""Test maximum context size with translation task.

This script tests LLM translation capabilities using UnifiedLLM with
automatic model family detection and optimized settings.

Usage:
    python examples/ctx_test.py --model models/Qwen3-30B-A3B-Instruct-2507-Q4_K_S.gguf --ctx 8192
    python examples/ctx_test.py --model models/Qwen3-8B-Q6_K.gguf --thinking
    python examples/ctx_test.py --model models/gpt-oss-20b-Q4_K_M.gguf --reasoning_level high
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

DEFAULT_INPUT: Path = Path(__file__).parent / "example.txt"
"""Default input file for translation."""

SYSTEM_PROMPT: str = """You are a professional English-to-Chinese translator. Follow these guidelines:
1. Use standard Simplified Chinese characters and expressions
2. Maintain the original meaning, tone, and style accurately
3. Use natural, fluent Chinese that reads well to native speakers
4. No explanations, no comments, just output the translation result"""
"""System prompt for translation task."""

USER_PROMPT_TEMPLATE: str = (
    "Translate the following English text into Simplified Chinese:\n\n{text}"
)
"""User prompt template with {text} placeholder."""


def get_gpu_free_memory_gb() -> float | None:
    """Get free GPU memory in GB using nvidia-smi.

    Returns:
        Free memory in GB, or None if unavailable.
    """
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Sum free memory across all GPUs, return in GB
            free_mb = sum(int(x.strip()) for x in result.stdout.strip().split("\n"))
            return free_mb / 1024
    except Exception:
        pass
    return None


def estimate_vram_gb(model_path: str, n_ctx: int) -> float:
    """Estimate VRAM usage based on model size and context.

    Args:
        model_path: Path to model file.
        n_ctx: Context size.

    Returns:
        Estimated VRAM in GB.
    """
    model_size_gb = Path(model_path).stat().st_size / (1024**3)
    # KV cache estimate: ~0.5MB per 1K context for Q4 models
    kv_cache_gb = (n_ctx / 1024) * 0.5 / 1024
    # Add overhead for compute buffers
    return model_size_gb + kv_cache_gb + 1.0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Test max context with translation")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--ctx", type=int, default=8192, help="Context size")
    parser.add_argument("--batch", type=int, default=4096, help="Batch size")
    parser.add_argument("--ubatch", type=int, default=512, help="Micro batch size")
    parser.add_argument(
        "--file", type=Path, default=DEFAULT_INPUT, help="Input text file"
    )
    parser.add_argument(
        "--thinking", action="store_true", help="Enable thinking mode (Qwen3)"
    )
    parser.add_argument(
        "--reasoning_level", default="medium", choices=["low", "medium", "high"]
    )
    parser.add_argument("--max_tokens", type=int, help="Max output tokens")
    parser.add_argument("--stop", type=str, nargs="*", help="Stop sequences")
    parser.add_argument(
        "--n_gpu_layers", type=int, default=-1, help="GPU layers (-1=all)"
    )
    parser.add_argument(
        "--no_flash_attn", action="store_true", help="Disable flash attention"
    )
    return parser.parse_args()


def main() -> int:
    """Run the translation test.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    args: argparse.Namespace = parse_args()

    # Import after arg parsing to avoid crash on -h
    from llama_cpp.unified import GPTOSSBackend, UnifiedLLM

    # Validate batch sizes
    n_batch: int = min(args.batch, args.ctx)
    n_ubatch: int = min(args.ubatch, n_batch)

    # Read input
    if not args.file.exists():
        print(f"Error: Input file not found: {args.file}")
        return 1
    input_text: str = args.file.read_text(encoding="utf-8")
    print(f"Input: {args.file} ({len(input_text)} chars)")

    # Check VRAM before loading
    free_vram = get_gpu_free_memory_gb()
    estimated_vram = estimate_vram_gb(args.model, args.ctx)
    if free_vram is not None:
        print(f"VRAM: {free_vram:.1f} GB free, ~{estimated_vram:.1f} GB estimated")
        if estimated_vram > free_vram * 0.9:
            print(
                f"\n⚠️  WARNING: Estimated VRAM ({estimated_vram:.1f} GB) may exceed "
                f"available ({free_vram:.1f} GB).\n"
                f"   Consider reducing --ctx (current: {args.ctx}) or use a smaller model.\n"
                f"   CUDA OOM errors crash the process without recovery.\n"
            )
            response = input("Continue anyway? [y/N]: ").strip().lower()
            if response != "y":
                print("Aborted.")
                return 1

    # Load model with context manager for proper cleanup
    start_time: float = time.perf_counter()
    try:
        with UnifiedLLM(
            args.model,
            n_ctx=args.ctx,
            n_batch=n_batch,
            n_ubatch=n_ubatch,
            n_gpu_layers=args.n_gpu_layers,
            verbose=False,
        ) as llm:
            # Configure model-specific settings
            if isinstance(llm.backend, GPTOSSBackend):
                llm.set_reasoning_level(args.reasoning_level)

            # Print config
            print(f"Model: {args.model} ({llm.family.name})")
            print(f"Context: {args.ctx}, batch: {n_batch}, ubatch: {n_ubatch}")

            # Prepare prompt and calculate tokens
            user_prompt: str = USER_PROMPT_TEMPLATE.format(text=input_text)
            prompt_tokens: int = (
                llm.llm.n_tokens(user_prompt) + llm.llm.n_tokens(SYSTEM_PROMPT) + 50
            )
            available: int = args.ctx - prompt_tokens - 10

            if available < 1:
                print(
                    f"Error: Prompt ({prompt_tokens} tokens) exceeds context ({args.ctx})"
                )
                return 1

            max_tokens: int = (
                min(args.max_tokens, available) if args.max_tokens else available
            )
            print(f"Tokens: ~{prompt_tokens} prompt, {max_tokens} max output")

            # Generate
            print("\n" + "=" * 50 + "\nTRANSLATION\n" + "=" * 50 + "\n")
            llm.llm.perf_reset()
            gen_start: float = time.perf_counter()

            thinking_tokens: int = 0
            if args.thinking:
                thinking_text, result = llm.generate_with_thinking(
                    user_prompt,
                    SYSTEM_PROMPT,
                    max_tokens=max_tokens,
                    stop=args.stop,
                )
                if thinking_text:
                    thinking_tokens = llm.n_tokens(thinking_text)
            else:
                result = llm.generate(
                    user_prompt,
                    SYSTEM_PROMPT,
                    max_tokens=max_tokens,
                    thinking=False,
                    stop=args.stop,
                )
            print(result)

            # Metrics
            gen_time: float = time.perf_counter() - gen_start
            perf: dict[str, int] = llm.llm.perf()
            n_eval: int = perf.get("n_eval", 0)

            # Save output
            timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file: Path = Path(f"{Path(args.model).stem}_{timestamp}.txt")
            output_file.write_text(result, encoding="utf-8")
            print(f"\n[Saved: {output_file}]")

            # Print metrics
            print(f"\n{'=' * 50}\nMETRICS\n{'=' * 50}")
            speed: float = n_eval / gen_time if gen_time > 0 else 0
            print(f"Generated: {n_eval} tokens in {gen_time:.1f}s ({speed:.1f} tok/s)")
            if thinking_tokens > 0:
                answer_tokens: int = llm.n_tokens(result)
                print(
                    f"Thinking: {thinking_tokens} tokens, "
                    f"Answer: {answer_tokens} tokens, "
                    f"Total: {thinking_tokens + answer_tokens} tokens"
                )
            print(f"Total time: {time.perf_counter() - start_time:.1f}s")

    except ValueError as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
