#!/usr/bin/env python3
# File              : translate.py
# Author            : Lianghong Fei <feilianghong@gmail.com>
# Date              : 2026-02-11
# Last Modified Date: 2026-03-03
# Last Modified By  : Lianghong Fei <feilianghong@gmail.com>
"""Translation tool using UnifiedLLM.

Translates text files into a target language (default: Simplified Chinese)
with automatic model family detection and optimized settings.

Usage:
    python examples/translate.py --model models/Qwen3.5-4B-Q4_K_M.gguf --ctx 8192
    python examples/translate.py --model models/Qwen3.5-4B-Q4_K_M.gguf -t Japanese
    python examples/translate.py --model models/Qwen3.5-4B-Q4_K_M.gguf --thinking
    python examples/translate.py --model models/Qwen3.5-4B-Q4_K_M.gguf -o out.txt  # specific output path
    python examples/translate.py --model models/Qwen3.5-4B-Q4_K_M.gguf --no-save  # console only
"""

import argparse
from pathlib import Path
import time

from llama_cpp.unified import GPTOSSBackend
from llama_cpp.unified import UnifiedLLM


DEFAULT_INPUT: Path = Path(__file__).parent / "example.md"
"""Default input file for translation."""

# ISO 639-1 codes → full language names for use in prompts
LANG_CODES: dict[str, str] = {
    "ar": "Arabic",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "zh": "Simplified Chinese",
    "zh-cn": "Simplified Chinese",
    "zh-tw": "Traditional Chinese",
}
"""Map ISO 639-1 codes to full language names for prompt clarity."""

SYSTEM_PROMPT_TEMPLATE: str = """\
You are an expert {target_lang} translator producing publication-ready output.

FAITHFULNESS:
Convey the original meaning, tone, and intent exactly. Never omit, add, \
editorialize, or soften content. Preserve the author's voice and stance — \
including sarcasm, criticism, irony, and controversial opinions — without \
commentary or moral hedging.

FLUENCY:
Write natural, idiomatic {target_lang} as if the text were originally composed \
by a native speaker. Restructure sentences to follow {target_lang} conventions \
rather than mirroring the source syntax word-for-word.

STYLE:
Match the source register — formal for formal, casual for casual, literary for \
literary. Use precise domain terms for technical text. Preserve rhetorical \
devices (metaphor, parallelism, understatement) with {target_lang} equivalents.

SPECIFICS:
- Proper nouns: Keep names, brands, and product names in original form unless \
{target_lang} has a widely established translation.
- Markdown: Preserve all syntax (links, images, headings, bold, code blocks). \
Translate display text only, never URLs or image paths.
- Numbers, dates, percentages: Keep in original format.
- Cultural references with no direct equivalent: Translate the meaning; \
optionally keep the original term in parentheses for clarity.

OUTPUT: Translation only. No notes, no commentary, no explanations. Stop \
immediately after the last translated sentence.
"""
"""System prompt template with {target_lang} placeholder."""

USER_PROMPT_TEMPLATE: str = "Translate the following into {target_lang}:\n\n{text}"
"""User prompt template with {target_lang} and {text} placeholders."""


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
    parser = argparse.ArgumentParser(
        description="Translate text files using UnifiedLLM"
    )
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument(
        "-t",
        "--target-lang",
        default="Simplified Chinese",
        help="Target language (default: Simplified Chinese)",
    )
    parser.add_argument("--ctx", type=int, default=10240, help="Context size")
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
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (default: 0.3, lower = more faithful)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: <source_stem>.<lang><source_ext> "
        "in source directory)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save translation to file (console output only)",
    )
    return parser.parse_args()


def main() -> int:
    """Run the translation test.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    args: argparse.Namespace = parse_args()

    # Resolve language codes (e.g. "zh" → "Simplified Chinese")
    lang_input: str = args.target_lang.strip()
    lang_key = lang_input.lower()
    if lang_key in LANG_CODES:
        args.target_lang = LANG_CODES[lang_key]
        args.lang_slug = lang_key  # keep short code for filenames
    else:
        args.lang_slug = lang_input.lower().replace(" ", "-")

    # Validate batch sizes
    n_batch: int = min(args.batch, args.ctx)
    n_ubatch: int = min(args.ubatch, n_batch)

    # Validate paths
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        return 1
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

            # Override temperature for translation (lower = more faithful)
            llm.model_config.temperature = args.temperature

            # Print config
            print(f"Model: {args.model} ({llm.family.name})")
            print(f"Target: {args.target_lang}")
            print(f"Context: {args.ctx}, batch: {n_batch}, ubatch: {n_ubatch}")
            print(f"Temperature: {args.temperature}")

            # Build prompts for target language
            system_prompt: str = SYSTEM_PROMPT_TEMPLATE.format(
                target_lang=args.target_lang,
            )
            user_prompt: str = USER_PROMPT_TEMPLATE.format(
                target_lang=args.target_lang,
                text=input_text,
            )
            prompt_tokens: int = (
                llm.n_tokens(user_prompt) + llm.n_tokens(system_prompt) + 50
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
                    system_prompt,
                    max_tokens=max_tokens,
                    stop=args.stop,
                )
                if thinking_text:
                    thinking_tokens = llm.n_tokens(thinking_text)
            else:
                result = llm.generate(
                    user_prompt,
                    system_prompt,
                    max_tokens=max_tokens,
                    thinking=False,
                    stop=args.stop,
                )
            print(result)

            # Metrics
            gen_time: float = time.perf_counter() - gen_start
            perf: dict[str, int] = llm.llm.perf()
            n_eval: int = perf.get("n_eval", 0)
            speed: float = n_eval / gen_time if gen_time > 0 else 0

            # Save output to file (default: alongside source file)
            output_file: Path | None = None
            if not args.no_save:
                if args.output is not None:
                    output_file = args.output
                else:
                    output_file = (
                        args.file.parent
                        / f"{args.file.stem}.{args.lang_slug}{args.file.suffix}"
                    )
                output_file.write_text(result, encoding="utf-8")

            # Print metrics
            print(f"\n{'=' * 50}\nMETRICS\n{'=' * 50}")
            print(f"Generated: {n_eval} tokens in {gen_time:.1f}s ({speed:.1f} tok/s)")
            if thinking_tokens > 0:
                answer_tokens: int = llm.n_tokens(result)
                print(
                    f"Thinking: {thinking_tokens} tokens, "
                    f"Answer: {answer_tokens} tokens, "
                    f"Total: {thinking_tokens + answer_tokens} tokens"
                )
            if output_file is not None:
                print(f"Saved: {output_file}")
            print(f"Total time: {time.perf_counter() - start_time:.1f}s")

    except (ValueError, RuntimeError, OSError) as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
