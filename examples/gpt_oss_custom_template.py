#!/usr/bin/env python3
# File              : gpt_oss_custom_template.py
# Author            : Lianghong Fei <feilianghong@gmail.com>
# Date              : 2025-12-08
# Last Modified Date: 2025-12-09
# Last Modified By  : Lianghong Fei <feilianghong@gmail.com>
"""GPT-OSS generation with custom chat template."""

import argparse
import re
import time
from datetime import datetime, timezone
from typing import TypedDict

from llama_cpp import Llama, LlamaConfig, SamplingParams  # type: ignore[import-untyped]


class GenerationResult(TypedDict):
    """Result from GPT-OSS generation."""

    analysis: str
    final: str
    elapsed_time: float
    prompt_tokens: int
    response_tokens: int


# Tuned for GPT-OSS-20B Q4_K_M on 20 GB VRAM (14 physical CPU cores)
#  - 8K context keeps KV cache (~6.5 GB) within headroom alongside ~12 GB weights
#  - Larger prompt batch with smaller microbatch balances throughput vs. memory
#  - Threads match physical cores for CPU-side ops (tokenization, I/O)
# DEFAULT_CTX = 8192
DEFAULT_CTX = 65536
DEFAULT_BATCH = 2048
DEFAULT_UBATCH = 2048
DEFAULT_THREADS = 8
DEFAULT_MAX_TOKENS = 32768

DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = 0
STOP_STRING = ["<|start|>user", "<|end|><|end|>"]

# Compiled regex patterns for performance
_ANALYSIS_PATTERN = re.compile(
    r"<\|channel\|\>\s*analysis\s*<\|message\|\>(.*?)" r"(?:<\|end\|\>|<\|start\|\>|$)",
    re.DOTALL,
)
_FINAL_PATTERN = re.compile(
    r"<\|channel\|\>\s*final\s*<\|message\|\>(.*?)(?:<\|end\|\>|$)", re.DOTALL
)


def get_today(tz: timezone | None = None) -> str:
    """Get current date in YYYY-MM-DD format.

    Args:
        tz: Timezone to use. Defaults to local timezone if None.

    Returns:
        Current date string in ISO format.
    """
    return datetime.now(tz).strftime("%Y-%m-%d")


def gpt_oss_config(model_path: str):
    # GPU-optimized config
    config = LlamaConfig(
        model_path=model_path,
        n_ctx=DEFAULT_CTX,
        n_batch=DEFAULT_BATCH,
        n_ubatch=DEFAULT_UBATCH,
        n_threads=DEFAULT_THREADS,
        n_gpu_layers=-1,
        offload_kqv=True,
        flash_attn=1,
        verbose=True,
    )
    return config


def model_sampling(temperature: float = 1.0, top_p: float = 10.0, top_k: int = 0):
    return SamplingParams(temperature=temperature, top_p=top_p, top_k=top_k)


def extract_channels(response: str) -> tuple[str, str] | None:
    """Extract analysis and final channels from GPT-OSS response.

    Parses response for <|channel|>analysis and <|channel|>final sections
    using tolerant regex that handles incomplete or malformed output.

    Args:
        response: Raw model output containing channel markers.

    Returns:
        Tuple of (analysis_text, final_text) if both found, None otherwise.
    """
    analysis_match = _ANALYSIS_PATTERN.search(response)
    if not analysis_match:
        return None

    final_match = _FINAL_PATTERN.search(response)
    if not final_match:
        return None

    return analysis_match.group(1).strip(), final_match.group(1).strip()


def formatted_template(
    user_prompt: str,
    system_prompt: str | None = None,
    reasoning_level: str | None = "medium",
) -> str:
    """Format GPT-OSS chat template with system context and reasoning level.

    Args:
        user_prompt: User's input text.
        system_prompt: System instruction. Defaults to ChatGPT persona.
        reasoning_level: Reasoning depth ("low", "medium", "high"). Defaults to "medium".

    Returns:
        Formatted prompt string ready for model input.
    """
    if not system_prompt:
        system_prompt = "You are ChatGPT, a large language model trained by OpenAI."
    today = get_today()
    return f"""<|start|>system<|message|>{system_prompt}
Knowledge cutoff: 2024-06
Current date: {today}
Reasoning: {reasoning_level}
<|end|>

<|start|>user<|message|>{user_prompt}<|end|><|start|>assistant"""


def main(model_path: str, reasoning_level: str, user_prompt: str) -> GenerationResult:
    """Run GPT-OSS generation with custom template.

    Args:
        model_path: Path to GGUF model file.
        reasoning_level: One of "low", "medium", "high".
        user_prompt: User input text.

    Returns:
        GenerationResult with analysis, final output, timing, and token counts.

    Raises:
        RuntimeError: If model fails to load.
        ValueError: If response format is unexpected.
    """
    config = gpt_oss_config(model_path)
    sampling = model_sampling()

    formatted = formatted_template(
        user_prompt=user_prompt, reasoning_level=reasoning_level
    )

    try:
        # Use context manager for proper cleanup
        with Llama(model_path, config=config, sampling=sampling) as llm:
            start = time.perf_counter()

            # Generate with stop strings to prevent starting new user turn
            response = llm.generate(
                formatted,
                max_tokens=DEFAULT_MAX_TOKENS,
                stop=["<|start|>user", "<|end|><|end|>"],
            )
            elapsed = time.perf_counter() - start

            # Use built-in API
            prompt_tokens = llm.n_tokens(formatted)
            response_tokens = llm.n_tokens(response)

            print(f"\n*** Prompt length : {len(formatted)}, {prompt_tokens}")
            print(f"\n*** Raw output length ({len(response)}, {response_tokens})")

            result = extract_channels(response)
            if not result:
                raise ValueError(
                    f"Unexpected response format. Got: {response[:200]}..."
                )

            return {
                "analysis": result[0],
                "final": result[1],
                "elapsed_time": elapsed,
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
            }
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GPT-OSS generation with custom template"
    )
    parser.add_argument(
        "--model", default="models/gpt-oss-20b-Q4_K_M.gguf", help="Path to GGUF model"
    )
    parser.add_argument(
        "--reasoning",
        default="high",
        choices=["low", "medium", "high"],
        help="Reasoning level",
    )
    parser.add_argument("--prompt", help="User prompt text")
    parser.add_argument(
        "--input-file", type=argparse.FileType("r"), help="Read prompt from file"
    )
    args = parser.parse_args()

    if args.input_file:
        user_prompt = args.input_file.read()
    elif args.prompt:
        user_prompt = args.prompt
    else:
        parser.error("Either --prompt or --input-file is required")

    user_prompt = (
        "Translate the following text into Simplified Chinese. "
        "Provide only the translation; no commentary.\n"
        f"{user_prompt}"
    )

    result = main(args.model, args.reasoning, user_prompt)
    print(f"\n*** Final result: {result['final']}")
    print(f"Execution time: {result['elapsed_time']:.2f} s")
