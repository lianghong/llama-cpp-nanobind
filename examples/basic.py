#!/usr/bin/env python3
# File              : basic.py
# Author            : Lianghong Fei <feilianghong@gmail.com>
# Date              : 2026-02-11
# Last Modified Date: 2026-02-11
# Last Modified By  : Lianghong Fei <feilianghong@gmail.com>
"""Minimal inference example with context manager.

Usage:
    python examples/basic.py models/Qwen3-8B-Q6_K.gguf
    python examples/basic.py path/to/large-model.gguf --ngl 20 --ctx 2048
"""

import argparse
import re
import time

from llama_cpp import Llama
from llama_cpp import LlamaConfig
from llama_cpp import ModelLoadError
from llama_cpp import SamplingParams


DEFAULT_MODEL = "models/Qwen3.5-4B-Q4_K_M.gguf"

# Matches <think>...</think> with optional whitespace after.
# Also handles missing opening <think> (strips everything up to </think>).
_THINKING_RE = re.compile(r"(<think>.*?</think>\s*|^.*?</think>\s*)", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from reasoning model output."""
    return _THINKING_RE.sub("", text)


class ThinkingFilter:
    """Streaming filter that suppresses <think>...</think> content.

    Handles tags that arrive split across multiple chunks. Only text
    outside thinking blocks is emitted.

    Tutorial reimplementation of what ``UnifiedLLM.sanitize_history`` and
    ``generate_with_thinking`` do internally. Prefer those in real code;
    this is kept here to show the underlying state machine.
    """

    def __init__(self) -> None:
        self._inside = False
        self._buf = ""

    def feed(self, chunk: str) -> str:
        """Feed a chunk and return the portion that should be displayed."""
        self._buf += chunk
        out: list[str] = []

        while self._buf:
            if self._inside:
                # Look for closing tag
                end = self._buf.find("</think>")
                if end == -1:
                    # Still inside thinking — discard buffered content but
                    # keep a tail in case "</think>" straddles the boundary.
                    if len(self._buf) > len("</think>"):
                        self._buf = self._buf[-(len("</think>") - 1) :]
                    break
                # Found closing tag — skip everything up to and including it
                after = end + len("</think>")
                self._buf = self._buf[after:]
                self._inside = False
            else:
                # Look for opening tag
                start = self._buf.find("<think>")
                if start == -1:
                    # No opening tag — emit everything except a tail that
                    # could be a partial "<think>" tag.
                    safe = len(self._buf) - (len("<think>") - 1)
                    if safe > 0:
                        out.append(self._buf[:safe])
                        self._buf = self._buf[safe:]
                    break
                # Emit text before the tag, then enter thinking mode
                if start > 0:
                    out.append(self._buf[:start])
                self._buf = self._buf[start + len("<think>") :]
                self._inside = True

        return "".join(out)

    def flush(self) -> str:
        """Flush remaining buffer (call after stream ends)."""
        if self._inside:
            # Unclosed <think> — discard remaining thinking content
            self._buf = ""
            return ""
        text = self._buf
        self._buf = ""
        return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal inference example")
    parser.add_argument(
        "model",
        nargs="?",
        default=DEFAULT_MODEL,
        help="Path to GGUF model (default: %(default)s)",
    )
    parser.add_argument(
        "--ngl", type=int, default=-1, help="GPU layers (-1=all, 0=CPU only)"
    )
    parser.add_argument(
        "--ctx", type=int, default=4096, help="Context size (default: 4096)"
    )
    args = parser.parse_args()

    total_tokens = 0
    start = time.perf_counter()

    try:
        config = LlamaConfig(
            model_path=args.model,
            n_ctx=args.ctx,
            n_gpu_layers=args.ngl,
        )
        # Use context manager for automatic resource cleanup
        with Llama(args.model, config=config) as llm:
            print("=== Single-call generation ===")
            t0 = time.perf_counter()
            text = llm.generate("Write a short greeting from a GPU.", max_tokens=512)
            dt = time.perf_counter() - t0
            n = llm.n_tokens(text)
            total_tokens += n
            print(strip_thinking(text))
            print(f"  [{n} tokens, {n / dt:.1f} tok/s]")

            print("\n=== True streaming (yields as tokens decode) ===")
            sampling = SamplingParams(temperature=0.7, top_p=0.9, repeat_penalty=1.05)
            filt = ThinkingFilter()
            t0 = time.perf_counter()
            n = 0
            for chunk in llm.generate_stream(
                "Name three oceans:", max_tokens=1024, sampling=sampling
            ):
                n += 1  # one chunk per token
                out = filt.feed(chunk)
                if out:
                    print(out, end="", flush=True)
            out = filt.flush()
            if out:
                print(out, end="", flush=True)
            dt = time.perf_counter() - t0
            total_tokens += n
            print(f"\n  [{n} tokens, {n / dt:.1f} tok/s]")

            print("\n=== Session continuation (reuse KV cache) ===")
            t0 = time.perf_counter()
            text1 = llm.generate("Hello", max_tokens=64, reset_kv_cache=True)
            text2 = llm.generate(
                "Continue the greeting:", max_tokens=512, reset_kv_cache=False
            )
            dt = time.perf_counter() - t0
            n = llm.n_tokens(text1) + llm.n_tokens(text2)
            total_tokens += n
            print(strip_thinking(text2))
            print(f"  [{n} tokens, {n / dt:.1f} tok/s]")

    except ModelLoadError as e:
        print(f"Failed to load model: {e}")
        return

    elapsed = time.perf_counter() - start
    print(
        f"\nTotal: {total_tokens} tokens, {total_tokens / elapsed:.1f} tok/s (including model load)"
    )
    print(f"Execution time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
