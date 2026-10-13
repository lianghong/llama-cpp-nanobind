"""Measure three-turn streamed chat latency, continuation, and memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import llama_cpp._llama as _llama
from llama_cpp import Llama, LlamaConfig, disable_logging
from tools.benchmark_upstream import gpu_query, memory_snapshot
from tools.check_runtime import runtime_info

QUESTIONS = [
    "Explain why a local language model needs both RAM and GPU memory.",
    "Give a concrete example of how quantization changes those requirements.",
    "For that example, explain how a longer conversation changes memory use.",
]
SYSTEM = "You are a helpful assistant. Use concise, concrete explanations."


def percentile(values: list[float], percent: int) -> float:
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[percent - 1]


def conversation(llm: Llama, draft_max: int, budget: int) -> list[dict[str, Any]]:
    llm.kv_cache_clear()
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM}]
    turns = []
    for index, question in enumerate(QUESTIONS):
        messages.append({"role": "user", "content": question})
        prompt = llm._format_chat_messages(messages)
        tokens = llm.tokenize(prompt, add_special=True, parse_special=True)
        cached = list(llm._cached_prompt_tokens)
        matched = 0
        for old, new in zip(cached, tokens, strict=False):
            if old != new:
                break
            matched += 1
        llm.perf_reset()
        first_at = None
        content = []
        started = time.perf_counter()
        for chunk in llm.create_chat_completion(
            messages,
            max_tokens=budget,
            stream=True,
            reset_kv_cache=index == 0,
            cache_prompt=True,
            speculative=draft_max > 0,
            n_draft_max=draft_max or 1,
            temperature=0.0,
            seed=0,
        ):
            text = chunk["choices"][0]["delta"].get("content", "")
            if text:
                if first_at is None:
                    first_at = time.perf_counter()
                content.append(text)
        finished = time.perf_counter()
        if first_at is None:
            raise RuntimeError("chat benchmark produced no text")
        response = "".join(content)
        messages.append({"role": "assistant", "content": response})
        turns.append(
            {
                "turn": index + 1,
                "prompt_tokens": len(tokens),
                "prompt_sha256": hashlib.sha256(
                    json.dumps(tokens).encode()
                ).hexdigest(),
                "cached_tokens_before": len(cached),
                "matching_prefix_tokens": matched,
                "cached_tokens_after": len(llm._cached_prompt_tokens),
                "response": response,
                "first_text_seconds": first_at - started,
                "elapsed_seconds": finished - started,
                "utf8_bytes_per_second": len(response.encode()) / (finished - started),
                "native_perf": llm.perf(),
                "memory_after_turn": memory_snapshot(),
            }
        )
    return turns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--runs", default=5, type=int)
    parser.add_argument("--warmup", default=2, type=int)
    parser.add_argument("--max-tokens", default=64, type=int)
    parser.add_argument("--draft-max", default=[1, 2], type=int, nargs="*")
    parser.add_argument("--n-ctx", default=2048, type=int)
    parser.add_argument("--threads", default=4, type=int)
    args = parser.parse_args()
    if (
        min(args.runs, args.warmup, args.max_tokens, args.n_ctx, args.threads) < 1
        or any(not 1 <= width <= 8 for width in args.draft_max)
        or not args.model.is_file()
    ):
        parser.error("invalid model, counts, context, or draft width")
    disable_logging()
    model_hash = hashlib.sha256()
    with args.model.open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            model_hash.update(data)
    cfg = LlamaConfig(
        model_path=str(args.model),
        n_ctx=args.n_ctx,
        n_batch=512,
        n_ubatch=512,
        n_gpu_layers=-1,
        n_threads=args.threads,
        n_threads_batch=args.threads,
        n_rs_seq=max([2, *args.draft_max]),
        verbose=False,
    )
    root = Path(__file__).resolve().parents[1]
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "runtime": runtime_info(),
        "native_extension_sha256": hashlib.sha256(
            Path(_llama.__file__).read_bytes()
        ).hexdigest(),
        "binding_source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "CMakeLists.txt",
                "src/bindings/llama_cpp.cpp",
                "src/llama_cpp/llama.py",
                "tools/benchmark_mtp_chat.py",
                "tools/benchmark_upstream.py",
                "tools/check_runtime.py",
            )
        },
        "gpu": gpu_query("name,driver_version,memory.total"),
        "model_sha256": model_hash.hexdigest(),
        "config": asdict(cfg),
        "settings": vars(args) | {"model": str(args.model), "output": str(args.output)},
        "timing_scope": (
            "Loaded model, public streamed chat API; includes template processing, "
            "tokenization, sampler and thread creation, queue, and text decoding. "
            "First-text time excludes role-only/empty chunks. Memory is sampled "
            "after each completed turn, not peak memory. Byte throughput is exact "
            "UTF-8 output size divided by whole request duration."
        ),
        "system_prompt": SYSTEM,
        "questions": QUESTIONS,
        "modes": [],
    }
    expected = None
    for width in [0, *args.draft_max]:
        with Llama(str(args.model), config=cfg) as llm:
            if width and not llm.supports_speculative_mtp():
                raise RuntimeError("chat benchmark requires an MTP-capable model")
            warmup = [
                conversation(llm, width, args.max_tokens) for _ in range(args.warmup)
            ]
            runs = [conversation(llm, width, args.max_tokens) for _ in range(args.runs)]
            outputs = [turn["response"] for turn in runs[0]]
            if expected is None:
                expected = outputs
            matches = all(
                [turn["response"] for turn in run] == expected for run in warmup + runs
            )
            summaries = []
            for turn_index in range(len(QUESTIONS)):
                records = [run[turn_index] for run in runs]
                times = [row["first_text_seconds"] for row in records]
                summaries.append(
                    {
                        "turn": turn_index + 1,
                        "p50_first_text_seconds": statistics.median(times),
                        "p95_first_text_seconds": percentile(times, 95),
                        "median_elapsed_seconds": statistics.median(
                            row["elapsed_seconds"] for row in records
                        ),
                        "median_utf8_bytes_per_second": statistics.median(
                            row["utf8_bytes_per_second"] for row in records
                        ),
                    }
                )
            report["modes"].append(
                {
                    "draft_max": width,
                    "greedy_matches_ordinary": matches,
                    "summaries": summaries,
                    "warmup_runs": warmup,
                    "runs": runs,
                }
            )
            print(
                f"draft={width}: "
                + ", ".join(
                    f"turn={row['turn']} "
                    f"p50={row['p50_first_text_seconds'] * 1000:.1f}ms "
                    f"p95={row['p95_first_text_seconds'] * 1000:.1f}ms"
                    for row in summaries
                )
                + f" greedy_match={matches}",
                flush=True,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if any(not mode["greedy_matches_ordinary"] for mode in report["modes"]):
        raise SystemExit("greedy chat mismatch; evidence saved")


if __name__ == "__main__":
    main()
