"""Sweep MTP draft widths over native prompts and three-turn cached chat."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import llama_cpp._llama as _llama
from llama_cpp import Llama, LlamaConfig, disable_logging
from tools.benchmark_memory import GpuMemorySampler
from tools.benchmark_mtp_chat import QUESTIONS, SYSTEM, percentile
from tools.benchmark_upstream import generation_run, gpu_query, prompt_tokens
from tools.check_runtime import runtime_info


def token_hash(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps(tokens).encode()).hexdigest()


def first_message(llm: Llama, count: int) -> dict[str, Any]:
    notes = "\n".join(
        f"Deployment {i}: a {4 + i % 4 * 3}B parameter model uses "
        f"{[4, 8, 16][i % 3]}-bit weights and a {2048 * (1 + i % 4)}-token "
        "context. Model weights, KV cache, temporary buffers, and concurrent "
        "requests all contribute to memory use. Quantization trades precision "
        "for storage; GPU offload trades device capacity for bandwidth."
        for i in range(1, count // 20 + 30)
    )
    padding = llm.tokenize(notes, add_special=False)
    base = "Reference deployment notes:\n\n\n" + QUESTIONS[0]
    overhead = len(
        llm.tokenize(
            llm._format_chat_messages(
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": base},
                ]
            ),
            add_special=True,
            parse_special=True,
        )
    )
    size = count - overhead
    for _ in range(12):
        body = llm.detokenize(padding[:size])
        message = {
            "role": "user",
            "content": f"Reference deployment notes:\n{body}\n\n{QUESTIONS[0]}",
        }
        actual = len(
            llm.tokenize(
                llm._format_chat_messages(
                    [{"role": "system", "content": SYSTEM}, message]
                ),
                add_special=True,
                parse_special=True,
            )
        )
        if actual == count:
            return message
        size += count - actual
    raise RuntimeError(f"cannot construct an exact {count}-token chat prompt")


def chat_run(
    llm: Llama,
    message: dict[str, Any],
    width: int,
    budget: int,
    memory: GpuMemorySampler,
) -> list[dict[str, Any]]:
    llm.kv_cache_clear()
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM}]
    turns = []
    generated_ids: list[int] = []
    original_commit = llm._commit_generation_to_cache

    def capture_commit(generated: Sequence[int]) -> None:
        # Observe exact IDs even when an EOG clears the retained cache.
        generated_ids.extend(generated)
        original_commit(generated)

    llm._commit_generation_to_cache = capture_commit  # type: ignore[method-assign]
    try:
        for index in range(len(QUESTIONS)):
            messages.append(
                message if index == 0 else {"role": "user", "content": QUESTIONS[index]}
            )
            tokens = llm.tokenize(
                llm._format_chat_messages(messages),
                add_special=True,
                parse_special=True,
            )
            cached = list(llm._cached_prompt_tokens)
            matched = 0
            for a, b in zip(cached, tokens, strict=False):
                if a != b:
                    break
                matched += 1
            llm.perf_reset()
            generated_ids.clear()
            content = []
            first_at = None
            memory.start()
            try:
                started = time.perf_counter()
                for chunk in llm.create_chat_completion(
                    messages,
                    max_tokens=budget,
                    stream=True,
                    reset_kv_cache=index == 0,
                    cache_prompt=True,
                    speculative=width > 0,
                    n_draft_max=width or 1,
                    temperature=0.0,
                    seed=0,
                ):
                    text = chunk["choices"][0]["delta"].get("content", "")
                    if text:
                        if first_at is None:
                            first_at = time.perf_counter()
                        content.append(text)
                finished = time.perf_counter()
            finally:
                peak = memory.finish()
            if first_at is None or not generated_ids:
                raise RuntimeError("chat benchmark produced no text or token IDs")
            response = "".join(content)
            perf = llm.perf()
            drafted, accepted = int(perf["n_drafted"]), int(perf["n_accepted"])
            turns.append(
                {
                    "turn": index + 1,
                    "prompt_tokens": len(tokens),
                    "prompt_sha256": token_hash(tokens),
                    "tokens": list(generated_ids),
                    "response": response,
                    "matching_prefix_tokens": matched,
                    "cached_tokens_before": len(cached),
                    "cached_tokens_after": len(llm._cached_prompt_tokens),
                    "elapsed_seconds": finished - started,
                    "first_text_seconds": first_at - started,
                    "tokens_per_second": len(generated_ids) / (finished - started),
                    "utf8_bytes_per_second": len(response.encode())
                    / (finished - started),
                    "acceptance_rate": accepted / drafted if drafted else None,
                    "native_perf": perf,
                    "memory": peak,
                }
            )
            messages.append({"role": "assistant", "content": response})
    finally:
        llm._commit_generation_to_cache = original_commit  # type: ignore[method-assign]
    return turns


def summarize(records: list[dict[str, Any]], latency_key: str) -> dict[str, Any]:
    latency = [r[latency_key] for r in records]
    drafts = [
        r.get("drafted", r.get("native_perf", {}).get("n_drafted", 0)) for r in records
    ]
    accepted = [
        r.get("accepted", r.get("native_perf", {}).get("n_accepted", 0))
        for r in records
    ]
    return {
        "p50_first_seconds": statistics.median(latency),
        "p95_first_seconds": percentile(latency, 95),
        "median_tokens_per_second": statistics.median(
            r["tokens_per_second"] for r in records
        ),
        "median_elapsed_seconds": statistics.median(
            r["elapsed_seconds"] for r in records
        ),
        "acceptance_rate": sum(accepted) / sum(drafts) if sum(drafts) else None,
        "sampled_peak_gpu_process_bytes": max(
            r["memory"]["sampled_peak_gpu_process_bytes"] for r in records
        ),
    }


def annotate_matches(report: dict[str, Any]) -> bool:
    successful = True
    for count in report["settings"]["prompt_tokens"]:
        cases = [c for c in report["cases"] if c["requested_prompt_tokens"] == count]
        baseline = next((c for c in cases if c["draft_max"] == 0), None)
        if baseline is None:
            continue
        for case in cases:
            for scope in report["settings"]["scope"]:
                ordinary = baseline[scope]["runs"][0]
                records = case[scope]["warmup_runs"] + case[scope]["runs"]
                expected = ordinary if scope == "chat" else [ordinary]
                matches = []
                for run in records:
                    actual = run if scope == "chat" else [run]
                    matches.append(
                        len(actual) == len(expected)
                        and all(
                            a["tokens"] == b["tokens"]
                            and (
                                scope != "chat"
                                or (
                                    a["response"] == b["response"]
                                    and a["prompt_sha256"] == b["prompt_sha256"]
                                )
                            )
                            for a, b in zip(actual, expected, strict=True)
                        )
                    )
                case[scope]["greedy_matches_ordinary"] = all(matches)
                case[scope]["matching_runs"] = matches
                successful = successful and all(matches)
    return successful


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--prompt-tokens", nargs="+", type=int, default=[512, 2048, 4096]
    )
    parser.add_argument("--draft-max", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument(
        "--scope", nargs="+", choices=["native", "chat"], default=["native", "chat"]
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if (
        min(args.runs, args.warmup, args.max_tokens, args.threads) < 1
        or min(args.prompt_tokens) < 128
        or max(args.prompt_tokens) + 3 * args.max_tokens + 256 >= args.n_ctx
        or any(not 1 <= w <= 4 for w in args.draft_max)
        or len(set(args.prompt_tokens)) != len(args.prompt_tokens)
        or len(set(args.draft_max)) != len(args.draft_max)
        or len(set(args.scope)) != len(args.scope)
        or not args.model.is_file()
    ):
        parser.error("invalid model, counts, context budget, or draft widths")
    disable_logging()
    hasher = hashlib.sha256()
    with args.model.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    root = Path(__file__).resolve().parents[1]
    cfg = LlamaConfig(
        model_path=str(args.model),
        n_ctx=args.n_ctx,
        n_batch=512,
        n_ubatch=512,
        n_rs_seq=4,
        n_threads=args.threads,
        n_threads_batch=args.threads,
        n_gpu_layers=-1,
        verbose=False,
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "runtime": runtime_info(),
        "model_sha256": hasher.hexdigest(),
        "native_extension_sha256": hashlib.sha256(
            Path(_llama.__file__).read_bytes()
        ).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "src/bindings/llama_cpp.cpp",
                "src/llama_cpp/llama.py",
                "tools/benchmark_mtp_sweep.py",
                "tools/benchmark_memory.py",
                "tools/benchmark_mtp_chat.py",
                "tools/benchmark_upstream.py",
                "tools/check_runtime.py",
                "CMakeLists.txt",
            )
        },
        "gpu": gpu_query("name,driver_version,memory.total"),
        "config": asdict(cfg),
        "settings": vars(args) | {"model": str(args.model), "output": str(args.output)},
        "timing_scope": (
            "Native: fixed token IDs, excludes sampler creation and text processing; "
            "chat: public streamed API including text processing, exact IDs observed "
            "at cache commit. Chat first latency means first nonempty text. All rates "
            "include prefill and full request duration. Loaded models; cold requests "
            "are recorded separately as warmups. Memory: per-process NVML across all "
            "physical GPUs, requested 10ms interval, inference only. Sampled peaks "
            "are lower bounds and may miss shorter allocations; actual gaps recorded."
        ),
        "system_prompt": SYSTEM,
        "questions": QUESTIONS,
        "completed": False,
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    memory = GpuMemorySampler()
    try:
        modes = [0, *args.draft_max]
        for index, count in enumerate(args.prompt_tokens):
            # Rotate mode order across lengths to reduce consistent ordering bias.
            offset = (2 * index) % len(modes)
            for width in modes[offset:] + modes[:offset]:
                with Llama(str(args.model), config=cfg) as llm:
                    if width and not llm.supports_speculative_mtp():
                        raise RuntimeError("sweep requires an MTP-capable checkpoint")
                    case: dict[str, Any] = {
                        "requested_prompt_tokens": count,
                        "draft_max": width,
                    }
                    tokens = prompt_tokens(llm, count)
                    message = (
                        first_message(llm, count) if "chat" in args.scope else None
                    )
                    if message is not None:
                        case["first_user_message"] = message
                    for scope in args.scope:
                        records: list[Any] = []
                        run: Any
                        for _ in range(args.warmup + args.runs):
                            if scope == "native":
                                memory.start()
                                try:
                                    run = generation_run(
                                        llm, tokens, args.max_tokens, width
                                    )
                                finally:
                                    peak = memory.finish()
                                run["memory"] = peak
                                run["prompt_sha256"] = token_hash(tokens)
                            else:
                                assert message is not None
                                run = chat_run(
                                    llm, message, width, args.max_tokens, memory
                                )
                            records.append(run)
                        timed = records[args.warmup :]
                        groups = (
                            [[r[t] for r in timed] for t in range(len(QUESTIONS))]
                            if scope == "chat"
                            else [timed]
                        )
                        summaries = [
                            summarize(
                                group,
                                "first_text_seconds"
                                if scope == "chat"
                                else "first_token_seconds",
                            )
                            for group in groups
                        ]
                        case[scope] = {
                            "warmup_runs": records[: args.warmup],
                            "runs": timed,
                            "summaries": summaries,
                        }
                        print(
                            f"prompt={count} draft={width} {scope}: "
                            + ", ".join(
                                f"{s['median_tokens_per_second']:.1f}tok/s "
                                f"first={s['p50_first_seconds'] * 1000:.1f}ms"
                                for s in summaries
                            ),
                            flush=True,
                        )
                    report["cases"].append(case)
                    annotate_matches(report)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        memory.close()
    report["completed"] = True
    report["exact_greedy_pass"] = annotate_matches(report)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["exact_greedy_pass"]:
        raise SystemExit("exact greedy mismatch; evidence saved in report")


if __name__ == "__main__":
    main()
