"""Repeatable Linux inference benchmark; run with python -m tools.benchmark_upstream."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from contextlib import suppress
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import llama_cpp._llama as _llama
from llama_cpp import Llama, LlamaConfig, SamplingParams, disable_logging
from tools.check_runtime import runtime_info

PROMPT = (
    "Explain how language models balance throughput, memory, and accuracy. "
    "Use concrete examples and describe the tradeoffs step by step. "
)


def gpu_query(fields: str, *, processes: bool = False) -> str | None:
    query = "--query-compute-apps" if processes else "--query-gpu"
    try:
        return subprocess.check_output(
            ["nvidia-smi", f"{query}={fields}", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
    except OSError, subprocess.SubprocessError:
        return None


def memory_snapshot() -> dict[str, int | None]:
    rss = None
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            rss = int(line.split()[1]) * 1024
    usage = gpu_query("pid,used_memory", processes=True)
    gpu_bytes = None
    if usage is not None:
        readings = []
        for row in usage.splitlines():
            fields = [field.strip() for field in row.split(",")]
            if len(fields) == 2 and fields[0] == str(os.getpid()):
                with suppress(ValueError):
                    readings.append(int(fields[1]) * 1024 * 1024)
        if readings:
            gpu_bytes = sum(readings)
    return {"rss_bytes": rss, "gpu_process_bytes": gpu_bytes}


def prompt_tokens(llm: Llama, count: int) -> list[int]:
    tokens = llm.tokenize(PROMPT, add_special=True)
    repeats = count // len(tokens) + 2
    tokens = llm.tokenize(PROMPT * repeats, add_special=True)
    # Repetition can tokenize more compactly than the standalone sentence.
    while len(tokens) < count:
        repeats *= 2
        tokens = llm.tokenize(PROMPT * repeats, add_special=True)
    return tokens[:count]


def generation_run(
    llm: Llama, tokens: list[int], max_tokens: int, draft_max: int
) -> dict[str, Any]:
    llm.kv_cache_clear()
    llm.perf_reset()
    sampler = llm._build_sampler(
        SamplingParams(seed=0, temperature=0.0, n_draft_max=draft_max or 1)
    )
    first_at: float | None = None
    output: list[int] = []

    def on_token(token: int) -> bool:
        nonlocal first_at
        if first_at is None:
            first_at = time.perf_counter()
        output.append(token)
        return True

    started = time.perf_counter()
    if draft_max:
        returned = list(
            _llama.generate_tokens_speculative_mtp(
                llm.ctx,
                sampler,
                None,
                tokens,
                max_tokens,
                False,
                llm.model.eos(),
                draft_max,
                [],
                on_token,
                0,
            )
        )
    else:
        count = _llama.generate_tokens_streaming(
            llm.ctx,
            sampler,
            tokens,
            max_tokens,
            False,
            llm.model.eos(),
            [],
            on_token,
        )
    finished = time.perf_counter()
    if (draft_max and returned != output) or (not draft_max and count != len(output)):
        raise RuntimeError("native output differs from streamed token callbacks")
    if first_at is None:
        raise RuntimeError("benchmark produced no output tokens")
    elapsed = finished - started
    decode_seconds = finished - first_at
    counters = llm.perf()
    drafted = int(counters["n_drafted"])
    accepted = int(counters["n_accepted"])
    return {
        "tokens": output,
        "elapsed_seconds": elapsed,
        "first_token_seconds": first_at - started,
        "tokens_per_second": len(output) / elapsed,
        "decode_tokens_per_second": (
            (len(output) - 1) / decode_seconds if len(output) > 1 else None
        ),
        "drafted": drafted,
        "accepted": accepted,
        "verify_rounds": int(counters["n_verify"]),
        "mtp_graph_switches": counters.get("n_mtp_graph_switches"),
        "acceptance_rate": accepted / drafted if drafted else None,
    }


def benchmark_model(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    model_hash = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            model_hash.update(chunk)
    result: dict[str, Any] = {
        "model": str(path.resolve()),
        "model_bytes": path.stat().st_size,
        "model_sha256": model_hash.hexdigest(),
        "workloads": [],
    }
    cfg = LlamaConfig(
        model_path=str(path),
        n_ctx=args.n_ctx,
        n_batch=args.n_batch,
        n_ubatch=args.n_batch,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=args.threads,
        n_threads_batch=args.threads,
        n_rs_seq=max([2, *args.draft_max]),
        verbose=False,
    )
    result["config"] = asdict(cfg)
    # Close between modes so the ordinary memory measurement has no draft context.
    modes = [0, *args.draft_max]
    baselines: dict[int, list[int]] = {}
    for draft_max in modes:
        with Llama(str(path), config=cfg) as llm:
            if draft_max and not llm.supports_speculative_mtp():
                result["mtp_available"] = False
                if llm.mtp_predict_layers() > 0:
                    raise RuntimeError(
                        "model declares MTP layers but draft context failed"
                    )
                break
            if draft_max:
                result["mtp_available"] = True
            for count in args.prompt_tokens:
                tokens = prompt_tokens(llm, count)
                prefill: list[float] = []
                if not draft_max:
                    for iteration in range(args.warmup + args.runs):
                        llm.kv_cache_clear()
                        started = time.perf_counter()
                        llm.ctx.decode(tokens)
                        llm.ctx.logits()  # synchronize before reading the clock
                        elapsed = time.perf_counter() - started
                        if iteration >= args.warmup:
                            prefill.append(elapsed)
                warmup_runs = [
                    generation_run(llm, tokens, args.max_tokens, draft_max)
                    for _ in range(args.warmup)
                ]
                runs = [
                    generation_run(llm, tokens, args.max_tokens, draft_max)
                    for _ in range(args.runs)
                ]
                output = runs[0]["tokens"]
                stable = all(run["tokens"] == output for run in runs)
                if not draft_max:
                    baselines[count] = output
                matches = all(run["tokens"] == baselines[count] for run in runs)
                summary: dict[str, Any] = {
                    "prompt_tokens": count,
                    "prompt_sha256": hashlib.sha256(
                        json.dumps(tokens).encode()
                    ).hexdigest(),
                    "draft_max": draft_max,
                    "greedy_stable": stable,
                    "greedy_matches_baseline": matches,
                    "memory_after_run": memory_snapshot(),
                    "prefill_tokens_per_second": (
                        count / statistics.median(prefill) if prefill else None
                    ),
                    "runs": runs,
                    "warmup_runs": warmup_runs,
                }
                for key in (
                    "elapsed_seconds",
                    "first_token_seconds",
                    "tokens_per_second",
                    "decode_tokens_per_second",
                    "acceptance_rate",
                ):
                    values = [run[key] for run in runs if run[key] is not None]
                    summary[f"median_{key}"] = (
                        statistics.median(values) if values else None
                    )
                result["workloads"].append(summary)
                print(
                    f"{path.name}: prompt={count} draft={draft_max} "
                    f"{summary['median_tokens_per_second']:.1f} tok/s "
                    f"first={summary['median_first_token_seconds'] * 1000:.1f}ms "
                    f"greedy_match={matches}",
                    flush=True,
                )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--draft-max", type=int, nargs="*", default=[1, 2])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--n-batch", type=int, default=512)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if (
        min(args.runs, args.warmup, args.max_tokens, args.n_batch, args.threads) < 1
        or min(args.prompt_tokens) < 1
        or max(args.prompt_tokens) + args.max_tokens > args.n_ctx
        or any(not 1 <= n <= 8 for n in args.draft_max)
        or any(not path.is_file() for path in args.model)
    ):
        parser.error("invalid counts, context budget, draft width, or model path")
    disable_logging()
    root = Path(__file__).resolve().parents[1]
    report: dict[str, Any] = {
        "schema_version": 2,
        "created_at": datetime.now(UTC).isoformat(),
        "runtime": runtime_info(),
        "native_extension_sha256": hashlib.sha256(
            Path(_llama.__file__).read_bytes()
        ).hexdigest(),
        "gpu": gpu_query("name,driver_version,memory.total"),
        "binding_source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "CMakeLists.txt",
                "src/bindings/llama_cpp.cpp",
                "src/llama_cpp/llama.py",
                "tools/benchmark_upstream.py",
                "tools/check_runtime.py",
            )
        },
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "model"
        },
        "timing_scope": (
            "Warmed native streaming from fixed token IDs; excludes model load, "
            "tokenization, sampler creation, and detokenization. First-token time "
            "includes prefill; decode rate counts subsequent tokens. Memory is "
            "a point sample after generation, not peak usage."
        ),
        "models": [benchmark_model(path, args) for path in args.model],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if any(
        not workload["greedy_matches_baseline"] or not workload["greedy_stable"]
        for model in report["models"]
        for workload in model["workloads"]
    ):
        raise SystemExit("greedy output mismatch; evidence saved in the JSON report")


if __name__ == "__main__":
    main()
