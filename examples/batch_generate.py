#!/usr/bin/env python3
# File              : batch_generate.py
# Author            : Lianghong Fei <feilianghong@gmail.com>
# Date              : 2025-12-07
# Last Modified Date: 2025-12-08
# Last Modified By  : Lianghong Fei <feilianghong@gmail.com>
"""Batch text generation with full GPU optimization and performance metrics."""

import argparse
import time
from pathlib import Path

from model_helper_utils import (
    detect_family,
    generate_with_model_stops,
    list_models,
    parse_output,
    recommended_generation_params,
    stop_strings_for_family,
)

from llama_cpp import Llama, LlamaConfig, SamplingParams, print_system_info
from llama_cpp._llama import chat_apply_template

parser = argparse.ArgumentParser(description="Batch text generation benchmark")
parser.add_argument("-m", "--model", help="Path to GGUF model file")
parser.add_argument("-l", "--list", action="store_true", help="List available models")
parser.add_argument(
    "--max-tokens", type=int, default=1024, help="Max tokens to generate"
)
parser.add_argument(
    "--think", action="store_true", help="Enable thinking mode (Qwen3 only)"
)
parser.add_argument(
    "--reasoning",
    choices=["low", "medium", "high"],
    default="low",
    help="Reasoning level for gpt-oss models (default: low)",
)
args = parser.parse_args()

if args.list:
    models = list_models()
    print(f"Available models ({len(models)}):")
    for m in models:
        fam = detect_family(m)
        print(f"  {m.name} [{fam}]")
    exit(0)

# Select model
if args.model:
    MODEL_PATH = args.model
else:
    models = list_models()
    MODEL_PATH = str(models[0]) if models else "models/Qwen3-8B-Q6_K.gguf"

# Check if model exists
if not Path(MODEL_PATH).exists():
    print(f"\n❌ Model not found: {MODEL_PATH}")
    print(f"\n💡 Tip: Use the full path, e.g., ./models/{Path(MODEL_PATH).name}")
    print(f"   Or list available models with: python {__file__} --list\n")
    exit(1)

# Detect family and get recommended params
family = detect_family(MODEL_PATH)
rec_params = recommended_generation_params(family)
stop_strings = stop_strings_for_family(family)

# System prompt (thinking control only for Qwen3)
SYSTEM_PROMPT = "You are a helpful assistant."
if family == "qwen3":
    if args.think:
        SYSTEM_PROMPT += "\n/think"
    else:
        SYSTEM_PROMPT += "\n/no_think\nProvide direct answers without showing your reasoning process."
elif family == "gpt-oss":
    SYSTEM_PROMPT += f"\n\nreasoning_effort: {args.reasoning}\n\n"

print("=== System Info ===")
print(print_system_info())

# GPU-optimized config
config = LlamaConfig(
    model_path=MODEL_PATH,
    n_ctx=8192,
    n_batch=1024,
    n_gpu_layers=-1,
    offload_kqv=True,
    flash_attn=1,
    verbose=False,
)

# Use recommended sampling params for the model family
sampling = SamplingParams(
    temperature=rec_params.get("temperature", 0.7),
    top_p=rec_params.get("top_p", 0.9),
    top_k=rec_params.get("top_k", 40),
)
with Llama(MODEL_PATH, config=config, sampling=sampling) as llm:
    # Show model info
    print("\n=== Model Info ===")
    meta = llm.metadata
    arch = meta.get("general.architecture", "")
    print(f"Name: {meta.get('general.name', 'N/A')}")
    print(f"Architecture: {arch}")
    print(f"Family: {family}")
    print(f"Model size: {llm.model_size() / 1024**3:.2f} GB")
    print(f"Parameters: {llm.n_params():,}")
    print(f"Layers: {llm.n_layer()}")
    print(f"Context: {llm.n_ctx()}")
    print(f"Vocab size: {llm.n_vocab()}")
    print(f"Embedding dim: {llm.n_embd()}")
    # Check for sliding window attention
    swa_key = f"{arch}.attention.sliding_window"
    swa = meta.get(swa_key)
    print(f"Sliding window: {swa if swa else 'N/A'}")
    print(f"Stop strings: {stop_strings}")
    print(f"Sampling: {rec_params}")
    if family == "qwen3":
        print(f"Thinking mode: {'enabled' if args.think else 'disabled'}")
    elif family == "gpt-oss":
        print(f"Reasoning level: {args.reasoning}")

    # Get and display chat template
    print("\n=== Chat Template ===")
    template = llm.get_chat_template()
    print(template[:200] + "..." if len(template) > 200 else template)

    # Demonstrate chat_apply_template with system prompt
    print("\n=== chat_apply_template Demo ===")
    chat_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    chat_messages.append({"role": "user", "content": "What is 2+2?"})
    print(f"Messages: {chat_messages}")

    messages_tuple = [(m["role"], m["content"]) for m in chat_messages]
    formatted = chat_apply_template(llm.model, messages_tuple, "", True)
    print(f"Formatted prompt:\n{formatted}")

    llm.reset()
    response = generate_with_model_stops(llm, formatted, family, max_tokens=1024)
    parsed = parse_output(response, family)
    print(f"Response: {parsed.content}")
    if parsed.reasoning:
        print(f"Reasoning: {parsed.reasoning[:200]}...")

    # Batch prompts
    prompts = [
        "Explain quantum computing in one sentence:",
        "Write a haiku about programming:",
        "What is the capital of France? Directly answer the name of the city.",
        "冬瓜、黄瓜、西瓜、南瓜都能吃，什么瓜不能吃？不要解释，直接给出答案。",
        "老王一天要刮四五十次脸，脸上却仍有胡子。这是什么原因？不要解释，直接给出答案。",
        "有一个字，人人见了都会念错。这是什么字？不要解释，直接给出答案。",
    ]

    print(f"\n=== Batch Generation ({len(prompts)} prompts) ===")
    print(f"System prompt: {SYSTEM_PROMPT}\n")

    results = []
    total_tokens = 0
    total_time = 0.0

    def format_prompt(user_prompt: str, system_prompt: str = None) -> str:
        """Format prompt with system message and chat template."""
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": user_prompt})
        msgs_tuple = [(m["role"], m["content"]) for m in msgs]
        return chat_apply_template(llm.model, msgs_tuple, "", True)

    for i, prompt in enumerate(prompts):
        llm.reset()
        formatted_prompt = format_prompt(prompt, SYSTEM_PROMPT)
        prompt_tokens = llm.n_tokens(formatted_prompt)

        start = time.perf_counter()
        text = generate_with_model_stops(
            llm, formatted_prompt, family, max_tokens=args.max_tokens
        )
        elapsed = time.perf_counter() - start

        parsed = parse_output(text, family)
        gen_tokens = llm.kv_cache_seq_pos_max() - prompt_tokens
        total_tokens += gen_tokens
        total_time += elapsed

        results.append((prompt_tokens, gen_tokens, elapsed))
        print(f"[{i + 1}] {prompt}")
        if parsed.content:
            print(f"    → {parsed.content}")
        elif parsed.reasoning:
            print(f"    → [thinking] {parsed.reasoning}")
        else:
            # Debug: show raw output when parsing fails
            print(f"    → [raw] {text if text.strip() else '[empty]'}")

    # Performance metrics
    print("\n=== Performance Metrics ===")
    for i, (p_tok, g_tok, elapsed) in enumerate(results):
        tps = g_tok / elapsed if elapsed > 0 else 0
        print(
            f"Prompt {i + 1}: {p_tok} prompt + {g_tok} gen in {elapsed * 1000:.1f}ms ({tps:.1f} t/s)"
        )

    avg_tps = total_tokens / total_time if total_time > 0 else 0
    print(
        f"\nTotal: {total_tokens} gen tokens in {total_time * 1000:.1f}ms ({avg_tps:.1f} t/s)"
    )
