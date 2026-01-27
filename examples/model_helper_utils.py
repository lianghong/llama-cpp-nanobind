#!/usr/bin/env python3
"""
Utility helpers for llama.cpp nanobind chat I/O across multiple templates.

Features:
- Detect model family from a GGUF file (no tensor loading; fast and safe).
- Prepare stop strings to pair with `llama.apply_chat_template`.
- Parse generated text into content / reasoning / tool calls per family.
- Small CLI demo that inspects models in ./models.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

_RE_TOOL_JSON = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_RE_GPTOSS_TOOL = re.compile(
    r"to=functions\.([^(<]+)<\|channel\|>commentary<\|message\|>(.*?)<\|call\|>",
    re.S,
)
_RE_REASONING = re.compile(r"Reasoning:\s*(low|medium|high)", re.I)


@dataclass
class ParsedOutput:
    """Normalized model output."""

    content: str
    reasoning: str | None = None
    tool_calls: list[str] | None = None
    raw: str | None = None


# -------------------------- model detection --------------------------- #


def _read_str_field(reader, key: str) -> str | None:
    field = reader.fields.get(key)
    if not field:
        return None
    try:
        return bytes(field.parts[-1]).decode()
    except Exception:
        return None


def detect_family(model_path: str | Path) -> str:
    """Return model family: mistral3|qwen3|gpt-oss|granite|gemma|phi|command-r|unknown."""
    path = Path(model_path)
    try:
        import gguf
    except ImportError as exc:
        raise RuntimeError("gguf is required for detect_family") from exc

    orig = gguf.GGUFReader._build_tensors
    gguf.GGUFReader._build_tensors = lambda self, offs, tf: None  # type: ignore[assignment,method-assign]
    try:
        reader = gguf.GGUFReader(path)
        arch = (_read_str_field(reader, "general.architecture") or "").lower()
        name = (_read_str_field(reader, "general.name") or "").lower()
    finally:
        gguf.GGUFReader._build_tensors = orig  # type: ignore[method-assign]

    if "mistral" in arch or "minis" in name:
        return "mistral3"
    if "qwen" in arch:
        return "qwen3"
    if any(x in arch for x in ("gpt-oss", "gpt_oss", "gptoss")):
        return "gpt-oss"
    if "granite" in arch:
        return "granite"
    if "gemma" in arch:
        return "gemma"
    if "phi" in arch or "phi" in name:
        return "phi"
    if "command-r" in arch or "aya" in name:
        return "command-r"
    return "unknown"


# -------------------------- stop strings ------------------------------ #

_STOP_STRINGS = {
    "mistral3": ["</s>", "<|im_end|>", "<|im_start|>"],
    "qwen3": ["<|im_end|>", "<|im_start|>"],
    "gpt-oss": ["<|end|>", "<|return|>", "<|start|>"],
    "granite": ["<|end_of_text|>"],
    "gemma": ["<end_of_turn>", "<|im_end|>", "<|im_start|>"],
    "phi": ["<|end|>", "<|endoftext|>"],
    "command-r": ["<|END_OF_TURN_TOKEN|>"],
}


def stop_strings_for_family(family: str) -> list[str]:
    return _STOP_STRINGS.get(family.lower(), []).copy()


# -------------------------- parsers ----------------------------------- #


def parse_mistral3(text: str) -> ParsedOutput:
    raw = text
    text = text.replace("</s>", "").replace("<|im_end|>", "").strip()
    tool_chunks = [
        m.group(0) for m in re.finditer(r"\[TOOL_CALLS].+?\[ARGS]\{.*?\}", text, re.S)
    ]
    cleaned = re.sub(
        r"\[TOOL_CALLS].*?\[ARGS].*?(?=\[TOOL_CALLS]|$)", "", text, flags=re.S
    ).strip()
    return ParsedOutput(content=cleaned, tool_calls=tool_chunks or None, raw=raw)


def parse_qwen3(text: str) -> ParsedOutput:
    raw = text
    body = text.strip()

    # Remove all Qwen3 special tokens
    body = re.sub(r"<\|im_end\|>", "", body)
    body = re.sub(r"<\|im_start\|>[^\n]*", "", body)
    body = re.sub(r"<\|endoftext\|>", "", body)
    body = re.sub(r"/no_think", "", body)
    body = re.sub(r"/think", "", body)

    # Extract and remove thinking blocks (both <thinking> and <> formats)
    reasoning = None

    # Handle <thinking> tags
    think_match = re.search(r"<think(?:ing)?>(.*?)(?:</think(?:ing)?>|$)", body, re.S)
    if think_match:
        reasoning = think_match.group(1).strip()
        body = re.sub(r"<think(?:ing)?>.*?(?:</think(?:ing)?>|$)", "", body, flags=re.S)

    # Handle <> reasoning markers (Qwen3 compact format)
    compact_think = re.search(r"<>(.*?)(?=\n\n[^\s]|\Z)", body, re.S)
    if compact_think and not reasoning:
        reasoning = compact_think.group(1).strip()
        body = re.sub(r"<>.*?(?=\n\n[^\s]|\Z)", "", body, flags=re.S)

    # Remove "Reasoning: <>" prefix and trailing artifacts
    body = re.sub(r"^Reasoning:\s*<>\s*", "", body)
    body = re.sub(r"<ing>\s*$", "", body)  # Remove trailing <ing>
    body = re.sub(r"</?[a-z]+>\s*$", "", body)  # Remove other trailing tags

    body = body.strip()
    tool_calls = _RE_TOOL_JSON.findall(body) or None

    return ParsedOutput(
        content=body, reasoning=reasoning, tool_calls=tool_calls, raw=raw
    )


def parse_gptoss(text: str) -> ParsedOutput:
    raw = text
    analysis = None
    content = text

    # Remove stop tokens first
    for token in ["<|return|>", "<|end|>", "<|start|>"]:
        content = content.replace(token, "")

    # Try to extract final channel content
    final_match = re.search(
        r"<\|channel\|>final<\|message\|>(.*?)(?:<\|channel\||$)", content, re.S
    )
    if final_match:
        content = final_match.group(1).strip()

    # Try to extract analysis channel
    analysis_match = re.search(
        r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|channel\||$)", text, re.S
    )
    if analysis_match:
        analysis = analysis_match.group(1).strip()

    # If no channels found, use everything
    if not final_match:
        content = content.strip()

    tool_calls = _RE_GPTOSS_TOOL.findall(text) or None
    return ParsedOutput(
        content=content, reasoning=analysis, tool_calls=tool_calls, raw=raw
    )


def parse_granite(text: str) -> ParsedOutput:
    raw = text
    content = text.split("<|end_of_text|>", 1)[0]
    tool_calls = _RE_TOOL_JSON.findall(content) or None
    cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.S).strip()
    return ParsedOutput(content=cleaned, tool_calls=tool_calls, raw=raw)


def parse_gemma(text: str) -> ParsedOutput:
    content = text.split("<end_of_turn>", 1)[0]
    content = content.split("<|im_end|>", 1)[0]
    content = content.split("<|im_start|>", 1)[0]
    # Remove trailing partial tokens and artifacts
    content = re.sub(r"\|[a-z_]*\|?>$", "", content)
    content = content.strip()
    return ParsedOutput(content=content, raw=text)


def parse_phi(text: str) -> ParsedOutput:
    return ParsedOutput(
        content=text.replace("<|end|>", "").replace("<|endoftext|>", "").strip(),
        raw=text,
    )


def parse_command_r(text: str) -> ParsedOutput:
    content = text.split("<|END_OF_TURN_TOKEN|>", 1)[0]
    content = content.replace("<|im_end|>", "").strip()
    return ParsedOutput(content=content, raw=text)


_PARSERS = {
    "mistral3": parse_mistral3,
    "qwen3": parse_qwen3,
    "gpt-oss": parse_gptoss,
    "granite": parse_granite,
    "gemma": parse_gemma,
    "phi": parse_phi,
    "command-r": parse_command_r,
}


def parse_output(text: str, family: str) -> ParsedOutput:
    parser = _PARSERS.get(family.lower())
    return parser(text) if parser else ParsedOutput(content=text.strip(), raw=text)


# ---------------------- recommended settings -------------------------- #

_RECOMMENDED_PARAMS = {
    "mistral3": {"temperature": 0.15, "top_p": 1.0},
    "qwen3": {"temperature": 0.7, "top_p": 0.8, "top_k": 20},
    "qwen3_thinking": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
    "granite": {"temperature": 0.0, "top_p": 1.0, "top_k": 0},
    "gpt-oss": {"temperature": 1.0, "top_p": 1.0},
    "gemma": {"temperature": 0.7, "top_p": 0.95},
    "phi": {"temperature": 0.7, "top_p": 0.9},
    "command-r": {"temperature": 0.3, "top_p": 0.75},
}


def recommended_generation_params(family: str, *, thinking: bool = False) -> dict:
    """Return recommended sampling parameters for a family."""
    fam = family.lower()
    if fam == "qwen3" and thinking:
        return _RECOMMENDED_PARAMS["qwen3_thinking"].copy()
    return _RECOMMENDED_PARAMS.get(fam, {}).copy()


# -------------------------- prompt prep -------------------------------- #


def _update_system_message(
    messages: list[dict], directive: str, pattern: re.Pattern | None = None
) -> list[dict]:
    """Helper to add/update directive in system message."""
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if pattern:
                content = pattern.sub("", content)
            content = (
                content.replace(directive.split(":")[0] + ":", "").strip()
                if ":" in directive
                else content
            )
            # Remove existing /think /no_think
            content = content.replace("/think", "").replace("/no_think", "").strip()
            msg["content"] = f"{content}\n{directive}".strip()
            return messages
    messages.insert(0, {"role": "system", "content": directive})
    return messages


def build_generation_kwargs(
    family: str, extra_stop: Iterable[str] | None = None
) -> dict:
    stops = stop_strings_for_family(family)
    if extra_stop:
        stops.extend(s for s in extra_stop if s not in stops)
    return {"stop": stops} if stops else {}


def set_thinking_mode(
    messages: list[dict], family: str, enable: bool = True
) -> list[dict]:
    """Add /think or /no_think directive for Qwen3 models."""
    if family.lower() != "qwen3":
        return messages
    return _update_system_message(messages, "/think" if enable else "/no_think")


def set_reasoning_level(
    messages: list[dict], family: str, level: str = "medium"
) -> list[dict]:
    """Set reasoning level (low/medium/high) for gpt-oss models."""
    if family.lower() != "gpt-oss":
        return messages
    level = level.lower() if level.lower() in ("low", "medium", "high") else "medium"
    return _update_system_message(messages, f"Reasoning: {level}", _RE_REASONING)


def apply_template(
    llama,
    messages: list[dict],
    tools: list | None = None,
    add_generation_prompt: bool = True,
    **template_kwargs,
) -> str:
    """Thin wrapper around llama.apply_chat_template."""
    kwargs = {"tools": tools} if tools else {}
    kwargs.update(template_kwargs)
    return llama.apply_chat_template(
        messages, add_generation_prompt=add_generation_prompt, **kwargs
    )


def generate_with_model_stops(
    llama,
    prompt: str,
    family: str,
    max_tokens: int = 128,
    **kwargs,
) -> str:
    """Generate text using the model's default stop strings."""
    stops = stop_strings_for_family(family)
    return llama.generate(prompt, max_tokens=max_tokens, stop=stops or None, **kwargs)


def list_models(models_dir: Path | str | None = None) -> list[Path]:
    """List all GGUF models in the models directory."""
    if models_dir is None:
        script_dir = Path(__file__).resolve().parent
        candidates = [script_dir / "models", script_dir.parent / "models"]
        models_dir = next((p for p in candidates if p.exists()), candidates[-1])
    else:
        models_dir = Path(models_dir)
    if not models_dir.exists():
        return []
    return sorted(p for p in models_dir.iterdir() if p.suffix == ".gguf")


# --------------------------- CLI demo ---------------------------------- #


def _demo():
    models = list_models()
    if not models:
        print("No GGUF models found in models/")
        return

    print(f"Found {len(models)} models:\n")
    for path in models:
        fam = detect_family(path)
        print(f"  {path.name}")
        print(f"    Family: {fam}")
        print(f"    Stop strings: {stop_strings_for_family(fam)}")
        print(f"    Recommended params: {recommended_generation_params(fam)}")
        print()


if __name__ == "__main__":
    _demo()
