"""Unified LLM wrapper supporting multiple model families.

This module provides a unified interface for working with different LLM families
(Qwen3, Gemma, Gemma 4, Mistral, GPT-OSS, Phi, GLM4, MiniCPM, Granite) with
automatic model detection and family-specific optimizations.

Example:
    >>> from llama_cpp.unified import UnifiedLLM
    >>> llm = UnifiedLLM("models/Qwen3-8B-Q6_K.gguf")
    >>> response = llm.generate("Hello, world!")
    >>> print(response)
"""

from abc import ABC, abstractmethod
import atexit
import contextlib
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from enum import auto
from enum import Enum
import gc
import logging
import os
import re
import threading
from typing import Any, cast, ClassVar
import weakref

from llama_cpp import Llama
from llama_cpp import LlamaConfig
from llama_cpp import LlamaError
from llama_cpp import SamplingParams


# ---------------------------------------------------------------------------
# Instance tracking for cleanup at exit
# ---------------------------------------------------------------------------
_unified_instances: set[weakref.ref[Any]] = set()
_cleanup_registered = False
_cleanup_lock = threading.Lock()


def _register_unified_cleanup() -> None:
    """Register cleanup handler only after an instance is created."""
    global _cleanup_registered
    with _cleanup_lock:
        if _cleanup_registered:
            return
        atexit.register(_cleanup_unified)
        _cleanup_registered = True


def _cleanup_unified() -> None:
    """Close all UnifiedLLM instances before interpreter shutdown."""
    for ref in list(_unified_instances):
        instance = ref()
        if instance is not None:
            with contextlib.suppress(Exception):
                instance.close()
    _unified_instances.clear()
    gc.collect()


class ModelFamily(Enum):
    """Supported model families.

    Each family has specific chat templates, sampling defaults, and capabilities.
    """

    GEMMA = auto()
    GEMMA4 = auto()
    GLM4 = auto()
    GRANITE = auto()
    MINICPM = auto()
    PHI = auto()
    MISTRAL = auto()
    QWEN3 = auto()
    QWEN3_5 = auto()
    GPT_OSS = auto()


@dataclass(slots=True)
class ModelConfig:
    """Model-specific configuration.

    Attributes:
        family: The model family this config belongs to.
        chat_format: llama.cpp chat format name (e.g., "chatml", "gemma").
        temperature: Default sampling temperature.
        top_p: Default nucleus sampling probability.
        top_k: Default top-k sampling value.
        min_p: Default min-p sampling threshold.
        max_ctx: Maximum supported context length.
        supports_thinking: Whether model supports thinking/reasoning mode.
        stop_sequences: Default stop sequences for this model.
        think_temperature: Temperature override for thinking mode.
        think_top_p: Top-p override for thinking mode.
        think_top_k: Top-k override for thinking mode.
        think_min_p: Min-p override for thinking mode.
        presence_penalty: Penalty for token presence (0.0-2.0, reduces repetition).
        repeat_penalty: Repetition penalty multiplier (1.0 = disabled, >1.0 penalizes).
    """

    family: ModelFamily
    chat_format: str | None = None
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    min_p: float = 0.0
    max_ctx: int = 8192
    supports_thinking: bool = False
    stop_sequences: list[str] = field(default_factory=list)
    think_temperature: float | None = None
    think_top_p: float | None = None
    think_top_k: int | None = None
    think_min_p: float | None = None
    presence_penalty: float = 0.0
    repeat_penalty: float = 1.1  # 1.0 = disabled


# Maps config key -> ModelConfig.  Used for auto-detection by filename.
MODEL_CONFIGS: dict[str, ModelConfig] = {
    "gemma": ModelConfig(
        ModelFamily.GEMMA,
        chat_format="gemma",
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        min_p=0.0,
        max_ctx=128000,
    ),
    # Gemma 4 E2B / E4B — dense + PLE, 128K context, thinking via <|think|> in system prompt
    "gemma-4": ModelConfig(
        ModelFamily.GEMMA4,
        chat_format="gemma",
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        min_p=0.0,
        max_ctx=131072,
        supports_thinking=True,
        repeat_penalty=1.0,  # Unsloth: keep disabled unless looping
        stop_sequences=["<turn|>", "<end_of_turn>"],
    ),
    # Gemma 4 26B-A4B (MoE) / 31B (dense) — 256K context
    "gemma-4-large": ModelConfig(
        ModelFamily.GEMMA4,
        chat_format="gemma",
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        min_p=0.0,
        max_ctx=262144,
        supports_thinking=True,
        repeat_penalty=1.0,
        stop_sequences=["<turn|>", "<end_of_turn>"],
    ),
    "glm-4": ModelConfig(
        ModelFamily.GLM4,
        chat_format="glm4",
        temperature=1.0,
        top_p=0.95,
        top_k=0,
        min_p=0.01,
        max_ctx=131072,
    ),
    "glm-4.7": ModelConfig(
        ModelFamily.GLM4,
        temperature=1.0,
        top_p=0.95,
        top_k=0,
        min_p=0.01,
        max_ctx=202752,
        supports_thinking=True,
        repeat_penalty=1.0,
        stop_sequences=["<|endoftext|>", "<|user|>", "<|observation|>"],
    ),
    "granite": ModelConfig(
        ModelFamily.GRANITE, temperature=0.0, top_p=1.0, top_k=1, max_ctx=128000
    ),
    "minicpm": ModelConfig(
        ModelFamily.MINICPM,
        chat_format="chatml",
        temperature=0.9,
        top_p=0.95,
        top_k=50,
        max_ctx=65536,
    ),
    "ministral-reasoning": ModelConfig(
        ModelFamily.MISTRAL,
        temperature=0.7,
        top_p=0.95,
        max_ctx=256000,
    ),
    "ministral-instruct": ModelConfig(
        ModelFamily.MISTRAL,
        temperature=0.15,
        max_ctx=256000,
    ),
    "mistral": ModelConfig(
        ModelFamily.MISTRAL,
        temperature=0.7,
        top_p=0.95,
        max_ctx=32768,
    ),
    "phi-4": ModelConfig(
        ModelFamily.PHI,
        temperature=0.8,
        top_p=0.95,
        top_k=50,
        max_ctx=16000,
        stop_sequences=["<|im_end|>"],
    ),
    "qwen3": ModelConfig(
        ModelFamily.QWEN3,
        chat_format="chatml",
        supports_thinking=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        max_ctx=131072,
        stop_sequences=["<|im_end|>", "<|endoftext|>"],
        think_temperature=0.6,
        think_top_p=0.95,
        think_top_k=20,
        think_min_p=0.0,
    ),
    "qwen3-instruct-2507": ModelConfig(
        ModelFamily.QWEN3,
        chat_format="chatml",
        supports_thinking=False,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        max_ctx=16384,
        presence_penalty=1.0,
        stop_sequences=["<|im_end|>", "<|endoftext|>"],
    ),
    "qwen3-thinking-2507": ModelConfig(
        ModelFamily.QWEN3,
        chat_format="chatml",
        supports_thinking=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        max_ctx=32768,
        presence_penalty=1.0,
        stop_sequences=["<|im_end|>", "<|endoftext|>"],
        think_temperature=0.6,
        think_top_p=0.95,
        think_top_k=20,
        think_min_p=0.0,
    ),
    "qwen3.5": ModelConfig(
        ModelFamily.QWEN3_5,
        chat_format="chatml",
        supports_thinking=True,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        max_ctx=262144,
        presence_penalty=1.5,
        repeat_penalty=1.0,
        stop_sequences=["<|im_end|>", "<|endoftext|>"],
        think_temperature=0.6,
        think_top_p=0.95,
        think_top_k=20,
        think_min_p=0.0,
    ),
    # Qwen3.5 Small (0.8B, 2B, 4B, 9B) — thinking disabled by default
    "qwen3.5-small": ModelConfig(
        ModelFamily.QWEN3_5,
        chat_format="chatml",
        supports_thinking=False,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        max_ctx=262144,
        presence_penalty=1.5,
        repeat_penalty=1.0,
        stop_sequences=["<|im_end|>", "<|endoftext|>"],
    ),
    "gpt-oss": ModelConfig(
        ModelFamily.GPT_OSS,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        max_ctx=131072,
        supports_thinking=True,
        repeat_penalty=1.0,
    ),
}

# Reverse mapping: ModelFamily enum -> default config key.
# When a user passes a ModelFamily enum directly (not a string key),
# this determines which config to use.  The first config key registered
# for each family is the default — additional keys (e.g.
# "qwen3-instruct-2507" for QWEN3) are reachable only by string key.
_FAMILY_DEFAULT_KEY: dict[ModelFamily, str] = {}
for _key, _cfg in MODEL_CONFIGS.items():
    _FAMILY_DEFAULT_KEY.setdefault(_cfg.family, _key)


_GEMMA4_LARGE_MARKERS: tuple[str, ...] = ("26b", "31b", "a4b")


def _is_gemma4_large(name_lower: str) -> bool:
    """Return True for Gemma 4 26B-A4B or 31B (256K context variants)."""
    return any(marker in name_lower for marker in _GEMMA4_LARGE_MARKERS)


def detect_from_metadata(model: "Llama") -> ModelConfig | None:
    """Detect model family using GGUF metadata (authoritative).

    Args:
        model: Loaded Llama instance with model metadata available.

    Returns:
        ModelConfig if detected from metadata, None if metadata unavailable or unrecognized.
    """
    try:
        # Try to get architecture from GGUF metadata (most authoritative)
        arch = model.model.meta_val_str("general.architecture")
        name = model.model.meta_val_str("general.name")
        name_lower = name.lower()

        # Architecture-based detection with name-based variant refinement
        if "qwen" in arch:
            if "qwen3.5" in name_lower or "qwen-3.5" in name_lower:
                # Check if it's a small model (thinking disabled)
                _QWEN35_SMALL_SIZES = {"0.8b", "2b", "4b", "9b"}
                for size in _QWEN35_SMALL_SIZES:
                    if f"-{size}" in name_lower or f" {size}" in name_lower:
                        return MODEL_CONFIGS["qwen3.5-small"]
                return MODEL_CONFIGS["qwen3.5"]
            elif "2507" in name_lower:
                if "thinking" in name_lower:
                    return MODEL_CONFIGS["qwen3-thinking-2507"]
                else:
                    return MODEL_CONFIGS["qwen3-instruct-2507"]
            elif "qwen3" in arch or "qwen3" in name_lower:
                return MODEL_CONFIGS["qwen3"]
            elif "qwen" in arch:
                # Older qwen models (qwen1, qwen2) might have different configs
                # Fall back to qwen3 as reasonable default
                return MODEL_CONFIGS["qwen3"]

        elif "gemma" in arch:
            # Gemma 4 detection — arch stays "gemma*" but name carries the version
            if "gemma-4" in name_lower or "gemma4" in name_lower:
                if _is_gemma4_large(name_lower):
                    return MODEL_CONFIGS["gemma-4-large"]
                return MODEL_CONFIGS["gemma-4"]
            return MODEL_CONFIGS["gemma"]

        elif "ministral" in arch or "ministral" in name_lower:
            if "reasoning" in name_lower:
                return MODEL_CONFIGS["ministral-reasoning"]
            return MODEL_CONFIGS["ministral-instruct"]

        elif "mistral" in arch:
            return MODEL_CONFIGS["mistral"]

        elif "phi" in arch:
            return MODEL_CONFIGS["phi-4"]

        elif "glm" in arch:
            if "glm-4.7" in name_lower or "glm4.7" in name_lower:
                return MODEL_CONFIGS["glm-4.7"]
            return MODEL_CONFIGS["glm-4"]

        elif "llama" in arch:
            # Generic llama architecture - could be llama2, llama3, etc.
            # No specific config needed for now
            return None

    except (RuntimeError, AttributeError):
        # Metadata not available or model not loaded
        pass

    return None


def detect_model_family(model_path: str) -> ModelConfig:
    """Detect model family from file path (filename-based fallback).

    This is used for initial detection before model is loaded. For more
    reliable detection, use detect_from_metadata() after loading.

    Args:
        model_path: Path to the GGUF model file.

    Returns:
        ModelConfig for the detected family.

    Raises:
        ValueError: If model family cannot be detected from path.
    """
    # Match against filename only to avoid false positives from directory
    # names (e.g. /home/user/phi-experiments/llama-model.gguf).
    filename_lower = os.path.basename(model_path).lower()

    if "ministral" in filename_lower:
        if "reasoning" in filename_lower:
            return MODEL_CONFIGS["ministral-reasoning"]
        return MODEL_CONFIGS["ministral-instruct"]

    # Qwen3-2507 variants (Instruct vs Thinking)
    if "qwen3" in filename_lower and "2507" in filename_lower:
        if "thinking" in filename_lower:
            return MODEL_CONFIGS["qwen3-thinking-2507"]
        return MODEL_CONFIGS["qwen3-instruct-2507"]

    # Qwen3.5 Small (0.8B, 2B, 4B, 9B) — thinking disabled by default
    # NOTE: Early return here prevents the generic loop below from matching 'qwen3.5'
    _QWEN35_SMALL_SIZES = {"0.8b", "2b", "4b", "9b"}
    if "qwen3.5" in filename_lower:
        for size in _QWEN35_SMALL_SIZES:
            if f"-{size}" in filename_lower:
                return MODEL_CONFIGS["qwen3.5-small"]
        return MODEL_CONFIGS["qwen3.5"]

    # Gemma 4 variants — routed by size marker (26B-A4B / 31B → large, else E2B/E4B)
    # NOTE: Early return prevents 'gemma' from matching Gemma 4 filenames in the generic loop
    if "gemma-4" in filename_lower or "gemma4" in filename_lower:
        if _is_gemma4_large(filename_lower):
            return MODEL_CONFIGS["gemma-4-large"]
        return MODEL_CONFIGS["gemma-4"]

    # Generic fallback: match longest config key first (prevents 'qwen3' matching 'qwen3.5')
    for key in sorted(MODEL_CONFIGS.keys(), key=len, reverse=True):
        if key in filename_lower:
            return MODEL_CONFIGS[key]

    raise ValueError(
        f"Unknown model family: {model_path}. "
        f"Supported: {', '.join(sorted(MODEL_CONFIGS.keys()))}"
    )


class Backend(ABC):
    """Abstract base class for model inference backends.

    Each backend implements family-specific prompt formatting and generation logic.

    Attributes:
        llm: The underlying Llama instance.
        config: Model-specific configuration.
        n_ctx: Context size for this instance.
    """

    def __init__(self, llm: Llama, config: ModelConfig, n_ctx: int) -> None:
        """Initialize backend.

        Args:
            llm: Llama instance for inference.
            config: Model configuration.
            n_ctx: Context size.
        """
        self.llm = llm
        self.config = config
        self.n_ctx = n_ctx

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None,
        max_tokens: int | None,
        *,
        thinking: bool = False,
        stop: list[str] | None = None,
    ) -> str:
        """Generate text response.

        Args:
            prompt: User prompt text to generate a response for.
            system_prompt: Optional system prompt that provides context and instructions
                for the model's behavior. If None, uses backend-specific default.
            max_tokens: Maximum tokens to generate. If None, automatically calculated
                based on prompt length and context window size.
            thinking: Enable thinking/reasoning mode for models that support it
                (e.g., Qwen3). When enabled, model shows its reasoning process.
            stop: Additional stop sequences to terminate generation. These are
                combined with backend-specific default stop sequences.

        Returns:
            Generated text response with thinking content removed if present.
        """

    @abstractmethod
    def generate_with_thinking(
        self,
        prompt: str,
        system_prompt: str | None,
        max_tokens: int | None,
        *,
        stop: list[str] | None = None,
    ) -> tuple[str, str]:
        """Generate with separate thinking and answer.

        Args:
            prompt: User prompt text to generate a response for.
            system_prompt: Optional system prompt that provides context. If None,
                uses backend-specific default with thinking enabled.
            max_tokens: Maximum tokens to generate. If None, automatically calculated
                based on prompt length and context window size.
            stop: Additional stop sequences to terminate generation. These are
                combined with backend-specific default stop sequences.

        Returns:
            Tuple of (thinking_text, answer_text) where thinking_text contains
            the model's reasoning process and answer_text contains the final response.
        """

    def _calc_max_tokens(self, formatted_text: str, requested: int | None) -> int:
        """Calculate max tokens, clamping to available context.

        Args:
            formatted_text: The fully formatted prompt.
            requested: User-requested max tokens (or None for auto).

        Returns:
            Safe max_tokens value that won't exceed context.

        Raises:
            ValueError: If requested is invalid (0 or negative) or prompt exceeds context.
        """
        if requested is not None and requested <= 0:
            raise ValueError(f"max_tokens must be positive, got {requested}")
        # Count tokens with BOS to match actual generation
        tokens = self.llm.n_tokens(
            formatted_text, add_special=bool(self.llm.config.add_bos)
        )
        return self._calc_max_tokens_from_count(tokens, requested)

    def _calc_max_tokens_from_count(
        self, token_count: int, requested: int | None
    ) -> int:
        """Calculate max tokens from pre-counted token count.

        Args:
            token_count: Number of tokens in prompt (including BOS if applicable).
            requested: User-requested max tokens (or None for auto).

        Returns:
            Safe max_tokens value that won't exceed context.

        Raises:
            ValueError: If requested is invalid or prompt exceeds context.
        """
        if token_count < 0:
            raise ValueError(f"invalid token count: {token_count}")
        if requested is not None and requested <= 0:
            raise ValueError(f"max_tokens must be positive, got {requested}")
        available = self.n_ctx - token_count - 10
        if available <= 0:
            raise ValueError(
                f"Prompt ({token_count} tokens) exceeds context ({self.n_ctx}). "
                "Reduce prompt length or increase n_ctx."
            )
        if requested is not None:
            return min(requested, available)
        return available


class ChatTemplateBackend(Backend):
    """Backend using llama.cpp built-in chat templates.

    Supports most model families including Qwen3, Gemma, GLM4, Mistral, etc.
    """

    _THINK_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"<think(?:ing)?>(.*?)</think(?:ing)?>(.*)", re.DOTALL
    )
    _THINK_BRACKET_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\[THINK\](.*?)\[/THINK\](.*)", re.DOTALL | re.IGNORECASE
    )
    # Gemma 4: <|channel>thought[content]<channel|>  (Unsloth spec)
    _GEMMA4_THINK_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"<\|channel>(.*?)<channel\|>(.*)", re.DOTALL
    )
    _CONTROL_TOKENS: ClassVar[re.Pattern[str]] = re.compile(
        r"<\|im_end\|>|<\|im_start\|>\w*\n?|<\|im_sep\|>|<end_of_turn>|<start_of_turn>\w*\n?"
        r"|<turn\|>|<\|think\|>",
        re.DOTALL,
    )
    # Thinking tag constants
    _THINKING_TAG_VARIANTS: ClassVar[tuple[str, ...]] = ("<thinking>", "<think>")

    def generate(
        self,
        prompt: str,
        system_prompt: str | None,
        max_tokens: int | None,
        *,
        thinking: bool = False,
        stop: list[str] | None = None,
    ) -> str:
        messages = self._build_messages(prompt, system_prompt, thinking=thinking)
        # Use _prepare_chat to format and tokenize once
        _, _, n_tokens = self.llm._prepare_chat(messages)
        max_tokens = self._calc_max_tokens_from_count(n_tokens, max_tokens)

        kwargs: dict[str, Any] = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "min_p": self.config.min_p,
            "repeat_penalty": self.config.repeat_penalty,
        }
        if self.config.presence_penalty > 0:
            kwargs["presence_penalty"] = self.config.presence_penalty
        all_stop = (
            list(self.config.stop_sequences) if self.config.stop_sequences else []
        )
        if stop:
            all_stop.extend(stop)
        if all_stop:
            kwargs["stop"] = all_stop

        resp = cast(
            dict[str, Any],
            self.llm.create_chat_completion(messages, max_tokens=max_tokens, **kwargs),
        )
        text = resp["choices"][0]["message"]["content"] or ""
        return self._clean_response(text)

    def generate_with_thinking(
        self,
        prompt: str,
        system_prompt: str | None,
        max_tokens: int | None,
        *,
        stop: list[str] | None = None,
    ) -> tuple[str, str]:
        if not self.config.supports_thinking:
            return "", self.generate(prompt, system_prompt, max_tokens, stop=stop)

        messages = self._build_messages(prompt, system_prompt, thinking=True)
        # Use _prepare_chat to format and tokenize once
        formatted, _, n_tokens = self.llm._prepare_chat(messages)
        max_tokens = self._calc_max_tokens_from_count(n_tokens, max_tokens)

        temp = (
            self.config.think_temperature
            if self.config.think_temperature is not None
            else self.config.temperature
        )
        top_p = (
            self.config.think_top_p
            if self.config.think_top_p is not None
            else self.config.top_p
        )
        top_k = (
            self.config.think_top_k
            if self.config.think_top_k is not None
            else self.config.top_k
        )
        min_p = (
            self.config.think_min_p
            if self.config.think_min_p is not None
            else self.config.min_p
        )

        kwargs: dict[str, Any] = {
            "temperature": temp,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "repeat_penalty": self.config.repeat_penalty,
        }
        if self.config.presence_penalty > 0:
            kwargs["presence_penalty"] = self.config.presence_penalty
        all_stop = (
            list(self.config.stop_sequences) if self.config.stop_sequences else []
        )
        if stop:
            all_stop.extend(stop)
        if all_stop:
            kwargs["stop"] = all_stop

        resp = cast(
            dict[str, Any],
            self.llm.create_chat_completion(messages, max_tokens=max_tokens, **kwargs),
        )
        text = resp["choices"][0]["message"]["content"] or ""
        return self._parse_thinking(text)

    def _build_messages(
        self, prompt: str, system_prompt: str | None, thinking: bool
    ) -> list[dict[str, str]]:
        """Build chat messages list.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.
            thinking: Whether to enable thinking mode.

        Returns:
            List of message dicts with role and content.
        """
        messages: list[dict[str, str]] = []

        # Gemma 4 activates thinking by prepending <|think|> to the system prompt
        # (Unsloth spec: https://unsloth.ai/docs/models/gemma-4)
        if (
            self.config.family == ModelFamily.GEMMA4
            and self.config.supports_thinking
            and thinking
        ):
            system_prompt = f"<|think|>{system_prompt}" if system_prompt else "<|think|>"

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Only add /think or /no_think for Qwen3 family models that support thinking
        if (
            self.config.family in (ModelFamily.QWEN3, ModelFamily.QWEN3_5)
            and self.config.supports_thinking
        ):
            suffix = " /think" if thinking else " /no_think"
            prompt = prompt + suffix

        messages.append({"role": "user", "content": prompt})
        return messages

    def _clean_response(self, text: str) -> str:
        """Strip thinking tags and control tokens from response."""
        # Handle complete thinking blocks
        text = re.sub(
            r"<think(?:ing)?>.*?</think(?:ing)?>\s*", "", text, flags=re.DOTALL
        )
        text = re.sub(
            r"\[THINK\].*?\[/THINK\]\s*", "", text, flags=re.DOTALL | re.IGNORECASE
        )
        # Gemma 4 thinking block
        text = re.sub(r"<\|channel>.*?<channel\|>\s*", "", text, flags=re.DOTALL)
        # Handle unclosed thinking tags (truncated output)
        text = re.sub(r"<think(?:ing)?>.*", "", text, flags=re.DOTALL)
        text = re.sub(r"\[THINK\].*", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<\|channel>.*", "", text, flags=re.DOTALL)
        text = re.sub(r"^/(?:no_)?think\n?", "", text)
        if "/response" in text:
            text = text.split("/response", 1)[1]
        if "<start_of_turn>" in text or "<end_of_turn>" in text:
            parts = re.split(r"<(?:start|end)_of_turn>(?:user|model)?\n?", text)
            for part in parts:
                part = part.strip()
                if part and not part.startswith("<"):
                    text = part
                    break
        text = self._CONTROL_TOKENS.sub("", text)
        return text.strip()

    def _parse_thinking(self, text: str) -> tuple[str, str]:
        """Parse thinking and answer from response.

        Args:
            text: Raw response text.

        Returns:
            Tuple of (thinking_text, answer_text).
        """
        text = self._CONTROL_TOKENS.sub("", text)
        match = self._THINK_PATTERN.search(text)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        match = self._THINK_BRACKET_PATTERN.search(text)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        match = self._GEMMA4_THINK_PATTERN.search(text)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        for tag in self._THINKING_TAG_VARIANTS:
            if tag in text:
                return text.split(tag, 1)[1].strip(), ""
        if "[THINK]" in text.upper():
            return re.split(r"\[THINK\]", text, flags=re.IGNORECASE)[1].strip(), ""
        return "", text.strip()


class PhiBackend(Backend):
    """Backend for Phi-4 with custom <|im_sep|> template."""

    def generate(
        self,
        prompt: str,
        system_prompt: str | None,
        max_tokens: int | None,
        *,
        thinking: bool = False,
        stop: list[str] | None = None,
    ) -> str:
        formatted = self._format(prompt, system_prompt)
        max_tokens = self._calc_max_tokens(formatted, max_tokens)
        all_stop = ["<|im_end|>"]
        if stop:
            all_stop.extend(stop)
        return cast(
            str, self.llm.generate(formatted, max_tokens=max_tokens, stop=all_stop)
        ).strip()

    def generate_with_thinking(
        self,
        prompt: str,
        system_prompt: str | None,
        max_tokens: int | None,
        *,
        stop: list[str] | None = None,
    ) -> tuple[str, str]:
        return "", self.generate(prompt, system_prompt, max_tokens, stop=stop)

    def _format(self, prompt: str, system_prompt: str | None) -> str:
        """Format prompt using Phi-4 template."""
        parts: list[str] = []
        if system_prompt:
            parts.append(f"<|im_start|>system<|im_sep|>\n{system_prompt}<|im_end|>\n")
        parts.append(f"<|im_start|>user<|im_sep|>\n{prompt}<|im_end|>\n")
        parts.append("<|im_start|>assistant<|im_sep|>\n")
        return "".join(parts)


class GPTOSSBackend(Backend):
    """Backend for GPT-OSS with dual-channel (analysis/final) output.

    Attributes:
        reasoning_level: Current reasoning level ("low", "medium", "high").
    """

    _ANALYSIS_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"<\|channel\|\>\s*analysis\s*<\|message\|\>(.*?)(?:<\|end\|\>|<\|start\|\>|$)",
        re.DOTALL,
    )
    _FINAL_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"<\|channel\|\>\s*final\s*<\|message\|\>(.*?)(?:<\|end\|\>|$)", re.DOTALL
    )
    SYSTEM: ClassVar[str] = "You are ChatGPT, a large language model trained by OpenAI."
    STOP: ClassVar[list[str]] = ["<|start|>user", "<|end|><|end|>", "<|return|>"]

    _date_lock: ClassVar = threading.Lock()
    _cached_date: ClassVar[str | None] = None
    _cached_date_key: ClassVar[tuple[int, int, int] | None] = None

    def __init__(self, llm: Llama, config: ModelConfig, n_ctx: int) -> None:
        super().__init__(llm, config, n_ctx)
        self.reasoning_level: str = "medium"

    @classmethod
    def _get_current_date(cls) -> str:
        """Get current date with daily caching (thread-safe)."""
        now = datetime.now()
        today = (now.year, now.month, now.day)
        with cls._date_lock:
            if cls._cached_date_key != today or cls._cached_date is None:
                cls._cached_date = now.strftime("%Y-%m-%d")
                cls._cached_date_key = today
            return cls._cached_date

    def generate(
        self,
        prompt: str,
        system_prompt: str | None,
        max_tokens: int | None,
        *,
        thinking: bool = False,
        stop: list[str] | None = None,
    ) -> str:
        _, final = self.generate_with_thinking(
            prompt, system_prompt, max_tokens, stop=stop
        )
        return final

    def generate_with_thinking(
        self,
        prompt: str,
        system_prompt: str | None,
        max_tokens: int | None,
        *,
        stop: list[str] | None = None,
    ) -> tuple[str, str]:
        formatted = self._format(
            prompt, system_prompt or self.SYSTEM, self.reasoning_level
        )
        max_tokens = self._calc_max_tokens(formatted, max_tokens)

        all_stop = list(self.STOP)
        if stop:
            all_stop.extend(stop)

        resp = cast(
            str, self.llm.generate(formatted, max_tokens=max_tokens, stop=all_stop)
        )

        analysis = self._ANALYSIS_PATTERN.search(resp)
        final = self._FINAL_PATTERN.search(resp)

        analysis_text = analysis.group(1).strip() if analysis else ""
        final_text = final.group(1).strip() if final else analysis_text

        return analysis_text, final_text

    def _format(self, prompt: str, system: str, reasoning: str) -> str:
        """Format prompt using GPT-OSS template."""
        today = self._get_current_date()
        return (
            f"<|start|>system<|message|>{system}\n"
            f"Knowledge cutoff: 2024-06\nCurrent date: {today}\n"
            f"Reasoning: {reasoning}\n<|end|>\n\n"
            f"<|start|>user<|message|>{prompt}<|end|><|start|>assistant"
        )


class UnifiedLLM:
    """Unified interface for multiple LLM families.

    Automatically detects model family from path and applies appropriate
    chat templates, sampling parameters, and generation strategies.

    Thread safety: Methods are NOT thread-safe. Do not call generate(),
    generate_with_thinking(), or other methods concurrently from multiple
    threads on the same instance. For parallelism, use multiple UnifiedLLM
    instances or LlamaPool.

    Attributes:
        llm: Underlying Llama instance.
        backend: Family-specific backend for generation.
        model_config: Configuration for detected model family.

    Example:
        >>> llm = UnifiedLLM("models/Qwen3-8B-Q6_K.gguf")
        >>> print(llm.generate("Hello"))

        >>> # With thinking mode
        >>> print(llm.generate("Solve x^2 = 4", thinking=True))

        >>> # As context manager
        >>> with UnifiedLLM("models/model.gguf") as llm:
        ...     print(llm.generate("Hi"))
    """

    BACKEND_MAP: ClassVar[dict[ModelFamily, type[Backend]]] = {
        ModelFamily.GEMMA: ChatTemplateBackend,
        ModelFamily.GEMMA4: ChatTemplateBackend,
        ModelFamily.GLM4: ChatTemplateBackend,
        ModelFamily.GRANITE: ChatTemplateBackend,
        ModelFamily.MINICPM: ChatTemplateBackend,
        ModelFamily.QWEN3: ChatTemplateBackend,
        ModelFamily.QWEN3_5: ChatTemplateBackend,
        ModelFamily.PHI: PhiBackend,
        ModelFamily.MISTRAL: ChatTemplateBackend,
        ModelFamily.GPT_OSS: GPTOSSBackend,
    }

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 8192,
        n_batch: int = 2048,
        n_ubatch: int = 512,
        n_gpu_layers: int = -1,
        verbose: bool = False,
        family: str | ModelFamily | None = None,
    ) -> None:
        """Initialize UnifiedLLM.

        Args:
            model_path: Path to GGUF model file.
            n_ctx: Context size (clamped to model's max).
            n_batch: Batch size for prompt processing.
            n_ubatch: Micro-batch size.
            n_gpu_layers: Layers to offload to GPU (-1 = all).
            verbose: Enable verbose logging.
            family: Explicit model family override (str key or ModelFamily enum).
                   If None, auto-detects from model path.

        Raises:
            ValueError: If model family cannot be detected.
        """
        # Resolve model config from explicit family or auto-detect
        if family is not None:
            if isinstance(family, ModelFamily):
                # O(1) lookup via reverse mapping
                default_key = _FAMILY_DEFAULT_KEY.get(family)
                if default_key is None:
                    raise ValueError(f"No config for family: {family}")
                self.model_config = MODEL_CONFIGS[default_key]
            elif isinstance(family, str):
                if family not in MODEL_CONFIGS:
                    raise ValueError(
                        f"Unknown family: {family}. "
                        f"Supported: {', '.join(sorted(MODEL_CONFIGS.keys()))}"
                    )
                self.model_config = MODEL_CONFIGS[family]
            else:
                raise TypeError("family must be str or ModelFamily")
        else:
            self.model_config = detect_model_family(model_path)

        n_ctx = min(n_ctx, self.model_config.max_ctx)
        n_batch = min(n_batch, n_ctx)
        n_ubatch = min(n_ubatch, n_batch)

        llama_config = LlamaConfig(
            model_path=model_path,
            chat_format=self.model_config.chat_format,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_ubatch=n_ubatch,
            n_gpu_layers=n_gpu_layers,
            offload_kqv=True,
            flash_attn=1,
            verbose=verbose,
        )

        sampling = SamplingParams(
            temperature=self.model_config.temperature,
            top_p=self.model_config.top_p,
            top_k=self.model_config.top_k,
            min_p=self.model_config.min_p,
            repeat_penalty=self.model_config.repeat_penalty,
        )

        self.llm = Llama(model_path, config=llama_config, sampling=sampling)

        # Refine model config detection using metadata (more reliable than filename)
        # Only refine if family was auto-detected (not user-specified)
        if family is None:
            metadata_config = detect_from_metadata(self.llm)
            if metadata_config is not None:
                logging.debug(
                    "Refined model detection: %s (metadata) vs %s (filename)",
                    metadata_config.family,
                    self.model_config.family,
                )
                self.model_config = metadata_config

        # Warn if n_ctx exceeds model's training context
        model_train_ctx = self.llm.model.n_ctx_train()
        if n_ctx > model_train_ctx:
            logging.warning(
                "Requested n_ctx=%d exceeds model training context %d. "
                "Generation quality may degrade beyond %d tokens.",
                n_ctx,
                model_train_ctx,
                model_train_ctx,
            )

        try:
            backend_cls = self.BACKEND_MAP[self.model_config.family]
            self.backend: Backend = backend_cls(self.llm, self.model_config, n_ctx)
        except Exception:
            self.llm.close()
            raise

        self._closed = False

        # Register for cleanup at exit (lazy registration on first instance)
        _register_unified_cleanup()
        self._ref = weakref.ref(self, lambda r: _unified_instances.discard(r))
        _unified_instances.add(self._ref)

    def _check_closed(self) -> None:
        """Raise LlamaError if instance has been closed."""
        if self._closed:
            raise LlamaError("UnifiedLLM instance has been closed")

    @property
    def family(self) -> ModelFamily:
        """Get the detected model family."""
        return self.model_config.family

    @property
    def supports_thinking(self) -> bool:
        """Check if model supports thinking/reasoning mode."""
        return self.model_config.supports_thinking

    def set_reasoning_level(self, level: str) -> None:
        """Set reasoning level for GPT-OSS models.

        Args:
            level: One of "low", "medium", "high".

        Raises:
            ValueError: If level is invalid.
        """
        if level not in ("low", "medium", "high"):
            raise ValueError(f"Invalid reasoning level: {level}")
        if isinstance(self.backend, GPTOSSBackend):
            self.backend.reasoning_level = level

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        *,
        thinking: bool = False,
        stop: list[str] | None = None,
    ) -> str:
        """Generate text response.

        Args:
            prompt: User prompt text.
            system_prompt: Optional system prompt.
            max_tokens: Maximum tokens to generate (auto if None).
            thinking: Enable thinking mode (Qwen3, GPT-OSS).
            stop: Additional stop sequences.

        Returns:
            Generated text response.
        """
        self._check_closed()
        return self.backend.generate(
            prompt, system_prompt, max_tokens, thinking=thinking, stop=stop
        )

    def generate_with_thinking(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        *,
        stop: list[str] | None = None,
    ) -> tuple[str, str]:
        """Generate with separate thinking and answer.

        Args:
            prompt: User prompt text.
            system_prompt: Optional system prompt.
            max_tokens: Maximum tokens to generate.
            stop: Additional stop sequences.

        Returns:
            Tuple of (thinking_text, answer_text).
        """
        self._check_closed()
        return self.backend.generate_with_thinking(
            prompt, system_prompt, max_tokens, stop=stop
        )

    def strip_thinking(self, text: str) -> str:
        """Remove thinking tags from text, return only the answer.

        Args:
            text: Text potentially containing thinking tags.

        Returns:
            Text with thinking content removed.
        """
        if isinstance(self.backend, ChatTemplateBackend):
            _, answer = self.backend._parse_thinking(text)
            return answer
        return text

    def __enter__(self) -> UnifiedLLM:
        """Context manager entry."""
        return self

    def __repr__(self) -> str:
        if getattr(self, "_closed", False):
            return "<UnifiedLLM (closed)>"
        if getattr(self, "llm", None) is None:
            return "<UnifiedLLM (uninitialized)>"
        model_name = os.path.basename(self.llm.config.model_path)
        return (
            f"<UnifiedLLM model={model_name!r} "
            f"family={self.family.name} n_ctx={self.model_config.max_ctx}>"
        )

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: object,
    ) -> None:
        """Context manager exit."""
        self.close()

    def n_tokens(self, text: str) -> int:
        """Count tokens for text."""
        self._check_closed()
        return self.llm.n_tokens(text)

    def n_ctx(self) -> int:
        """Get context window size."""
        self._check_closed()
        return self.llm.n_ctx()

    def kv_cache_clear(self) -> None:
        """Clear KV cache."""
        self._check_closed()
        self.llm.kv_cache_clear()

    def close(self) -> None:
        """Release model resources. Safe to call multiple times (idempotent)."""
        if getattr(self, "_closed", True):
            return
        self._closed = True
        # Remove from instance tracking
        if hasattr(self, "_ref"):
            _unified_instances.discard(self._ref)
        if hasattr(self, "llm") and self.llm is not None:
            self.llm.close()
            self.llm = None  # type: ignore[assignment]
        if hasattr(self, "backend"):
            self.backend = None  # type: ignore[assignment]
        # Force GC to collect any reference cycles while interpreter is safe
        gc.collect()
