"""Unified LLM wrapper supporting multiple model families.

This module provides a unified interface for working with different LLM families
(Qwen3, Gemma, Mistral, GPT-OSS, Phi, etc.) with automatic model detection and
family-specific optimizations.

Example:
    >>> from llama_cpp.unified import UnifiedLLM
    >>> llm = UnifiedLLM("models/Qwen3-8B-Q6_K.gguf")
    >>> response = llm.generate("Hello, world!")
    >>> print(response)
"""

from __future__ import annotations

import atexit
import contextlib
import gc
import re
import sys
import threading
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, ClassVar, cast

from llama_cpp import Llama, LlamaConfig, SamplingParams

# ---------------------------------------------------------------------------
# Instance tracking for cleanup at exit
# ---------------------------------------------------------------------------
_unified_instances: set[weakref.ref[Any]] = set()
_cleanup_registered = False
_cleanup_lock = threading.Lock()


def _register_unified_cleanup() -> None:
    """Register cleanup handler only after an instance is created."""
    global _cleanup_registered
    if _cleanup_registered:
        return
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

    AYA = auto()
    GEMMA = auto()
    GLM4 = auto()
    GRANITE = auto()
    MINICPM = auto()
    PHI = auto()
    MISTRAL = auto()
    QWEN3 = auto()
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


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "aya": ModelConfig(
        ModelFamily.AYA, temperature=0.7, top_p=0.9, top_k=40, max_ctx=8192
    ),
    "gemma": ModelConfig(
        ModelFamily.GEMMA,
        chat_format="gemma",
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        min_p=0.0,
        max_ctx=128000,
    ),
    "glm-4": ModelConfig(
        ModelFamily.GLM4,
        chat_format="glm4",
        temperature=0.95,
        top_p=0.7,
        top_k=1,
        max_ctx=131072,
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
        min_p=0.01,
        max_ctx=32768,
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
    "gpt-oss": ModelConfig(
        ModelFamily.GPT_OSS,
        temperature=1.0,
        top_p=1.0,
        top_k=40,
        max_ctx=128000,
        supports_thinking=True,
    ),
}


def detect_model_family(model_path: str) -> ModelConfig:
    """Detect model family from file path.

    Args:
        model_path: Path to the GGUF model file.

    Returns:
        ModelConfig for the detected family.

    Raises:
        ValueError: If model family cannot be detected from path.
    """
    path_lower = model_path.lower()

    if "ministral" in path_lower:
        if "reasoning" in path_lower:
            return MODEL_CONFIGS["ministral-reasoning"]
        return MODEL_CONFIGS["ministral-instruct"]

    # Qwen3-2507 variants (Instruct vs Thinking)
    if "qwen3" in path_lower and "2507" in path_lower and "instruct" in path_lower:
        return MODEL_CONFIGS["qwen3-instruct-2507"]

    for key in sorted(MODEL_CONFIGS.keys(), key=len, reverse=True):
        if key in path_lower:
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
        tokens = self.llm.n_tokens(formatted_text, add_special=self.llm.config.add_bos)
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
    _CONTROL_TOKENS: ClassVar[re.Pattern[str]] = re.compile(
        r"<\|im_end\|>|<\|im_start\|>\w*\n?|<\|im_sep\|>|<end_of_turn>|<start_of_turn>\w*\n?",
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
        formatted, _, n_tokens = self.llm._prepare_chat(messages)
        max_tokens = self._calc_max_tokens_from_count(n_tokens, max_tokens)

        kwargs: dict[str, Any] = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "min_p": self.config.min_p,
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

        temp = self.config.think_temperature or self.config.temperature
        top_p = self.config.think_top_p or self.config.top_p
        top_k = self.config.think_top_k or self.config.top_k
        min_p = self.config.think_min_p or self.config.min_p

        kwargs: dict[str, Any] = {
            "temperature": temp,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
        }
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
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Only add /think or /no_think for Qwen3 models that support thinking mode
        if self.config.family == ModelFamily.QWEN3 and self.config.supports_thinking:
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
        # Handle unclosed thinking tags (truncated output)
        text = re.sub(r"<think(?:ing)?>.*", "", text, flags=re.DOTALL)
        text = re.sub(r"\[THINK\].*", "", text, flags=re.DOTALL | re.IGNORECASE)
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
    STOP: ClassVar[list[str]] = ["<|start|>user", "<|end|><|end|>"]

    _date_lock: ClassVar = threading.Lock()
    _cached_date: ClassVar[str | None] = None
    _cached_date_day: ClassVar[int | None] = None

    def __init__(self, llm: Llama, config: ModelConfig, n_ctx: int) -> None:
        super().__init__(llm, config, n_ctx)
        self.reasoning_level: str = "medium"

    @classmethod
    def _get_current_date(cls) -> str:
        """Get current date with daily caching (thread-safe)."""
        now = datetime.now()
        with cls._date_lock:
            if cls._cached_date_day != now.day or cls._cached_date is None:
                cls._cached_date = now.strftime("%Y-%m-%d")
                cls._cached_date_day = now.day
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
        ModelFamily.AYA: ChatTemplateBackend,
        ModelFamily.GEMMA: ChatTemplateBackend,
        ModelFamily.GLM4: ChatTemplateBackend,
        ModelFamily.GRANITE: ChatTemplateBackend,
        ModelFamily.MINICPM: ChatTemplateBackend,
        ModelFamily.QWEN3: ChatTemplateBackend,
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
                # Find config by family enum
                for cfg in MODEL_CONFIGS.values():
                    if cfg.family == family:
                        self.model_config = cfg
                        break
                else:
                    raise ValueError(f"No config for family: {family}")
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
        )

        self.llm = Llama(model_path, config=llama_config, sampling=sampling)
        try:
            backend_cls = self.BACKEND_MAP[self.model_config.family]
            self.backend: Backend = backend_cls(self.llm, self.model_config, n_ctx)
        except Exception:
            self.llm.close()
            raise

        # Register for cleanup at exit (lazy registration on first instance)
        _register_unified_cleanup()
        self._ref = weakref.ref(self, lambda r: _unified_instances.discard(r))
        _unified_instances.add(self._ref)

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
        if hasattr(self.backend, "_parse_thinking"):
            _, answer = self.backend._parse_thinking(text)
            result: str = answer
            return result
        return text

    def __enter__(self) -> UnifiedLLM:
        """Context manager entry."""
        return self

    def __repr__(self) -> str:
        if self.llm is None:
            return "<UnifiedLLM (closed)>"
        import os

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

    def __del__(self) -> None:
        # Avoid cleanup during interpreter shutdown - atexit handler handles this
        if sys.is_finalizing():
            return
        # Only attempt cleanup if fully initialized
        if hasattr(self, "llm") and self.llm is not None:
            with contextlib.suppress(Exception):
                self.close()

    def n_tokens(self, text: str) -> int:
        """Count tokens for text."""
        return self.llm.n_tokens(text)

    def n_ctx(self) -> int:
        """Get context window size."""
        return self.llm.n_ctx()

    def kv_cache_clear(self) -> None:
        """Clear KV cache."""
        self.llm.kv_cache_clear()

    def close(self) -> None:
        """Release model resources."""
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
