"""Unified LLM wrapper for a curated set of model families.

Supported families (other architectures are rejected at construction time):

- **Qwen 3.5** (hybrid attention; thinking on by default for 27B / 35B-A3B
  / 122B-A10B / 397B-A17B; thinking off for 0.8B / 2B / 4B / 9B small variants)
- **Qwen 3.6** (27B dense, 35B-A3B MoE; both thinking and instruct modes)
- **Gemma 4** (E2B / E4B 128K; 26B-A4B / 31B 256K; thinking via ``<|think|>``
  prefix in the system prompt per Unsloth spec)
- **IBM Granite 4.1** (3B / 8B / 30B dense; deterministic defaults
  T=0.0, top_p=1.0, top_k=0; ctx 16K-131K)

All sampling defaults follow the recipes published at https://unsloth.ai/docs.

Example:
    >>> from llama_cpp.unified import UnifiedLLM
    >>> llm = UnifiedLLM("models/Qwen3.5-4B-Q4_K_M.gguf")
    >>> response = llm.generate("Hello, world!")
    >>> print(response)
"""

from abc import ABC, abstractmethod
import atexit
from collections.abc import Iterator
import contextlib
from dataclasses import dataclass
from dataclasses import field
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
# Serializes every _unified_instances mutation/snapshot so the set's internal
# structure is never modified concurrently (a data race under PEP 703
# free-threaded builds; harmless under the GIL). It is a RLock — NOT the plain
# _cleanup_lock — because the weakref finalizer can fire at an arbitrary GC
# point on a thread that already holds this lock; a non-reentrant lock would
# self-deadlock there. Mirrors llama.py's _instances_lock. Critical sections are
# a single set operation and never span close()/__init__ work, so there is no
# lock inversion. (close() runs OUTSIDE this lock for the same reason.)
_instances_lock = threading.RLock()


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
    # Snapshot under the lock so iteration can't race a concurrent mutation of
    # the set's internal structure; close() runs OUTSIDE the lock (it
    # re-acquires _instances_lock for its own discard).
    with _instances_lock:
        snapshot = list(_unified_instances)
    for ref in snapshot:
        instance = ref()
        if instance is not None:
            with contextlib.suppress(Exception):
                instance.close()
    with _instances_lock:
        _unified_instances.clear()
    gc.collect()


class ModelFamily(Enum):
    """Supported model families.

    Each family has specific chat templates, sampling defaults, and capabilities.
    Models outside this set are rejected at construction time.
    """

    QWEN3_5 = auto()
    QWEN3_6 = auto()
    GEMMA4 = auto()
    GRANITE = auto()


class UnsupportedModelError(ValueError):
    """Raised when a model file does not match any supported family."""


@dataclass(slots=True)
class ModelConfig:
    """Model-specific configuration.

    Per the supported families' Unsloth recipes, thinking mode reuses the
    base sampling profile (Qwen 3.5/3.6) or has no separate spec (Gemma 4,
    Granite 4.1). There is therefore no per-mode sampling override —
    callers needing different sampling for thinking vs non-thinking should
    construct two ``UnifiedLLM`` instances with different ``family=``
    presets (e.g. ``qwen3.6`` vs ``qwen3.6-coding``).

    Attributes:
        family: The model family this config belongs to.
        chat_format: llama.cpp chat format name (e.g., "chatml", "gemma"),
            or None to use the model's embedded chat template.
        temperature: Default sampling temperature.
        top_p: Default nucleus sampling probability.
        top_k: Default top-k sampling value.
        min_p: Default min-p sampling threshold.
        max_ctx: Maximum supported context length.
        supports_thinking: Whether model supports thinking/reasoning mode.
        stop_sequences: Default stop sequences for this model.
        presence_penalty: Token-presence penalty (Unsloth uses 1.5 for Qwen
            thinking; 0.0 disables).
        repeat_penalty: Repetition penalty multiplier (1.0 = disabled).
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
    presence_penalty: float = 0.0
    repeat_penalty: float = 1.0


# All sampling defaults below follow https://unsloth.ai/docs/models/...
# Stop sequences include both the canonical chatml/gemma turn ends and the
# common GGUF eos variants — extra entries are harmless if the model never
# emits them.

_QWEN_CHATML_STOPS = ["<|im_end|>", "<|endoftext|>"]
# Gemma 4 emits "<end_of_turn>" per the upstream tokenizer (chatml-derived
# template). "<turn|>" is kept as a defensive belt-and-suspenders entry: it
# was observed in an early GGUF re-encoding and harmlessly never matches if
# the model never emits it. Drop only after confirming via raw verbose=True
# output across all 4 supported Gemma-4 variants (E2B/E4B/26B-A4B/31B).
_GEMMA4_STOPS = ["<turn|>", "<end_of_turn>"]
_GRANITE_STOPS = ["<|end_of_text|>", "<|endoftext|>"]

# Headroom subtracted from n_ctx when computing the auto max_tokens budget.
# Reserves space for any post-prompt control tokens the chat template may
# emit (assistant turn-start markers, BOS variants) and avoids a hard
# context-overflow on the very last token.
_CTX_HEADROOM_TOKENS = 10


# Maps config key -> ModelConfig.  Used for auto-detection by filename.
MODEL_CONFIGS: dict[str, ModelConfig] = {
    # ------------------------------------------------------------------
    # Qwen 3.5 — hybrid-attention (262K ctx, 1M via YaRN).
    #
    # Large variants (27B / 35B-A3B / 122B-A10B / 397B-A17B) default to
    # thinking on; small variants (0.8B / 2B / 4B / 9B) default to
    # thinking off but use the same sampling profile.
    # ------------------------------------------------------------------
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
        stop_sequences=_QWEN_CHATML_STOPS,
    ),
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
        stop_sequences=_QWEN_CHATML_STOPS,
    ),
    # Opt-in coding preset (Unsloth: precise coding / WebDev / Arena).
    "qwen3.5-coding": ModelConfig(
        ModelFamily.QWEN3_5,
        chat_format="chatml",
        supports_thinking=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        max_ctx=262144,
        presence_penalty=0.0,
        repeat_penalty=1.0,
        stop_sequences=_QWEN_CHATML_STOPS,
    ),
    # ------------------------------------------------------------------
    # Qwen 3.6 — 27B dense + 35B-A3B MoE.  Thinking and instruct modes
    # share max_ctx but diverge on sampling.  MTP variants accept the
    # same presets — MTP affects the runtime, not the sampling recipe.
    # ------------------------------------------------------------------
    "qwen3.6": ModelConfig(
        ModelFamily.QWEN3_6,
        chat_format="chatml",
        supports_thinking=True,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        max_ctx=262144,
        presence_penalty=1.5,
        repeat_penalty=1.0,
        stop_sequences=_QWEN_CHATML_STOPS,
    ),
    "qwen3.6-coding": ModelConfig(
        ModelFamily.QWEN3_6,
        chat_format="chatml",
        supports_thinking=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        max_ctx=262144,
        presence_penalty=0.0,
        repeat_penalty=1.0,
        stop_sequences=_QWEN_CHATML_STOPS,
    ),
    "qwen3.6-instruct": ModelConfig(
        ModelFamily.QWEN3_6,
        chat_format="chatml",
        supports_thinking=False,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        max_ctx=262144,
        presence_penalty=1.5,
        repeat_penalty=1.0,
        stop_sequences=_QWEN_CHATML_STOPS,
    ),
    "qwen3.6-instruct-reasoning": ModelConfig(
        ModelFamily.QWEN3_6,
        chat_format="chatml",
        supports_thinking=False,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        max_ctx=262144,
        presence_penalty=1.5,
        repeat_penalty=1.0,
        stop_sequences=_QWEN_CHATML_STOPS,
    ),
    # ------------------------------------------------------------------
    # Gemma 4 — E2B/E4B 128K, 26B-A4B/31B 256K.  Thinking is opt-in via
    # `<|think|>` prefix in the system prompt (handled in
    # ChatTemplateBackend._build_messages).  Repetition penalty disabled
    # per Unsloth.
    # ------------------------------------------------------------------
    "gemma-4": ModelConfig(
        ModelFamily.GEMMA4,
        chat_format="gemma",
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        min_p=0.0,
        max_ctx=131072,
        supports_thinking=True,
        repeat_penalty=1.0,
        stop_sequences=_GEMMA4_STOPS,
    ),
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
        stop_sequences=_GEMMA4_STOPS,
    ),
    # ------------------------------------------------------------------
    # IBM Granite 4.1 — deterministic defaults per Unsloth (T=0.0, top_p=1.0,
    # top_k=0).  3B / 8B / 30B dense; 16K min recommended ctx, 131K max.
    # No thinking-mode spec from upstream, so we expose it as off.
    # ------------------------------------------------------------------
    "granite": ModelConfig(
        ModelFamily.GRANITE,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        max_ctx=131072,
        supports_thinking=False,
        repeat_penalty=1.0,
        stop_sequences=_GRANITE_STOPS,
    ),
}

# Reverse mapping: ModelFamily enum -> default config key.  When a user
# passes a ModelFamily enum directly, this picks the canonical preset.  The
# first config key registered for each family is the default; additional
# keys (e.g. "qwen3.5-coding", "qwen3.6-instruct") require an explicit
# string.  Order in MODEL_CONFIGS therefore matters.
_FAMILY_DEFAULT_KEY: dict[ModelFamily, str] = {}
for _key, _cfg in MODEL_CONFIGS.items():
    _FAMILY_DEFAULT_KEY.setdefault(_cfg.family, _key)


# Filename / metadata markers used to pick the right preset within a family.
_GEMMA4_LARGE_MARKERS: tuple[str, ...] = ("26b", "31b", "a4b")
_QWEN_SMALL_SIZES: tuple[str, ...] = ("0.8b", "2b", "4b", "9b")


def _is_gemma4_large(name_lower: str) -> bool:
    """Return True for Gemma 4 26B-A4B or 31B (256K context variants)."""
    return any(marker in name_lower for marker in _GEMMA4_LARGE_MARKERS)


def _classify_qwen35_variant(name_lower: str) -> ModelConfig:
    """Qwen 3.5: size-based routing for thinking-default-on vs -off."""
    for size in _QWEN_SMALL_SIZES:
        if f"-{size}" in name_lower or f" {size}" in name_lower:
            return MODEL_CONFIGS["qwen3.5-small"]
    return MODEL_CONFIGS["qwen3.5"]


def _classify_qwen36_variant(name_lower: str) -> ModelConfig:
    """Qwen 3.6: thinking-mode default unless 'instruct' is in the name.

    Coding preset is not auto-detected; opt in via ``family="qwen3.6-coding"``.
    """
    if "instruct" in name_lower:
        if "reasoning" in name_lower:
            return MODEL_CONFIGS["qwen3.6-instruct-reasoning"]
        return MODEL_CONFIGS["qwen3.6-instruct"]
    return MODEL_CONFIGS["qwen3.6"]


def detect_from_metadata(model: Llama) -> ModelConfig | None:
    """Detect a supported family from GGUF metadata (authoritative).

    Returns ``None`` if metadata is unavailable or the architecture is not
    in the supported set — caller is expected to fall back to filename
    detection or raise.
    """
    try:
        arch = model.model.meta_val_str("general.architecture").lower()
        name = model.model.meta_val_str("general.name").lower()
    except (RuntimeError, AttributeError):
        return None

    # Qwen family — distinguish 3.5 vs 3.6 by name; arch stays "qwen*".
    if "qwen" in arch:
        if "qwen3.6" in name or "qwen-3.6" in name or "qwen3_6" in name:
            return _classify_qwen36_variant(name)
        if "qwen3.5" in name or "qwen-3.5" in name or "qwen3_5" in name:
            return _classify_qwen35_variant(name)
        return None

    # Gemma 4 — only Gemma 4 is supported, not Gemma 2/3.
    if "gemma" in arch:
        if "gemma-4" in name or "gemma4" in name:
            if _is_gemma4_large(name):
                return MODEL_CONFIGS["gemma-4-large"]
            return MODEL_CONFIGS["gemma-4"]
        return None

    # Granite 4.x — also covers granitehybrid / granitemoe arch tags from
    # llama.cpp upstream; sampling preset is the same deterministic recipe
    # regardless of layout variant.
    if "granite" in arch or "granite" in name:
        if "granite-4" in name or "granite4" in name or "granite-4.1" in name:
            return MODEL_CONFIGS["granite"]
        # An older Granite (3.x) — not in the supported set.
        return None

    return None


def detect_model_family(model_path: str) -> ModelConfig:
    """Detect a supported family from a filename.

    Used before the model loads (initial config). Refined post-load via
    ``detect_from_metadata`` for cases where the filename lies (e.g. mirror
    repos that rename files).

    Raises:
        UnsupportedModelError: if the filename does not match any of
            Qwen 3.5, Qwen 3.6, Gemma 4, or Granite 4.1.
    """
    # Match against the filename only — avoid false positives from
    # directory names (e.g. ``/home/user/granite-experiments/foo.gguf``).
    filename = os.path.basename(model_path).lower()

    if "qwen3.6" in filename or "qwen-3.6" in filename:
        return _classify_qwen36_variant(filename)
    if "qwen3.5" in filename or "qwen-3.5" in filename:
        return _classify_qwen35_variant(filename)
    if "gemma-4" in filename or "gemma4" in filename:
        if _is_gemma4_large(filename):
            return MODEL_CONFIGS["gemma-4-large"]
        return MODEL_CONFIGS["gemma-4"]
    if "granite-4" in filename or "granite4" in filename:
        return MODEL_CONFIGS["granite"]

    raise UnsupportedModelError(
        f"Model {os.path.basename(model_path)!r} does not match any supported "
        f"family. UnifiedLLM supports Qwen 3.5, Qwen 3.6, Gemma 4, and "
        f"IBM Granite 4.1. Use the lower-level Llama class for other models."
    )


class Backend(ABC):
    """Abstract base class for model inference backends.

    Each backend implements family-specific prompt formatting and generation logic.

    Attributes:
        llm: The underlying Llama instance.
        config: Model-specific configuration.
        n_ctx: Context size for this instance.
    """

    def __init__(
        self,
        llm: Llama,
        config: ModelConfig,
        n_ctx: int,
        *,
        speculative: bool = False,
        n_draft_max: int | None = None,
    ) -> None:
        """Initialize backend.

        Args:
            llm: Llama instance for inference.
            config: Model configuration.
            n_ctx: Context size.
            speculative: When True, all generations forward
                ``speculative=True`` to ``Llama.create_chat_completion`` to
                use the draft-MTP path. UnifiedLLM resolves this at
                construction from ``Context.supports_speculative_mtp()``.
            n_draft_max: Optional override for the number of draft tokens
                proposed per verify step. None defers to the SamplingParams
                default (2).
        """
        self.llm = llm
        self.config = config
        self.n_ctx = n_ctx
        self.speculative = speculative
        self.n_draft_max = n_draft_max

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None,
        max_tokens: int | None,
        *,
        thinking: bool = False,
        stop: list[str] | None = None,
        reset_kv_cache: bool = True,
        cache_prompt: bool = True,
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
            reset_kv_cache: Forwarded to ``Llama.create_chat_completion``.
                Default True clears KV before this turn.
            cache_prompt: Forwarded to ``Llama.create_chat_completion``.
                When ``reset_kv_cache=False`` and this is True, trims KV to
                the longest matching prefix and decodes only the divergent
                suffix.

        Returns:
            Generated text response with thinking content removed if present.
        """

    def strip_thinking(self, text: str) -> str:
        """Remove thinking tags from generated text.

        Default implementation returns the text unchanged. Backends that
        produce interleaved thinking output (e.g. ChatTemplateBackend) should
        override this to strip their reasoning markup.
        """
        return text

    @abstractmethod
    def generate_with_thinking(
        self,
        prompt: str,
        system_prompt: str | None,
        max_tokens: int | None,
        *,
        stop: list[str] | None = None,
        reset_kv_cache: bool = True,
        cache_prompt: bool = True,
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
            reset_kv_cache: Forwarded to ``Llama.create_chat_completion``.
            cache_prompt: Forwarded to ``Llama.create_chat_completion``.

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
            formatted_text, add_special=self.llm._effective_add_bos
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
        available = self.n_ctx - token_count - _CTX_HEADROOM_TOKENS
        if available <= 0:
            raise ValueError(
                f"Prompt ({token_count} tokens) exceeds context ({self.n_ctx}). "
                "Reduce prompt length or increase n_ctx."
            )
        if requested is not None:
            return min(requested, available)
        return available

    def _sampling_kwargs(self, stop: list[str] | None) -> dict[str, Any]:
        """Build per-call sampling kwargs from this backend's ModelConfig.

        Centralizes the kwargs construction so ``generate``,
        ``generate_with_thinking``, and ``UnifiedLLM.chat`` all use the same
        recipe. Stop sequences from ``self.config.stop_sequences`` are
        unioned with the caller's ``stop`` argument.
        """
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
        if self.speculative:
            kwargs["speculative"] = True
            if self.n_draft_max is not None:
                kwargs["n_draft_max"] = self.n_draft_max
        return kwargs


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
    )
    # _clean_response patterns — pre-compiled here so each call avoids the
    # implicit re-module LRU cache lookup and stays compile-free even when
    # the module-level cache is evicted under churn.
    _CLEAN_THINK_BLOCK: ClassVar[re.Pattern[str]] = re.compile(
        r"<think(?:ing)?>.*?</think(?:ing)?>\s*", re.DOTALL
    )
    _CLEAN_THINK_BRACKET: ClassVar[re.Pattern[str]] = re.compile(
        r"\[THINK\].*?\[/THINK\]\s*", re.DOTALL | re.IGNORECASE
    )
    _CLEAN_GEMMA_CHANNEL: ClassVar[re.Pattern[str]] = re.compile(
        r"<\|channel>.*?<channel\|>\s*", re.DOTALL
    )
    _CLEAN_THINK_OPEN: ClassVar[re.Pattern[str]] = re.compile(
        r"<think(?:ing)?>.*", re.DOTALL
    )
    _CLEAN_THINK_BRACKET_OPEN: ClassVar[re.Pattern[str]] = re.compile(
        r"\[THINK\].*", re.DOTALL | re.IGNORECASE
    )
    _CLEAN_GEMMA_CHANNEL_OPEN: ClassVar[re.Pattern[str]] = re.compile(
        r"<\|channel>.*", re.DOTALL
    )
    _CLEAN_THINK_TAG_LEAD: ClassVar[re.Pattern[str]] = re.compile(r"^/(?:no_)?think\n?")
    _CLEAN_TURN_SPLIT: ClassVar[re.Pattern[str]] = re.compile(
        r"<(?:start|end)_of_turn>(?:user|model)?\n?"
    )
    _THINK_BRACKET_SPLIT: ClassVar[re.Pattern[str]] = re.compile(
        r"\[THINK\]", re.IGNORECASE
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
        reset_kv_cache: bool = True,
        cache_prompt: bool = True,
    ) -> str:
        messages = self._build_messages(prompt, system_prompt, thinking=thinking)
        _, _, n_tokens = self.llm._prepare_chat(messages)
        max_tokens = self._calc_max_tokens_from_count(n_tokens, max_tokens)

        kwargs = self._sampling_kwargs(stop)
        kwargs["reset_kv_cache"] = reset_kv_cache
        kwargs["cache_prompt"] = cache_prompt
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
        reset_kv_cache: bool = True,
        cache_prompt: bool = True,
    ) -> tuple[str, str]:
        if not self.config.supports_thinking:
            return "", self.generate(
                prompt,
                system_prompt,
                max_tokens,
                stop=stop,
                reset_kv_cache=reset_kv_cache,
                cache_prompt=cache_prompt,
            )

        messages = self._build_messages(prompt, system_prompt, thinking=True)
        _, _, n_tokens = self.llm._prepare_chat(messages)
        max_tokens = self._calc_max_tokens_from_count(n_tokens, max_tokens)

        kwargs = self._sampling_kwargs(stop)
        kwargs["reset_kv_cache"] = reset_kv_cache
        kwargs["cache_prompt"] = cache_prompt
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
            system_prompt = (
                f"<|think|>{system_prompt}" if system_prompt else "<|think|>"
            )

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Qwen 3.5 / 3.6 use the chatml ``/think`` and ``/no_think`` user-side
        # toggles to switch reasoning behavior at runtime.  Gemma 4 has its
        # own ``<|think|>`` system-prefix mechanism (handled above) and does
        # NOT understand /think — never append the suffix for it.
        if (
            self.config.family in (ModelFamily.QWEN3_5, ModelFamily.QWEN3_6)
            and self.config.supports_thinking
        ):
            suffix = " /think" if thinking else " /no_think"
            prompt = prompt + suffix

        messages.append({"role": "user", "content": prompt})
        return messages

    def _clean_response(self, text: str) -> str:
        """Strip thinking tags and control tokens from response."""
        # Handle complete thinking blocks
        text = self._CLEAN_THINK_BLOCK.sub("", text)
        text = self._CLEAN_THINK_BRACKET.sub("", text)
        # Gemma 4 thinking block
        text = self._CLEAN_GEMMA_CHANNEL.sub("", text)
        # Handle unclosed thinking tags (truncated output)
        text = self._CLEAN_THINK_OPEN.sub("", text)
        text = self._CLEAN_THINK_BRACKET_OPEN.sub("", text)
        text = self._CLEAN_GEMMA_CHANNEL_OPEN.sub("", text)
        text = self._CLEAN_THINK_TAG_LEAD.sub("", text)
        if "/response" in text:
            text = text.split("/response", 1)[1]
        if "<start_of_turn>" in text or "<end_of_turn>" in text:
            parts = self._CLEAN_TURN_SPLIT.split(text)
            for part in parts:
                part = part.strip()
                if part and not part.startswith("<"):
                    text = part
                    break
        text = self._CONTROL_TOKENS.sub("", text)
        return text.strip()

    def strip_thinking(self, text: str) -> str:
        """Return only the answer portion of a thinking-aware response."""
        _, answer = self._parse_thinking(text)
        return answer

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
            return self._THINK_BRACKET_SPLIT.split(text)[1].strip(), ""
        return "", text.strip()


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
        >>> llm = UnifiedLLM("models/Qwen3.5-4B-Q4_K_M.gguf")
        >>> print(llm.generate("Hello"))

        >>> # With thinking mode
        >>> print(llm.generate("Solve x^2 = 4", thinking=True))

        >>> # As context manager
        >>> with UnifiedLLM("models/model.gguf") as llm:
        ...     print(llm.generate("Hi"))
    """

    # All four supported families share the chat-template backend; family-
    # specific behavior (Gemma 4 ``<|think|>`` prefix, Qwen ``/think`` suffix)
    # is handled inside ChatTemplateBackend._build_messages.
    BACKEND_MAP: ClassVar[dict[ModelFamily, type[Backend]]] = {
        ModelFamily.QWEN3_5: ChatTemplateBackend,
        ModelFamily.QWEN3_6: ChatTemplateBackend,
        ModelFamily.GEMMA4: ChatTemplateBackend,
        ModelFamily.GRANITE: ChatTemplateBackend,
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
        cache_type_k: int = 1,
        cache_type_v: int = 1,
        speculative: bool | str = "auto",
        n_draft_max: int | None = None,
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
            cache_type_k: ggml_type for K cache (default 1=f16). Use
                ``GGML_TYPE_Q8_0`` etc. from ``llama_cpp`` to quantize.
            cache_type_v: ggml_type for V cache (default 1=f16). Quantized V
                typically requires flash attention, which is enabled by default.
            speculative: Draft-MTP speculative decoding mode. ``"auto"``
                (default) probes ``Context.supports_speculative_mtp()`` after
                the model loads and enables speculative iff the checkpoint
                exposes an MTP graph (e.g. Qwen3.6 *-MTP.gguf). ``True``
                forces it on and raises if the model lacks an MTP graph.
                ``False`` disables it unconditionally. Speculative requires
                the user-facing context to use the DEFAULT graph variant —
                UnifiedLLM always loads with ``LLAMA_CONTEXT_TYPE_DEFAULT``
                so this is satisfied by construction.
            n_draft_max: Number of draft tokens proposed per verify step
                (range [1, 8]). None defers to the SamplingParams default
                (2). Has no effect when speculative is disabled.

        Raises:
            ValueError: If model family cannot be detected, or
                ``speculative=True`` was forced but the model lacks an MTP
                graph.
        """
        # Initialize close-state FIRST, before any operation that can raise.
        # If __init__ fails partway through, close() (including the atexit
        # cleanup handler) needs to observe _closed=False to take the
        # resource-release path rather than short-circuiting as "already
        # closed" and leaking whatever was loaded so far.
        self._closed: bool = False
        # self.llm / self.backend are typed as non-None elsewhere; during
        # __init__ they're transiently None until the model loads. Callers
        # are shielded from that state by _check_closed(), which fires as
        # soon as any public method runs.
        self.llm: Llama = None  # type: ignore[assignment]
        self.backend: Backend = None  # type: ignore[assignment]

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
            cache_type_k=cache_type_k,
            cache_type_v=cache_type_v,
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

        # Resolve speculative mode. "auto" probes the model's MTP support
        # after load; True/False are forced. supports_speculative_mtp() is
        # cheap (a single bool check on the loaded model).
        self.speculative = self._resolve_speculative(speculative)
        self.n_draft_max = n_draft_max

        try:
            backend_cls = self.BACKEND_MAP[self.model_config.family]
            self.backend = backend_cls(
                self.llm,
                self.model_config,
                n_ctx,
                speculative=self.speculative,
                n_draft_max=n_draft_max,
            )
        except Exception:
            self.llm.close()
            self.llm = None  # type: ignore[assignment]
            raise

        # Register for cleanup at exit (lazy registration on first instance).
        # The finalizer discards under _instances_lock (RLock — finalizers can
        # fire on a thread already holding it).
        _register_unified_cleanup()

        def _discard_unified_ref(r: weakref.ref[Any]) -> None:
            with _instances_lock:
                _unified_instances.discard(r)

        self._ref = weakref.ref(self, _discard_unified_ref)
        with _instances_lock:
            _unified_instances.add(self._ref)

    def _resolve_speculative(self, mode: bool | str) -> bool:
        """Resolve the ``speculative`` constructor argument to a bool.

        ``"auto"`` enables speculative iff the model exposes an MTP graph;
        ``True`` requires it (raises if missing); ``False`` disables it.
        Logs the decision so operators can confirm whether the speedup is
        actually engaged at startup.
        """
        has_mtp = self.llm.ctx.supports_speculative_mtp()
        if mode == "auto":
            if has_mtp:
                logging.info(
                    "UnifiedLLM: MTP graph detected, enabling speculative decoding"
                )
                return True
            return False
        if mode is True:
            if not has_mtp:
                raise ValueError(
                    "speculative=True was requested but the loaded model does "
                    "not expose an MTP graph variant. Use a *-MTP.gguf "
                    "checkpoint (e.g. Qwen3.6-MoE) or speculative='auto'."
                )
            return True
        if mode is False:
            return False
        raise ValueError(f"speculative must be True, False, or 'auto'; got {mode!r}")

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

    @property
    def speculative_enabled(self) -> bool:
        """Whether draft-MTP speculative decoding is engaged for this instance."""
        return self.speculative

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        *,
        thinking: bool = False,
        stop: list[str] | None = None,
        reset_kv_cache: bool = True,
        cache_prompt: bool = True,
    ) -> str:
        """Generate text response.

        Args:
            prompt: User prompt text.
            system_prompt: Optional system prompt.
            max_tokens: Maximum tokens to generate (auto if None).
            thinking: Enable thinking mode (Qwen 3.5 / 3.6, Gemma 4).
            stop: Additional stop sequences.
            reset_kv_cache: Forwarded to ``Llama.create_chat_completion``.
                Default True clears KV before this turn (safe but discards
                prior prefix-cache state).
            cache_prompt: Forwarded to ``Llama.create_chat_completion``.
                Only meaningful when ``reset_kv_cache=False``.

        Returns:
            Generated text response.
        """
        self._check_closed()
        return self.backend.generate(
            prompt,
            system_prompt,
            max_tokens,
            thinking=thinking,
            stop=stop,
            reset_kv_cache=reset_kv_cache,
            cache_prompt=cache_prompt,
        )

    def generate_with_thinking(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        *,
        stop: list[str] | None = None,
        reset_kv_cache: bool = True,
        cache_prompt: bool = True,
    ) -> tuple[str, str]:
        """Generate with separate thinking and answer.

        Args:
            prompt: User prompt text.
            system_prompt: Optional system prompt.
            max_tokens: Maximum tokens to generate.
            stop: Additional stop sequences.
            reset_kv_cache: Forwarded to ``Llama.create_chat_completion``.
            cache_prompt: Forwarded to ``Llama.create_chat_completion``.

        Returns:
            Tuple of (thinking_text, answer_text).
        """
        self._check_closed()
        return self.backend.generate_with_thinking(
            prompt,
            system_prompt,
            max_tokens,
            stop=stop,
            reset_kv_cache=reset_kv_cache,
            cache_prompt=cache_prompt,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        thinking: bool = False,
        stop: list[str] | None = None,
        sanitize_history: bool = True,
        reset_kv_cache: bool = True,
        cache_prompt: bool = True,
        stream: bool = False,
    ) -> str | Iterator[str]:
        """Multi-turn chat entry point.

        Renders the messages through the chat template, runs generation, and
        returns the assistant's content with thinking blocks stripped (when
        ``stream=False``).

        ``sanitize_history`` defaults to True for any family with thinking
        support — Unsloth's Gemma 4 guidance is "do not feed prior thought
        blocks back into the next turn", and the same hygiene helps Qwen 3.5
        / 3.6 thinking models stay coherent across turns. Pass
        ``sanitize_history=False`` to inject raw history yourself (e.g. if
        you want the model to see its prior reasoning verbatim).

        Args:
            messages: Full conversation. ``system`` / ``user`` / ``assistant``
                roles supported. Not mutated in place.
            max_tokens: Cap on generated tokens (auto-clamped to context).
            thinking: For Qwen 3.5 / 3.6 only — appends ``/think`` to the
                last user turn. Gemma 4 thinking is handled per-message via
                its ``<|think|>`` system prefix; pass thinking=True to
                trigger it (UnifiedLLM auto-injects the prefix).
            stop: Additional stop sequences (combined with family defaults).
            sanitize_history: When True (default), strip thinking blocks
                from prior assistant turns before sending. Set False to
                opt out — useful for replaying a saved transcript verbatim.
            reset_kv_cache: Forwarded to the underlying Llama. Default True
                clears KV before this turn (safe but discards prior
                prefix-cache state). Set False with cache_prompt=True for
                fast multi-turn continuation.
            cache_prompt: Forwarded to the underlying Llama. Only meaningful
                when reset_kv_cache=False; trims KV to the longest matching
                prefix and decodes only the divergent suffix. See
                ``docs/API.md`` for the full contract.
            stream: When True, return an iterator of text deltas (raw — no
                thinking-block stripping). The caller is responsible for
                rendering thinking output. Use ``self.strip_thinking(...)``
                or ``re`` patterns on the joined output to post-process.

        Returns:
            ``str`` (default) — the assistant message content, with thinking
            output removed.

            ``Iterator[str]`` (when ``stream=True``) — text deltas as they
            arrive, raw (no stripping).
        """
        self._check_closed()

        effective_messages = (
            self.sanitize_history(messages)
            if sanitize_history and self.model_config.supports_thinking
            else list(messages)
        )

        # Apply family-specific thinking-mode plumbing on the *last user turn*.
        # We mutate a copy, never the caller's list. The helper is called
        # for both thinking=True and thinking=False on supports_thinking
        # families because Qwen 3.5 / 3.6 default to reasoning-on at the
        # model level — without an explicit /no_think suffix the model
        # ignores thinking=False and reasons anyway.
        effective_messages = [dict(m) for m in effective_messages]
        if self.model_config.supports_thinking:
            self._apply_thinking_to_last_user(effective_messages, thinking)

        # Tokenize once via _prepare_chat — Llama.create_chat_completion will
        # also tokenize inside, but token-count budgeting needs the value
        # *here* to clamp max_tokens before dispatch. We accept the second
        # tokenization downstream (avoiding it would require a private
        # token-list entry point on Llama).
        _, _, token_count = self.llm._prepare_chat(effective_messages)
        max_tokens_resolved = self.backend._calc_max_tokens_from_count(
            token_count, max_tokens
        )

        kwargs = self.backend._sampling_kwargs(stop)
        kwargs["max_tokens"] = max_tokens_resolved
        kwargs["reset_kv_cache"] = reset_kv_cache
        kwargs["cache_prompt"] = cache_prompt

        if stream:
            kwargs["stream"] = True
            chunks = cast(
                Iterator[dict[str, Any]],
                self.llm.create_chat_completion(effective_messages, **kwargs),
            )
            return self._stream_chat_deltas(chunks)

        resp = cast(
            dict[str, Any],
            self.llm.create_chat_completion(effective_messages, **kwargs),
        )
        text = resp["choices"][0]["message"]["content"] or ""
        return self.backend.strip_thinking(text)

    @staticmethod
    def _stream_chat_deltas(
        chunks: Iterator[dict[str, Any]],
    ) -> Iterator[str]:
        """Unwrap OpenAI-style chat.completion.chunk deltas into raw text.

        Each chunk has shape ``{"choices": [{"delta": {"content": "..."}, ...}]}``.
        We yield the ``delta.content`` strings only, skipping empty-content
        chunks (the final stop-marker chunk has ``delta == {}``).
        """
        for chunk in chunks:
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            piece = delta.get("content")
            if piece:
                yield piece

    def _apply_thinking_to_last_user(
        self, messages: list[dict[str, Any]], thinking: bool
    ) -> None:
        """Mutate ``messages`` in place to set thinking-mode on the last user turn.

        Family-specific:
        - Qwen 3.5 / 3.6: append ``/think`` or ``/no_think`` based on flag.
          Both suffixes are explicit because Qwen defaults to reasoning-on;
          ``thinking=False`` without ``/no_think`` is silently ignored.
        - Gemma 4: prepend ``<|think|>`` to the system message (or insert one)
          only when ``thinking=True``. Gemma defaults to non-thinking, so
          ``thinking=False`` is the no-op path.
        - Granite: no thinking spec from upstream; no-op.
        """
        family = self.model_config.family
        if family in (ModelFamily.QWEN3_5, ModelFamily.QWEN3_6):
            suffix = " /think" if thinking else " /no_think"
            for m in reversed(messages):
                if m.get("role") == "user" and isinstance(m.get("content"), str):
                    m["content"] = f"{m['content']}{suffix}"
                    return
        elif family is ModelFamily.GEMMA4 and thinking:
            # Find an existing system message and prepend the prefix.
            for m in messages:
                if m.get("role") == "system" and isinstance(m.get("content"), str):
                    if not m["content"].lstrip().startswith("<|think|>"):
                        m["content"] = f"<|think|>{m['content']}"
                    return
            # No system message — insert one.
            messages.insert(0, {"role": "system", "content": "<|think|>"})

    def strip_thinking(self, text: str) -> str:
        """Remove thinking tags from text, return only the answer.

        Delegates to the backend's ``strip_thinking`` method. Backends that
        don't produce thinking output return the text unchanged (ABC default).

        Args:
            text: Text potentially containing thinking tags.

        Returns:
            Text with thinking content removed.
        """
        return self.backend.strip_thinking(text)

    def sanitize_history(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Strip thinking / reasoning blocks from historical assistant turns.

        Unsloth's Gemma 4 guidance is explicit: "only keep the final visible
        answer in chat history. Do not feed prior thought blocks back into
        the next turn." Qwen 3 / 3.5 thinking models benefit from the same
        hygiene. Call this on your conversation list before each new turn:

            history.append({"role": "assistant", "content": raw_response})
            next_turn_messages = llm.sanitize_history(history) + [
                {"role": "user", "content": next_user_prompt},
            ]

        Args:
            messages: Conversation so far. Not mutated in place.

        Returns:
            New list with assistant-turn ``content`` run through the
            backend's ``strip_thinking``. System/user turns pass through.
        """
        cleaned: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role == "assistant" and isinstance(content, str):
                cleaned.append({**msg, "content": self.backend.strip_thinking(content)})
            else:
                cleaned.append(dict(msg))
        return cleaned

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
        # _closed is set in __init__ before any operation that can raise, so
        # direct access is safe. If close() runs after a partial __init__
        # failure, self.llm / self.backend may still be None — the
        # per-attribute guards below handle that.
        if self._closed:
            return
        self._closed = True
        # Remove from instance tracking
        if hasattr(self, "_ref"):
            with _instances_lock:
                _unified_instances.discard(self._ref)
        if hasattr(self, "llm") and self.llm is not None:
            self.llm.close()
            self.llm = None  # type: ignore[assignment]
        if hasattr(self, "backend") and self.backend is not None:
            # Null the backend's reference to the closed Llama before
            # dropping our own reference. This way any stray caller who
            # stashed a `backend` handle gets a clean AttributeError instead
            # of operating on a closed Llama (which would fail deeper with
            # a less clear message).
            self.backend.llm = None  # type: ignore[assignment]
            self.backend = None  # type: ignore[assignment]
