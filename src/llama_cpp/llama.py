"""Pythonic interface around the nanobind llama.cpp bindings."""

import asyncio
import atexit
import codecs
from collections.abc import AsyncIterator, Iterator, Sequence
import contextlib
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import fields as _dc_fields
from dataclasses import replace as dc_replace
import gc
import json
import logging
import math
import os
import queue
import threading
import time
from typing import Any
import uuid
import weakref

from . import _about  # noqa: F401
from . import _llama  # type: ignore[attr-defined]  # C++ extension module


def _uuid7_hex() -> str:
    """Generate a UUID7 hex string (Python 3.14+)."""
    return str(uuid.uuid7().hex)


# ---------------------------------------------------------------------------
# Instance tracking for cleanup at exit
# ---------------------------------------------------------------------------
_instances: set[weakref.ref[Any]] = set()
_shutdown_called = False
_cleanup_registered = False
_llama_initialized = False
_cleanup_lock = threading.Lock()

# ggml_type constants for KV cache quantization (cache_type_k, cache_type_v).
# Values mirror ggml.h's `enum ggml_type`.
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1  # default
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_IQ4_NL = 20
GGML_TYPE_BF16 = 30
# Newer ggml weight types (NOT valid for KV cache, but exposed for forward
# compatibility with newer GGUFs that may quantize tensors with these types).
GGML_TYPE_MXFP4 = 39
GGML_TYPE_NVFP4 = 40
GGML_TYPE_Q1_0 = 41

# llama_context_type — selects the graph variant a context constructs.
# MTP (Multi-Token Prediction) requires a model that ships MTP layers
# (currently Qwen3.5 and Qwen3.5-MoE MTP checkpoints); on a model without
# MTP layers, context construction fails.
LLAMA_CONTEXT_TYPE_DEFAULT = 0
LLAMA_CONTEXT_TYPE_MTP = 1
_VALID_CONTEXT_TYPES: frozenset[int] = frozenset(
    {LLAMA_CONTEXT_TYPE_DEFAULT, LLAMA_CONTEXT_TYPE_MTP}
)

# ggml_types accepted by llama.cpp for KV cache (k/v). Used for validation in
# LlamaConfig. Keep in sync with llama.cpp's llama_kv_cache_unified; unsupported
# types (e.g. k-quants like Q4_K) crash context construction.
_VALID_CACHE_TYPES: frozenset[int] = frozenset(
    {
        GGML_TYPE_F32,
        GGML_TYPE_F16,
        GGML_TYPE_BF16,
        GGML_TYPE_Q4_0,
        GGML_TYPE_Q4_1,
        GGML_TYPE_Q5_0,
        GGML_TYPE_Q5_1,
        GGML_TYPE_Q8_0,
        GGML_TYPE_IQ4_NL,
    }
)

# Configuration constants
_ALL_GPU_LAYERS_SENTINEL = (
    1_000_000  # Special value meaning "offload all layers to GPU"
)
_MAX_PROMPT_LENGTH = 10_000_000  # Maximum prompt length in characters (10MB limit)
_MAX_STOP_SEQUENCES = 20  # Maximum number of stop sequences allowed
_MAX_STOP_SEQUENCE_LENGTH = 500  # Maximum length of each stop sequence in characters
_MAX_PROMPT_MULTIPLIER = 2  # Maximum multiplier for tokenized prompt validation


def _register_cleanup() -> None:
    """Register cleanup handlers only after a model is successfully loaded."""
    global _cleanup_registered
    if _cleanup_registered:
        return
    with _cleanup_lock:
        if _cleanup_registered:
            return
        atexit.register(_cleanup_all)
        _cleanup_registered = True


def mark_llama_initialized() -> None:
    """Mark that llama.cpp has been initialized via a model load."""
    global _llama_initialized
    should_register = False
    with _cleanup_lock:
        if _llama_initialized:
            return
        _llama_initialized = True
        should_register = True
    # Call outside lock to avoid deadlock (register_cleanup also uses _cleanup_lock)
    if should_register:
        _register_cleanup()


def _cleanup_all() -> None:
    """Close all Llama instances before interpreter shutdown."""
    global _shutdown_called
    # Check shutdown state with lock to prevent race conditions
    with _cleanup_lock:
        if _shutdown_called:
            return
        if not _llama_initialized:
            return
        _shutdown_called = True
    # Perform cleanup outside lock to avoid deadlock
    for ref in list(_instances):
        instance = ref()
        if instance is not None:
            with contextlib.suppress(Exception):
                instance.close()
    _instances.clear()
    gc.collect()
    # Free llama.cpp backend only if all models released (guarded in C++)
    with contextlib.suppress(Exception):
        _llama.backend_free()


def shutdown() -> None:
    """Explicitly shutdown all Llama instances and free backend resources.

    Call this at the end of your program before exit() to avoid segfaults
    when using logging or other modules that hold references during cleanup.

    Example:
        from llama_cpp import shutdown

        def main():
            with UnifiedLLM(...) as llm:
                ...
            shutdown()  # Clean up before Python's shutdown sequence

        if __name__ == "__main__":
            main()
    """
    _cleanup_all()


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class LlamaError(Exception):
    """Base exception for llama-cpp-nanobind errors."""


class ModelLoadError(LlamaError):
    """Failed to load model file."""


class GenerationError(LlamaError):
    """Text generation failed."""


class ValidationError(LlamaError):
    """Invalid input parameters."""


# ---------------------------------------------------------------------------
# Configuration Classes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SamplingParams:
    """Sampling configuration mirroring llama-cpp-python defaults."""

    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.0
    # Locally-typical sampling (Meister et al., arXiv:2202.00666).
    # 1.0 = disabled (matches llama.cpp convention).
    typical_p: float = 1.0
    min_keep: int = 1
    repeat_penalty: float = 1.1
    repeat_last_n: int = 64
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int | None = None
    # Dynamic temperature
    temp_delta: float = 0.0
    temp_exponent: float = 1.0
    # XTC sampler
    xtc_probability: float = 0.0
    xtc_threshold: float = 0.1
    # Top-n-sigma (negative = disabled)
    top_n_sigma: float = -1.0
    # DRY (Don't Repeat Yourself) anti-repetition
    dry_multiplier: float = 0.0
    dry_base: float = 1.75
    dry_allowed_length: int = 2
    dry_penalty_last_n: int = -1
    dry_seq_breakers: list[str] | None = None
    # Adaptive-P (llama.cpp PR #17927): terminal sampler that selects tokens near
    # a target probability over time. When enabled (target >= 0), it REPLACES the
    # dist sampler at the end of the chain. Recommended: combine with mild
    # truncation (e.g. min_p) only, and disable temperature.
    adaptive_p_target: float = -1.0
    adaptive_p_decay: float = 0.85
    # Logit bias: per-token additive bias applied before all other samplers.
    # Maps token_id -> bias (use float("-inf") or a large negative to ban).
    # OpenAI-API parity (mirrors `logit_bias` in chat/completions).
    logit_bias: dict[int, float] | None = None
    # Draft-MTP speculative decoding: max number of draft tokens proposed per
    # verify step. Active only when generate(..., speculative=True) is set on
    # an MTP-capable context. Range [1, 8]; default 2 follows unsloth.
    n_draft_max: int = 2

    def __post_init__(self) -> None:
        """Validate sampling parameters."""
        if self.temperature < 0:
            raise ValidationError("temperature must be non-negative")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValidationError("top_p must be between 0.0 and 1.0")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValidationError("min_p must be between 0.0 and 1.0")
        if not 0.0 < self.typical_p <= 1.0:
            raise ValidationError("typical_p must be in (0.0, 1.0] (1.0 = disabled)")
        if self.top_k < 0:
            raise ValidationError("top_k must be non-negative (0 = disabled)")
        if self.repeat_penalty < 0:
            raise ValidationError("repeat_penalty must be non-negative")
        if self.min_keep < 1:
            raise ValidationError("min_keep must be at least 1")
        if self.temp_delta < 0:
            raise ValidationError("temp_delta must be non-negative")
        if self.temp_exponent <= 0:
            raise ValidationError("temp_exponent must be positive")
        if not 0.0 <= self.xtc_probability <= 1.0:
            raise ValidationError("xtc_probability must be between 0.0 and 1.0")
        if self.xtc_threshold < 0:
            raise ValidationError("xtc_threshold must be non-negative")
        if self.dry_multiplier < 0:
            raise ValidationError("dry_multiplier must be non-negative")
        if self.dry_base <= 0:
            raise ValidationError("dry_base must be positive")
        if self.dry_allowed_length < 1:
            raise ValidationError("dry_allowed_length must be at least 1")
        if self.dry_seq_breakers is None:
            self.dry_seq_breakers = ["\n", ":", '"', "*"]
        # adaptive-p: target < 0 disables; otherwise must be in [0, 1].
        if self.adaptive_p_target >= 0.0 and self.adaptive_p_target > 1.0:
            raise ValidationError(
                "adaptive_p_target must be in [0.0, 1.0] when enabled (or negative to disable)"
            )
        if not 0.0 <= self.adaptive_p_decay <= 0.99:
            raise ValidationError("adaptive_p_decay must be in [0.0, 0.99]")
        if self.logit_bias is not None:
            for token_id, bias in self.logit_bias.items():
                if not isinstance(token_id, int) or token_id < 0:
                    raise ValidationError(
                        f"logit_bias token id must be a non-negative int, got {token_id!r}"
                    )
                if not isinstance(bias, (int, float)) or math.isnan(float(bias)):
                    raise ValidationError(
                        f"logit_bias[{token_id}] must be a real number, got {bias!r}"
                    )
        if not isinstance(self.n_draft_max, int) or not 1 <= self.n_draft_max <= 8:
            raise ValidationError(
                f"n_draft_max must be an int in [1, 8]; got {self.n_draft_max!r}"
            )

    def to_native(self) -> _llama.SamplerParams:
        native = _llama.SamplerParams()
        native.top_k = self.top_k
        native.top_p = float(self.top_p)
        native.min_p = float(self.min_p)
        native.typical_p = float(self.typical_p)
        native.min_keep = int(self.min_keep)
        native.temp = float(self.temperature)
        native.penalty_last_n = int(self.repeat_last_n)
        native.repeat_penalty = float(self.repeat_penalty)
        native.freq_penalty = float(self.frequency_penalty)
        native.presence_penalty = float(self.presence_penalty)
        native.seed = -1 if self.seed is None else int(self.seed)
        native.temp_delta = float(self.temp_delta)
        native.temp_exponent = float(self.temp_exponent)
        native.xtc_probability = float(self.xtc_probability)
        native.xtc_threshold = float(self.xtc_threshold)
        native.top_n_sigma = float(self.top_n_sigma)
        native.dry_multiplier = float(self.dry_multiplier)
        native.dry_base = float(self.dry_base)
        native.dry_allowed_length = int(self.dry_allowed_length)
        native.dry_penalty_last_n = int(self.dry_penalty_last_n)
        native.dry_seq_breakers = list(self.dry_seq_breakers or [])
        native.adaptive_p_target = float(self.adaptive_p_target)
        native.adaptive_p_decay = float(self.adaptive_p_decay)
        if self.logit_bias:
            native.logit_bias = [
                (int(tok), float(bias)) for tok, bias in self.logit_bias.items()
            ]
        return native


@dataclass(slots=True)
class LlamaConfig:
    model_path: str
    n_ctx: int = 4096
    n_batch: int = 2048
    n_ubatch: int = 512
    n_seq_max: int = 1  # Max parallel sequences (1 = single sequence, simplest)
    # Recurrent-state snapshots per seq for partial-rollback (0 = no rollback).
    # Required for draft-MTP speculative decoding on hybrid recurrent models
    # like Qwen3.6-MoE: must be >= n_draft_max so rejected drafts can be
    # rolled back from the recurrent state. Default 2 matches the default
    # SamplingParams.n_draft_max; bump to your max n_draft_max when you go
    # higher. Each unit costs additional VRAM proportional to the recurrent
    # state size, so don't oversize.
    n_rs_seq: int = 2
    n_threads: int | None = None
    n_threads_batch: int | None = None
    n_gpu_layers: int = -1
    main_gpu: int = 0
    split_mode: int = 0
    use_mmap: bool = True
    use_mlock: bool = False
    check_tensors: bool = False
    no_host: bool = False
    flash_attn: int = 1
    ctx_type: int = (
        LLAMA_CONTEXT_TYPE_DEFAULT  # 0=default, 1=MTP (Multi-Token Prediction)
    )
    offload_kqv: bool = True
    embeddings: bool = False
    rope_freq_base: float = 0.0
    rope_freq_scale: float = 0.0
    cache_type_k: int = 1  # ggml_type for K cache (1=f16, 3=q4_1, etc.)
    cache_type_v: int = 1  # ggml_type for V cache (1=f16, 3=q4_1, etc.)
    add_bos: bool | None = None  # None = auto-detect from model preference
    parse_special: bool = False
    chat_format: str | None = None  # e.g. "llama-2", "chatml", "gemma", etc.
    verbose: bool = True  # WARNING: Affects logging GLOBALLY (llama.cpp limitation)
    seed: int = -1  # RNG seed (-1 for random)

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.n_ctx < 1:
            raise ValidationError("n_ctx must be at least 1")
        if self.n_batch < 1:
            raise ValidationError("n_batch must be at least 1")
        if self.n_ubatch < 1:
            raise ValidationError("n_ubatch must be at least 1")
        if self.n_ubatch > self.n_batch:
            self.n_ubatch = self.n_batch  # Enforce n_batch >= n_ubatch
        if self.n_gpu_layers < -1:
            raise ValidationError("n_gpu_layers must be >= -1 (-1 means all layers)")
        if self.n_seq_max < 1:
            raise ValidationError("n_seq_max must be at least 1")
        if self.ctx_type not in _VALID_CONTEXT_TYPES:
            raise ValidationError(
                f"ctx_type={self.ctx_type} is not a supported llama_context_type. "
                f"Use LLAMA_CONTEXT_TYPE_DEFAULT (0) or LLAMA_CONTEXT_TYPE_MTP (1)."
            )
        if self.cache_type_k not in _VALID_CACHE_TYPES:
            raise ValidationError(
                f"cache_type_k={self.cache_type_k} is not a supported ggml_type "
                f"for KV cache. Use one of the GGML_TYPE_* constants from "
                f"llama_cpp (F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL)."
            )
        if self.cache_type_v not in _VALID_CACHE_TYPES:
            raise ValidationError(
                f"cache_type_v={self.cache_type_v} is not a supported ggml_type "
                f"for KV cache. Use one of the GGML_TYPE_* constants from "
                f"llama_cpp (F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL)."
            )
        # Quantized V-cache requires flash attention; llama.cpp produces
        # NaN/garbage output otherwise. F16/F32/BF16 V is safe without FA.
        if (
            self.cache_type_v not in {GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_BF16}
            and self.flash_attn == 0
        ):
            raise ValidationError(
                "Quantized cache_type_v requires flash_attn=1 "
                "(quantized V without flash attention produces invalid output)"
            )


class Llama:
    """High level client compatible with llama-cpp-python's :class:`Llama` API.

    **WARNING: NOT THREAD-SAFE** - Do not call methods concurrently from multiple
    threads on the same instance. Use multiple instances or LlamaPool for parallelism.

    Supports context manager protocol for automatic resource cleanup:
        with Llama("model.gguf") as llm:
            text = llm.generate("Hello", max_tokens=32)

    Thread Safety Details:
        - Sync methods (generate, create_chat_completion, etc.) are NOT thread-safe.
          Concurrent calls from multiple threads may cause crashes or data corruption.
        - Async methods use an internal lock to serialize concurrent calls on the same
          instance. This prevents crashes but provides no parallelism benefit.
        - For true parallel inference, use LlamaPool with multiple independent instances.
        - verbose=False affects logging globally across all instances (llama.cpp limitation).
          Construction is internally serialized via a class-level lock
          (``Llama._log_lock``); the verbose setting of the most recently
          constructed instance wins. No external synchronization is required.
    """

    # Class-level lock for global state changes (logging)
    _log_lock = threading.Lock()
    _global_verbose: bool | None = None  # Track if verbose has been set globally
    # Max time generate_stream will wait for its worker thread to exit after
    # cancellation. If the worker is still inside a C++ call after this
    # timeout, we consider the instance permanently unsafe; see
    # generate_stream's finally block.
    _STREAM_JOIN_TIMEOUT: float = 5.0

    def __init__(
        self,
        model_path: str,
        *,
        config: LlamaConfig | None = None,
        sampling: SamplingParams | None = None,
    ) -> None:
        cfg = config or LlamaConfig(model_path=model_path)
        self.config = cfg
        self.sampling = sampling or SamplingParams()
        self._metadata_cache: dict[str, str] | None = None
        self._closed = False
        self._lock = threading.Lock()  # Thread safety for async methods
        self._lora_adapters: list[Any] = []  # Keep adapters alive
        self._lora_configs: list[
            tuple[str, float]
        ] = []  # (path, scale) for reapplication
        # Mirror of tokens currently materialized in seq 0 of the KV cache.
        # Used by cache_prompt-style prefix reuse: when a new prompt shares
        # a leading prefix with this list, we trim the divergent suffix from
        # the KV cache and only decode the new tail. Any code path that
        # mutates KV state outside the generation loop (kv_cache_clear,
        # kv_cache_seq_*, set_state_data, load_state, reset) must call
        # _invalidate_prompt_cache() to keep this mirror consistent.
        self._cached_prompt_tokens: list[int] = []

        # Apply verbose setting with class-level synchronization
        # WARNING: This affects logging globally, not per-instance.
        # Logging can be re-enabled by creating instance with verbose=True after disable.
        with Llama._log_lock:
            if not cfg.verbose:
                # Disable logging globally if not already disabled
                if Llama._global_verbose is not False:
                    disable_logging()
                    Llama._global_verbose = False
            elif cfg.verbose and Llama._global_verbose is False:
                # Re-enable logging after previous disable
                reset_logging()  # Reset to default llama.cpp logging
                Llama._global_verbose = True

        # Apply seed to default sampling if specified
        if cfg.seed >= 0 and self.sampling.seed is None:
            self.sampling = SamplingParams(
                **{**asdict(self.sampling), "seed": cfg.seed}
            )

        model_params = _llama.ModelParams()
        # llama.cpp treats negative n_gpu_layers as "all layers" in the CLI wrapper,
        # but the low-level API expects a non-negative count. Translate -1 to a large
        # sentinel so users can keep using -1 to mean "full offload".
        gpu_layers = (
            cfg.n_gpu_layers if cfg.n_gpu_layers >= 0 else _ALL_GPU_LAYERS_SENTINEL
        )
        model_params.n_gpu_layers = gpu_layers
        model_params.main_gpu = cfg.main_gpu
        model_params.split_mode = cfg.split_mode
        model_params.use_mmap = cfg.use_mmap
        model_params.use_mlock = cfg.use_mlock
        model_params.check_tensors = cfg.check_tensors
        model_params.no_host = cfg.no_host

        ctx_params = _llama.ContextParams()
        ctx_params.n_ctx = int(cfg.n_ctx)
        ctx_params.n_batch = int(cfg.n_batch)
        ctx_params.n_ubatch = int(cfg.n_ubatch)
        ctx_params.n_seq_max = int(cfg.n_seq_max)
        ctx_params.n_rs_seq = int(cfg.n_rs_seq)
        ctx_params.n_threads = int(cfg.n_threads or os.cpu_count() or 1)
        ctx_params.n_threads_batch = int(cfg.n_threads_batch or ctx_params.n_threads)
        ctx_params.flash_attn_type = int(cfg.flash_attn)
        ctx_params.ctx_type = int(cfg.ctx_type)
        ctx_params.offload_kqv = bool(cfg.offload_kqv)
        ctx_params.embeddings = bool(cfg.embeddings)
        ctx_params.rope_freq_base = float(cfg.rope_freq_base)
        ctx_params.rope_freq_scale = float(cfg.rope_freq_scale)
        ctx_params.type_k = int(cfg.cache_type_k)
        ctx_params.type_v = int(cfg.cache_type_v)

        try:
            self.model = _llama.Model(cfg.model_path, model_params)
        except RuntimeError as e:
            raise ModelLoadError(f"Failed to load model: {cfg.model_path}") from e

        # Resolve add_bos once, after the model is loaded, into an internal
        # attribute. The user-supplied config is not mutated, so the same
        # LlamaConfig instance can be safely shared across Llama(...) calls
        # for different models.
        self._effective_add_bos: bool = (
            bool(self.model.get_add_bos()) if cfg.add_bos is None else bool(cfg.add_bos)
        )

        try:
            self.ctx = _llama.Context(self.model, ctx_params)
        except RuntimeError as e:
            # Ensure model is released if context creation fails
            with contextlib.suppress(Exception):
                self.model.close()
            self.model = None
            raise ModelLoadError(f"Failed to create context: {e}") from e

        mark_llama_initialized()

        # Register for cleanup at exit
        self._ref = weakref.ref(self, lambda r: _instances.discard(r))
        _instances.add(self._ref)

    def __enter__(self) -> Llama:
        return self

    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        if self._closed:
            return "<Llama (closed)>"
        model_name = os.path.basename(self.config.model_path)
        return (
            f"<Llama model={model_name!r} "
            f"n_ctx={self.config.n_ctx} n_gpu_layers={self.config.n_gpu_layers}>"
        )

    # Note: __del__ intentionally omitted - C++ RAII handles cleanup via nanobind,
    # and the atexit handler _cleanup_all() handles explicit shutdown. Using __del__
    # can cause segfaults during interpreter shutdown when accessing partially
    # destroyed Python objects. Use context manager or explicit close() instead.

    @classmethod
    def reset_verbose(cls) -> None:
        """Reset the global verbose state to allow re-configuration.

        This is a convenience method to reset logging state without creating
        a new instance. Useful for testing or manual logging control.

        Example:
            >>> Llama.reset_verbose()  # Reset to default state
            >>> llm = Llama(model_path, config=LlamaConfig(..., verbose=True))
        """
        with cls._log_lock:
            reset_logging()
            cls._global_verbose = None

    def _check_closed(self) -> None:
        """Raise error if instance has been closed."""
        if self._closed:
            raise LlamaError("Llama instance has been closed")

    @staticmethod
    def _validate_stop_sequences(stop: Sequence[str | int] | None) -> None:
        """Validate stop sequences against configured limits.

        Args:
            stop: Stop sequences to validate (strings or token IDs).

        Raises:
            ValidationError: If validation fails (too many sequences or sequence too long).
        """
        if not stop:
            return

        if len(stop) > _MAX_STOP_SEQUENCES:
            raise ValidationError(
                f"too many stop sequences (max {_MAX_STOP_SEQUENCES})"
            )

        for item in stop:
            if isinstance(item, str) and len(item) > _MAX_STOP_SEQUENCE_LENGTH:
                raise ValidationError(
                    f"stop sequence too long (max {_MAX_STOP_SEQUENCE_LENGTH} chars)"
                )

    @staticmethod
    def _validate_sampling_overrides(overrides: dict[str, Any]) -> None:
        """Reject unknown kwargs at public entry points.

        Without this, an unknown override (e.g. a typo or a kwarg meant for a
        different API, like OpenAI's `logprobs`) would propagate to
        `SamplingParams(**...)` inside `_build_sampler` and surface as a
        generic ``TypeError: got an unexpected keyword argument`` with no
        indication that the user's public call is the source.
        """
        if not overrides:
            return
        valid = {f.name for f in _dc_fields(SamplingParams)}
        unknown = set(overrides) - valid
        if unknown:
            raise ValidationError(
                f"unknown sampling override(s): {sorted(unknown)!r}. "
                f"Valid keys: {sorted(valid)!r}"
            )

    def _tokenize_stop_sequences(
        self, stop: Sequence[str | int] | None
    ) -> list[list[int]]:
        """Convert user-supplied stops (strings or token IDs) to token-id lists."""
        if not stop:
            return []
        stop_sequences: list[list[int]] = []
        for item in stop:
            if isinstance(item, str):
                tks = self.tokenize(item, add_special=False, parse_special=False)
                if tks:
                    stop_sequences.append([int(t) for t in tks])
            else:
                stop_sequences.append([int(item)])
        return stop_sequences

    def _validate_prompt_token_count(self, n_tokens: int) -> None:
        """Reject tokenized prompts that would exceed _MAX_PROMPT_MULTIPLIER * n_ctx."""
        max_reasonable_tokens = self.n_ctx() * _MAX_PROMPT_MULTIPLIER
        if n_tokens > max_reasonable_tokens:
            raise ValidationError(
                f"tokenized prompt ({n_tokens} tokens) exceeds "
                f"reasonable limit ({max_reasonable_tokens}). "
                "Reduce prompt length or increase n_ctx."
            )

    def _validate_speculative(self, speculative: bool) -> None:
        """Validate the precondition for ``speculative=True`` calls.

        Speculative decoding requires:

        * The user-facing context is the default graph variant
          (``ctx_type=LLAMA_CONTEXT_TYPE_DEFAULT``). The MTP graph is used
          internally for the draft context only.
        * The model exposes an MTP graph variant
          (``Context.supports_speculative_mtp()``).
        * The context is **not** embeddings-only.
        """
        if not speculative:
            return
        if self.config.embeddings:
            raise ValidationError(
                "speculative=True is incompatible with embeddings-only contexts"
            )
        if int(self.config.ctx_type) != LLAMA_CONTEXT_TYPE_DEFAULT:
            raise ValidationError(
                f"speculative=True requires LlamaConfig("
                f"ctx_type=LLAMA_CONTEXT_TYPE_DEFAULT); got "
                f"ctx_type={self.config.ctx_type}. The MTP graph is used "
                "internally as the draft context — do not pass it as the "
                "user-facing ctx_type."
            )
        if not self.ctx.supports_speculative_mtp():
            raise ValidationError(
                "speculative=True requires a model with an MTP graph "
                "variant (e.g. Qwen3.6-MoE *-MTP.gguf checkpoints). The "
                "loaded model does not expose one."
            )

    def close(self) -> None:
        """Release model and context resources."""
        # _closed is set in __init__ before any operation that can fail, so
        # direct access is safe and makes init-time bugs visible instead of
        # swallowing them as a no-op close.
        if self._closed:
            return

        # Serialize shutdown with active generation/async paths
        with self._lock:
            if self._closed:
                return

            # Remove from instance tracking first
            if hasattr(self, "_ref"):
                _instances.discard(self._ref)
            if hasattr(self, "_lora_adapters"):
                self._lora_adapters.clear()

            # Track exceptions during cleanup
            close_errors: list[Exception] = []

            # Explicitly free context before model (C++ dependency)
            # Use try/finally to ensure cleanup even on exception
            if getattr(self, "ctx", None) is not None:
                try:
                    self.ctx.close()
                except Exception as e:
                    close_errors.append(e)
                finally:
                    self.ctx = None

            if getattr(self, "model", None) is not None:
                try:
                    self.model.close()
                except Exception as e:
                    close_errors.append(e)
                finally:
                    self.model = None

            # Drop prompt-cache mirror — KV is gone, mirror is meaningless
            self._cached_prompt_tokens = []

            # Mark as closed AFTER attempting all cleanup
            self._closed = True

        # Force GC outside lock to avoid potential deadlocks from finalizers
        gc.collect()

        # Raise first exception if any cleanup failed
        if close_errors:
            raise LlamaError(
                f"Errors during close: {close_errors[0]}"
            ) from close_errors[0]

    # Compatibility helpers -------------------------------------------------
    def tokenize(
        self,
        text: str,
        *,
        add_special: bool | None = None,
        parse_special: bool | None = None,
    ) -> list[int]:
        self._check_closed()
        return list(
            self.model.tokenize(
                text,
                add_special=(
                    self._effective_add_bos if add_special is None else add_special
                ),
                parse_special=(
                    self.config.parse_special
                    if parse_special is None
                    else parse_special
                ),
            )
        )

    def detokenize(
        self,
        tokens: Sequence[int],
        *,
        remove_special: bool = True,
        unparse_special: bool = False,
    ) -> str:
        self._check_closed()
        raw: bytes = self.model.detokenize_bytes(
            list(tokens), remove_special=remove_special, unparse_special=unparse_special
        )
        return raw.decode("utf-8", errors="replace")

    def detokenize_bytes(
        self,
        tokens: Sequence[int],
        *,
        remove_special: bool = True,
        unparse_special: bool = False,
    ) -> bytes:
        """Return raw bytes from detokenization (no UTF-8 validation).

        Useful for incremental/streaming decoding where individual tokens
        may produce incomplete multi-byte UTF-8 sequences.
        """
        self._check_closed()
        return bytes(
            self.model.detokenize_bytes(
                list(tokens),
                remove_special=remove_special,
                unparse_special=unparse_special,
            )
        )

    def n_tokens(self, text: str, *, add_special: bool = False) -> int:
        """Return number of tokens in text.

        Args:
            text: Text to tokenize.
            add_special: Whether to include BOS token in count.
        """
        return len(self.tokenize(text, add_special=add_special))

    def kv_cache_clear(self) -> None:
        """Clear the KV cache."""
        self._check_closed()
        self.ctx.kv_cache_clear()
        self._invalidate_prompt_cache()

    def token_bos(self) -> int:
        """Return BOS token id."""
        self._check_closed()
        result: int = self.model.bos()
        return result

    def token_eos(self) -> int:
        """Return EOS token id."""
        self._check_closed()
        result: int = self.model.eos()
        return result

    def token_eot(self) -> int:
        """Return EOT token id."""
        self._check_closed()
        result: int = self.model.eot()
        return result

    def n_ctx(self) -> int:
        """Return context size."""
        self._check_closed()
        result: int = self.ctx.n_ctx()
        return result

    def n_vocab(self) -> int:
        """Return vocabulary size."""
        self._check_closed()
        result: int = self.model.n_vocab()
        return result

    def n_embd(self) -> int:
        """Return embedding dimension."""
        self._check_closed()
        result: int = self.model.n_embd()
        return result

    def model_size(self) -> int:
        """Return total size of all tensors in bytes."""
        self._check_closed()
        result: int = self.model.model_size()
        return result

    def n_params(self) -> int:
        """Return total number of parameters."""
        self._check_closed()
        result: int = self.model.n_params()
        return result

    def n_layer(self) -> int:
        """Return number of layers."""
        self._check_closed()
        result: int = self.model.n_layer()
        return result

    def n_head(self) -> int:
        """Return number of attention heads."""
        self._check_closed()
        result: int = self.model.n_head()
        return result

    def has_encoder(self) -> bool:
        """Return whether model has an encoder component."""
        self._check_closed()
        result: bool = self.model.has_encoder()
        return result

    def has_decoder(self) -> bool:
        """Return whether model has a decoder component."""
        self._check_closed()
        result: bool = self.model.has_decoder()
        return result

    def is_recurrent(self) -> bool:
        """Return whether model uses recurrent architecture."""
        self._check_closed()
        result: bool = self.model.is_recurrent()
        return result

    def is_hybrid(self) -> bool:
        """Return whether model uses hybrid attention architecture."""
        self._check_closed()
        result: bool = self.model.is_hybrid()
        return result

    def token_sep(self) -> int:
        """Return separator token id."""
        self._check_closed()
        result: int = self.model.sep()
        return result

    def token_nl(self) -> int:
        """Return newline token id."""
        self._check_closed()
        result: int = self.model.nl()
        return result

    def token_pad(self) -> int:
        """Return padding token id."""
        self._check_closed()
        result: int = self.model.pad()
        return result

    def get_add_bos(self) -> bool:
        """Return whether model prefers BOS token to be added."""
        self._check_closed()
        result: bool = self.model.get_add_bos()
        return result

    def kv_cache_seq_pos_min(self, seq_id: int = 0) -> int:
        """Return minimum position in KV cache for sequence."""
        self._check_closed()
        result: int = self.ctx.kv_cache_seq_pos_min(seq_id)
        return result

    def memory_can_shift(self) -> bool:
        """Return whether memory supports KV cache shifting."""
        self._check_closed()
        result: bool = self.ctx.memory_can_shift()
        return result

    def set_embeddings(self, enabled: bool) -> None:
        """Enable or disable embedding extraction at runtime."""
        self._check_closed()
        self.ctx.set_embeddings(enabled)

    def set_causal_attn(self, enabled: bool) -> None:
        """Enable or disable causal attention at runtime."""
        self._check_closed()
        self.ctx.set_causal_attn(enabled)

    def get_chat_template(self, name: str = "") -> str:
        """Return model's chat template. Empty string if not available."""
        self._check_closed()
        result: str = self.model.chat_template(name)
        return result

    def token_to_piece(self, token: int) -> str:
        """Return the text representation of a token."""
        self._check_closed()
        result: str = self.model.token_to_piece(token)
        return result

    @property
    def metadata(self) -> dict[str, str]:
        """Return model metadata as a dictionary (cached)."""
        self._check_closed()
        if self._metadata_cache is None:
            result = {}
            count = self.model.meta_count()
            for i in range(count):
                key = self.model.meta_key_by_index(i)
                val = self.model.meta_val_by_index(i)
                if key:
                    result[key] = val
            self._metadata_cache = result
        return self._metadata_cache

    def _invalidate_prompt_cache(self) -> None:
        """Drop the mirror of KV-cached prompt tokens.

        Call after any operation that mutates seq 0 outside of the normal
        generation flow (full kv_cache_clear, direct kv_cache_seq_*,
        set_state_data, load_state, ctx.reset). Subsequent generations will
        fall back to a full prompt decode until the mirror is rebuilt.
        """
        self._cached_prompt_tokens = []

    def _apply_prefix_reuse(self, new_tokens: list[int]) -> int:
        """Trim KV seq 0 to the longest common prefix with ``new_tokens``.

        Returns the number of leading tokens that are guaranteed to be in the
        KV cache after this call (i.e. the LCP length). The caller is
        responsible for decoding ``new_tokens[n_match:]`` and updating
        ``self._cached_prompt_tokens`` once generation begins.

        If LCP equals the full cached length, no llama_memory_seq_rm call is
        issued. If LCP is shorter, the divergent suffix is removed from KV
        and ``cur_pos_`` is updated by the C++ binding.

        Hybrid-attention models (Qwen3.5, Granite 4 hybrid, …) and other
        configurations report ``memory_can_shift() == False``; for those,
        ``llama_memory_seq_rm(seq, p0, -1)`` returns False without modifying
        the KV cache. In that case we fall back to a full clear so KV stays
        consistent with the mirror — caller pays the full prompt-decode cost
        for that turn but correctness is preserved.

        The mirror is replaced with ``new_tokens`` here; generation appends
        produced tokens via _commit_generation_to_cache. On unexpected error
        the caller should invalidate the mirror via _invalidate_prompt_cache.
        """
        cached = self._cached_prompt_tokens
        n_match = 0
        limit = min(len(cached), len(new_tokens))
        for i in range(limit):
            if cached[i] != new_tokens[i]:
                break
            n_match += 1
        if n_match < len(cached):
            # Trim divergent suffix; bindings update cur_pos_ via seq_pos_max+1
            # iff the trim succeeded. Hybrid-attention models silently refuse
            # mid-sequence trim; fall back to a full clear so KV doesn't end
            # up with stale tokens past the new prefix.
            ok = self.ctx.kv_cache_seq_rm(0, n_match, -1)
            if not ok:
                self.ctx.kv_cache_clear()
                self._cached_prompt_tokens = list(new_tokens)
                return 0
        # Mirror the new prompt; generation extends it as tokens arrive.
        self._cached_prompt_tokens = list(new_tokens)
        return n_match

    def _commit_generation_to_cache(self, generated: Sequence[int]) -> None:
        """Append generated tokens to the prompt-cache mirror.

        Stop tokens are NOT decoded and NOT in ``generated`` (see
        bindings/llama_cpp.cpp stop-handling), so this is just a tail extend.
        """
        if generated:
            self._cached_prompt_tokens.extend(generated)

    def _apply_adapters(self) -> None:
        """Push current adapter list to the C++ context."""
        scales = [scale for _, scale in self._lora_configs]
        self.ctx.set_adapters_lora(self._lora_adapters, scales)

    def _reapply_lora_adapters(self) -> None:
        """Reapply all loaded LoRA adapters after context reset."""
        if self._lora_configs:
            self._lora_adapters.clear()
            for path, _scale in self._lora_configs:
                adapter = _llama.LoraAdapter(self.model, path)
                self._lora_adapters.append(adapter)
            self._apply_adapters()

    def embed(self, text: str) -> list[float]:
        """Get embedding for text. Clears KV cache."""
        if not self.config.embeddings:
            raise ValidationError(
                "Embeddings not enabled. Set embeddings=True in LlamaConfig."
            )
        self.ctx.kv_cache_clear()
        self._invalidate_prompt_cache()
        tokens = self.tokenize(text)
        self.ctx.decode(tokens, return_logits=False)
        return list(self.ctx.embeddings())

    def create_embedding(
        self, input: str | list[str], model: str | None = None
    ) -> dict[str, Any]:
        """Create embeddings in OpenAI-compatible format.

        Args:
            input: Text string or list of strings to embed.
            model: Optional model identifier for response.

        Returns:
            OpenAI-compatible embedding response dict.

        Note:
            Each input is processed independently with a cleared KV cache.
            For high-throughput batch embedding, consider using LlamaPool
            to parallelize across multiple model instances.
        """
        if not self.config.embeddings:
            raise ValidationError(
                "Embeddings not enabled. Set embeddings=True in LlamaConfig."
            )

        inputs = [input] if isinstance(input, str) else list(input)

        data = []
        total_tokens = 0
        # Each embedding pass clears KV; ensure prompt-cache mirror is dropped
        # once for the batch — subsequent loop iterations re-clear, but the
        # mirror stays empty regardless.
        self._invalidate_prompt_cache()
        for i, text in enumerate(inputs):
            tokens = self.tokenize(text)
            total_tokens += len(tokens)
            self.ctx.kv_cache_clear()
            self.ctx.decode(tokens, return_logits=False)
            embedding = list(self.ctx.embeddings())
            data.append(
                {
                    "object": "embedding",
                    "index": i,
                    "embedding": embedding,
                }
            )

        return {
            "object": "list",
            "data": data,
            "model": model or os.path.basename(self.config.model_path),
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens,
            },
        }

    # Generation ------------------------------------------------------------
    def _build_sampler(
        self, sampling: SamplingParams | None = None, **overrides: Any
    ) -> _llama.SamplerChain:
        params_obj = sampling or self.sampling
        # Allow llama-cpp-python style kw overrides (temperature, top_p, etc.)
        # Filter only None (not 0, 0.0, or False) so that explicit zero-valued
        # overrides like temperature=0.0 or top_k=0 are respected.
        if overrides:
            params_obj = SamplingParams(
                **{
                    **asdict(params_obj),
                    **{k: v for k, v in overrides.items() if v is not None},
                }
            )
        params = params_obj.to_native()
        return _llama.SamplerChain(self.model, params)

    def _format_chat_messages(self, messages: Sequence[dict[str, Any]]) -> str:
        """Format chat messages using llama.cpp chat template or fallback."""
        # Convert messages to (role, content) pairs
        msg_pairs: list[tuple[str, str]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "".join(
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in content
                )
            msg_pairs.append((role, content))

        # Try to use llama.cpp chat template
        chat_format = self.config.chat_format or ""
        try:
            result: str = _llama.chat_apply_template(
                self.model, msg_pairs, chat_format, True
            )
            return result
        except Exception as e:
            # Fallback to simple format — log so users can diagnose template issues
            logging.debug(
                "chat_apply_template failed (format=%r): %s; using simple format",
                chat_format,
                e,
            )
            parts = [f"{role}: {content}" for role, content in msg_pairs]
            parts.append("assistant:")
            return "\n".join(parts)

    def _prepare_chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        add_bos: bool | None = None,
    ) -> tuple[str, list[int], int]:
        """Prepare chat for generation: format and tokenize once.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            add_bos: Override BOS behavior. If None, uses the resolved
                effective add_bos (model preference or explicit config).

        Returns:
            Tuple of (formatted_prompt, prompt_tokens, token_count).
        """
        prompt = self._format_chat_messages(messages)
        effective_bos = self._effective_add_bos if add_bos is None else add_bos
        tokens = self.tokenize(prompt, add_special=effective_bos)
        return prompt, tokens, len(tokens)

    def _token_to_text_incremental(self, tokens: Iterator[int]) -> Iterator[str]:
        """Convert token stream to text with incremental UTF-8 decoding.

        Handles multi-byte UTF-8 characters that may be split across token
        boundaries by using an incremental decoder that accumulates incomplete
        byte sequences.

        Args:
            tokens: Iterator of token IDs.

        Yields:
            Text pieces as UTF-8 decoding completes. Empty strings are filtered out.
        """
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        for tok in tokens:
            raw = self.detokenize_bytes(
                [tok], remove_special=True, unparse_special=True
            )
            text_piece = decoder.decode(raw)
            if text_piece:
                yield text_piece

        # Flush any remaining bytes in the decoder buffer
        final_piece = decoder.decode(b"", final=True)
        if final_piece:
            yield final_piece

    def _generate_from_tokens(
        self,
        prompt_tokens: list[int],
        *,
        max_tokens: int,
        sampler: _llama.SamplerChain,
        stop_sequences: list[list[int]] | None = None,
        grammar: Any | None = None,
        reset_kv_cache: bool = True,
        add_bos: bool | None = None,
        cache_prompt: bool = True,
        speculative: bool = False,
        n_draft_max: int = 2,
    ) -> list[int]:
        """Internal generation from pre-tokenized input.

        Args:
            prompt_tokens: Pre-tokenized prompt.
            max_tokens: Maximum tokens to generate.
            sampler: SamplerChain instance.
            stop_sequences: Optional multi-token stop sequences.
            grammar: Optional GrammarSampler for constrained generation.
            reset_kv_cache: Whether to clear KV cache before generation.
            add_bos: Override BOS behavior. If None, derived from config and
                reset_kv_cache (no BOS inserted mid-session).
            cache_prompt: When True (default), reuse the longest matching
                prefix of the previous prompt+completion that is still in the
                KV cache, decoding only the divergent suffix. Ignored when
                reset_kv_cache=True (full clear) or when ``prompt_tokens``
                shares no prefix with the mirror.

        Returns:
            List of generated token IDs.
        """
        if reset_kv_cache:
            self.ctx.kv_cache_clear()
            self._invalidate_prompt_cache()

        eos = self.model.eos()
        stop_seqs = stop_sequences or []
        # BOS rules:
        # - reset_kv_cache=True: use the model's BOS preference. C++ prepends
        #   BOS to its priming when needed.
        # - reset_kv_cache=False + cache_prompt=True: also use the model's
        #   BOS preference — C++ prepends BOS to its internal priming, the
        #   matching BOS already sits at position 0 of the mirror (and KV)
        #   from the first reset_kv_cache=True turn, so LCP covers the BOS
        #   slot and the suffix-only decode lines up. C++'s own
        #   "front == bos" guard prevents double-BOS when the user already
        #   tokenized with BOS.
        # - reset_kv_cache=False + cache_prompt=False (legacy manual session
        #   continuation): suppress BOS so the user can stitch turns by
        #   hand without injecting a stray BOS mid-stream.
        if add_bos is None:
            if reset_kv_cache or cache_prompt:
                effective_add_bos = self._effective_add_bos
            else:
                effective_add_bos = False
        else:
            effective_add_bos = add_bos

        # Build the priming sequence the C++ generator will see, so we can
        # compute skip_decode_prefix against the mirror. C++ prepends BOS
        # itself when add_bos=True; mirror it here so the LCP comparison
        # operates on identical sequences.
        primed = (
            [self.model.bos(), *prompt_tokens]
            if effective_add_bos
            and (not prompt_tokens or prompt_tokens[0] != self.model.bos())
            else list(prompt_tokens)
        )

        skip_decode_prefix = 0
        if reset_kv_cache:
            # Fresh KV: priming will be the entire seq 0 after this call.
            self._cached_prompt_tokens = list(primed)
        elif cache_prompt and self._cached_prompt_tokens:
            # Trim divergent suffix and reuse the matching prefix.
            skip_decode_prefix = self._apply_prefix_reuse(primed)
        elif cache_prompt:
            # Empty mirror but caching enabled: full prime, then track.
            self._cached_prompt_tokens = list(primed)
        else:
            # Caller opted out of caching mid-session — KV state is whatever
            # they were managing. Drop the mirror so the next cache_prompt
            # call falls back to a clean full prime.
            self._invalidate_prompt_cache()

        try:
            if speculative:
                generated = list(
                    _llama.generate_tokens_speculative_mtp(
                        self.ctx,
                        sampler,
                        grammar,
                        prompt_tokens,
                        int(max_tokens),
                        effective_add_bos,
                        eos,
                        int(n_draft_max),
                        stop_seqs,
                        None,  # no streaming callback in this entry point
                        skip_decode_prefix,
                    )
                )
            elif grammar is not None:
                generated = list(
                    _llama.generate_tokens_grammar_multi_stop(
                        self.ctx,
                        sampler,
                        grammar,
                        prompt_tokens,
                        int(max_tokens),
                        effective_add_bos,
                        eos,
                        stop_seqs,
                        skip_decode_prefix,
                    )
                )
            elif stop_seqs:
                generated = list(
                    _llama.generate_tokens_multi_stop(
                        self.ctx,
                        sampler,
                        prompt_tokens,
                        int(max_tokens),
                        effective_add_bos,
                        eos,
                        stop_seqs,
                        skip_decode_prefix,
                    )
                )
            else:
                generated = list(
                    _llama.generate_tokens(
                        self.ctx,
                        sampler,
                        prompt_tokens,
                        int(max_tokens),
                        effective_add_bos,
                        eos,
                        [],
                        skip_decode_prefix,
                    )
                )
        except Exception:
            # On C++ failure the KV state is unknown; drop the mirror so the
            # next call falls back to a clean full prime.
            self._invalidate_prompt_cache()
            raise

        # Only extend the mirror when caching is on; with cache_prompt=False
        # the mirror was just invalidated and must stay empty so the next
        # cache_prompt=True call falls back to a clean full prime.
        if cache_prompt or reset_kv_cache:
            self._commit_generation_to_cache(generated)
        return generated

    def generate_stream(
        self,
        prompt: str,
        *,
        max_tokens: int = 128,
        sampling: SamplingParams | None = None,
        stop: Sequence[str | int] | None = None,
        seed: int | None = None,
        reset_kv_cache: bool = True,
        cache_prompt: bool = True,
        speculative: bool = False,
        n_draft_max: int | None = None,
    ) -> Iterator[str]:
        """True streaming generation - yields text as tokens are decoded.

        Unlike generate(..., stream=True) which buffers all tokens first,
        this yields each token immediately as it's generated in a background thread.

        **Thread Safety & Locking Behavior:**
            - This method spawns a background thread that holds ``self._lock``
              for the entire generation duration.
            - Do NOT call other Llama methods (``generate``, ``create_chat_completion``,
              ``close``) from another thread while streaming is in progress.
              They will block until streaming completes.
            - The Llama class is NOT thread-safe. For concurrent inference,
              use ``LlamaPool`` with multiple instances instead.

        **Performance Note:**
            - In single-threaded Python code (typical use case), the lock has
              negligible overhead (~microseconds) due to no contention.
            - The background thread enables true incremental streaming with
              low latency, perfect for SSE/WebSocket endpoints.

        Args:
            prompt: Input prompt string.
            max_tokens: Maximum tokens to generate.
            sampling: Optional sampling parameters.
            stop: Optional stop sequences.
            seed: Optional RNG seed.
            reset_kv_cache: Clear KV cache before generation (default True).
            cache_prompt: When True (default) and reset_kv_cache=False, reuse
                the longest matching prefix of the previous turn's KV state and
                decode only the divergent suffix. Ignored when reset_kv_cache
                is True.

        Yields:
            Text chunks as they're generated.

        Example:
            >>> llm = Llama("model.gguf")
            >>> for chunk in llm.generate_stream("Hello"):
            ...     print(chunk, end="", flush=True)
        """
        self._check_closed()
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValidationError("prompt must be a non-empty string")
        if max_tokens < 1:
            raise ValidationError("max_tokens must be positive")
        if len(prompt) > _MAX_PROMPT_LENGTH:
            raise ValidationError(
                f"prompt exceeds maximum length ({_MAX_PROMPT_LENGTH} chars)"
            )
        self._validate_stop_sequences(stop)
        self._validate_speculative(speculative)
        effective_n_draft_max = (
            int(n_draft_max)
            if n_draft_max is not None
            else (
                sampling.n_draft_max
                if sampling is not None
                else self.sampling.n_draft_max
            )
        )

        sampler_params = sampling or self.sampling
        if seed is not None:
            sampler_params = dc_replace(sampler_params, seed=seed)
        sampler = self._build_sampler(sampler_params)

        if reset_kv_cache:
            self.ctx.kv_cache_clear()
            self._invalidate_prompt_cache()

        # Tokenize without BOS — C++ generator prepends based on add_bos and
        # the `prompt[0] != bos` guard.
        prompt_tokens = self.tokenize(prompt, add_special=False)
        self._validate_prompt_token_count(len(prompt_tokens))

        # BOS rules — see _generate_from_tokens for full rationale.
        if reset_kv_cache or cache_prompt:
            effective_add_bos = self._effective_add_bos
        else:
            effective_add_bos = False

        stop_sequences = self._tokenize_stop_sequences(stop)

        eos = self.model.eos()

        # Compute prefix reuse before spawning the worker. Mirror must reflect
        # the priming sequence the C++ generator sees: BOS-prepended when
        # add_bos is True. We compute skip_decode_prefix here under the lock
        # acquired below; appending generated tokens to the mirror happens in
        # finally on success.
        primed_for_mirror = (
            [self.model.bos(), *prompt_tokens]
            if effective_add_bos
            and (not prompt_tokens or prompt_tokens[0] != self.model.bos())
            else list(prompt_tokens)
        )
        skip_decode_prefix = 0
        if reset_kv_cache:
            self._cached_prompt_tokens = list(primed_for_mirror)
        elif cache_prompt and self._cached_prompt_tokens:
            skip_decode_prefix = self._apply_prefix_reuse(primed_for_mirror)
        elif cache_prompt:
            self._cached_prompt_tokens = list(primed_for_mirror)
        else:
            self._invalidate_prompt_cache()
        # Tokens generated in this stream — appended to mirror on success,
        # discarded if the worker dies. Lives outside the worker closure so
        # the finally block can read it.
        generated_in_stream: list[int] = []

        # Queue carries already-detokenized raw bytes from the worker thread,
        # never raw token IDs. This keeps all llama.cpp calls on the same Model
        # inside a single thread (Llama is not thread-safe); the main thread
        # only does UTF-8 decoding on bytes it pulled from the queue.
        token_queue: queue.Queue[bytes | None | Exception] = queue.Queue()
        cancel_event = threading.Event()

        def worker() -> None:
            """Background thread that generates tokens, detokenizes to bytes,
            and pushes raw bytes onto the queue.

            self._lock is already held by the caller before this thread was
            spawned; we do NOT re-acquire it here. Holding the lock across
            thread.start() + iteration is what guarantees no other Llama
            method can enter while streaming is in progress.
            """
            try:

                def on_token(token: int) -> bool:
                    if cancel_event.is_set():
                        return False  # Stop generation
                    try:
                        raw = bytes(
                            self.model.detokenize_bytes(
                                [token], remove_special=True, unparse_special=True
                            )
                        )
                    except Exception as exc:  # noqa: BLE001
                        token_queue.put(exc)
                        return False
                    # Record generated token for the prompt-cache mirror.
                    # The mirror is only committed if the worker exits cleanly
                    # (see finally below). C++ does not call on_token for
                    # stop tokens, so this list reflects what's actually in KV.
                    generated_in_stream.append(token)
                    token_queue.put(raw)
                    return True

                if speculative:
                    _llama.generate_tokens_speculative_mtp(
                        self.ctx,
                        sampler,
                        None,  # no grammar in generate_stream
                        prompt_tokens,
                        int(max_tokens),
                        effective_add_bos,
                        eos,
                        int(effective_n_draft_max),
                        stop_sequences,
                        on_token,
                        skip_decode_prefix,
                    )
                else:
                    _llama.generate_tokens_streaming(
                        self.ctx,
                        sampler,
                        prompt_tokens,
                        int(max_tokens),
                        effective_add_bos,
                        eos,
                        stop_sequences,
                        on_token,
                        skip_decode_prefix,
                    )
                token_queue.put(None)  # Sentinel: generation complete
            except Exception as e:
                token_queue.put(e)  # Propagate exception to main thread

        # Acquire the lock BEFORE spawning the worker, so that any concurrent
        # Llama call is blocked the moment generate_stream is invoked (not
        # just once the worker thread schedules the `with self._lock:` body).
        # Release it in the finally below — after the generator completes,
        # raises, or is closed by the caller.
        self._lock.acquire()
        thread: threading.Thread | None = None
        try:
            thread = threading.Thread(target=worker, daemon=True)
            thread.start()

            # Incremental UTF-8 decode of bytes from the queue. The worker has
            # already done all Model.detokenize_bytes calls; here we only decode.
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            while True:
                try:
                    queue_item = token_queue.get(timeout=0.5)
                except queue.Empty:
                    if not thread.is_alive():
                        raise RuntimeError(
                            "generate_stream worker thread died unexpectedly"
                        ) from None
                    continue
                if queue_item is None:
                    break  # Generation complete
                if isinstance(queue_item, Exception):
                    raise queue_item  # Propagate exception from worker thread
                piece = decoder.decode(queue_item)
                if piece:
                    yield piece
            # Flush any remaining bytes buffered in the decoder
            final_piece = decoder.decode(b"", final=True)
            if final_piece:
                yield final_piece
        finally:
            # Signal background thread to stop and wait for it to finish.
            # If it's still alive after the timeout, the underlying C++ call
            # has NOT returned — the Llama is now unsafe to reuse, because
            # another caller would race on ctx/model. We keep self._lock held
            # in that case so subsequent method calls block (loud hang) rather
            # than silently corrupt state. Process restart is the correct
            # recovery.
            cancel_event.set()
            worker_zombie = False
            if thread is not None:
                thread.join(timeout=self._STREAM_JOIN_TIMEOUT)
                worker_zombie = thread.is_alive()
            if not worker_zombie:
                # Commit the streamed tokens to the prompt-cache mirror only
                # on clean exit AND when caching is on. C++ skips on_token for
                # matched stop tokens, so generated_in_stream is exactly what's
                # now in KV beyond the priming sequence.
                if cache_prompt or reset_kv_cache:
                    self._commit_generation_to_cache(generated_in_stream)
                self._lock.release()
            else:
                # Worker is stuck in C++; KV state is unknown. Drop the mirror
                # so any future call (after lock recovery) starts clean.
                self._invalidate_prompt_cache()
                # Intentionally leave self._lock held and raise so the caller
                # learns of the problem. Do NOT release — another call
                # entering generation concurrently with a zombie worker would
                # race on the same ctx/model.
                raise LlamaError(
                    "generate_stream worker thread did not stop within "
                    f"{self._STREAM_JOIN_TIMEOUT}s; the C++ generation call "
                    "has not returned. This Llama instance is now unusable "
                    "(lock intentionally held to prevent data races). Restart "
                    "the process to recover."
                )

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 128,
        sampling: SamplingParams | None = None,
        stop: Sequence[str | int] | None = None,
        echo: bool = False,
        logprobs: int | None = None,
        stream: bool = False,
        seed: int | None = None,
        reset_kv_cache: bool = True,
        cache_prompt: bool = True,
        speculative: bool = False,
        n_draft_max: int | None = None,
    ) -> str | Iterator[str] | dict[str, Any]:
        """Generate text for ``prompt``.

        Args:
            prompt: Input prompt string.
            max_tokens: Maximum number of new tokens to generate (must be > 0).
            sampling: Optional overrides for sampling parameters.
            stop: Optional stop tokens or strings (multi-token supported).
            echo: If True, include prompt in the returned text.
            logprobs: If set, return token-level logprobs (top_n = logprobs).
            stream: If True, yields text chunks (note: generation completes first,
                    then chunks are yielded - not true incremental streaming).
            seed: Optional per-request RNG seed.
            reset_kv_cache: If True (default), clear KV cache before generation.
                Set to False for session-style continuation to reduce recompute.
            cache_prompt: When True (default) and reset_kv_cache=False, reuse
                the longest matching prefix of the previous turn that's still
                in the KV cache and decode only the divergent suffix. Ignored
                when reset_kv_cache=True.

        Raises:
            ValidationError: If prompt is not a string or max_tokens is invalid.
        """
        self._check_closed()
        # Input validation
        if not isinstance(prompt, str):
            raise ValidationError("prompt must be a string")
        if not prompt.strip():
            raise ValidationError("prompt cannot be empty")
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValidationError("max_tokens must be a positive integer")
        if len(prompt) > _MAX_PROMPT_LENGTH:
            raise ValidationError(
                f"prompt exceeds maximum length ({_MAX_PROMPT_LENGTH} chars)"
            )
        self._validate_stop_sequences(stop)
        self._validate_speculative(speculative)
        if speculative and logprobs is not None:
            raise ValidationError("logprobs is not supported on the speculative path")
        # Default n_draft_max from sampling params if caller didn't override.
        effective_n_draft_max = (
            int(n_draft_max)
            if n_draft_max is not None
            else (
                sampling.n_draft_max
                if sampling is not None
                else self.sampling.n_draft_max
            )
        )

        sampler_params = sampling or self.sampling
        if seed is not None:
            sampler_params = dc_replace(sampler_params, seed=seed)
        sampler = self._build_sampler(sampler_params)
        # We tokenize WITHOUT BOS — C++ generators prepend BOS independently
        # based on the `add_bos` argument and a `prompt[0] != bos` guard.
        # Whether C++ actually prepends is decided below per call mode.
        prompt_tokens = self.tokenize(prompt, add_special=False)
        self._validate_prompt_token_count(len(prompt_tokens))

        stop_sequences = self._tokenize_stop_sequences(stop)

        eos = self.model.eos()

        if stream and logprobs is not None:
            raise ValueError(
                "Streaming with logprobs is not supported; set stream=False or logprobs=None"
            )

        # BOS rules — see _generate_from_tokens for full rationale.
        if reset_kv_cache or cache_prompt:
            effective_add_bos = self._effective_add_bos
        else:
            effective_add_bos = False

        # Only use expensive details path when logprobs is actually requested.
        # echo alone is handled cheaply by prepending prompt tokens after
        # generation.
        token_probs = None
        if logprobs is not None:
            # Logprobs path: dispatch directly so we can pull TokenProb structs
            # back out. Replicate the prefix-reuse + KV-clear bookkeeping that
            # _generate_from_tokens does for the non-logprobs paths.
            primed = (
                [self.model.bos(), *prompt_tokens]
                if effective_add_bos
                and (not prompt_tokens or prompt_tokens[0] != self.model.bos())
                else list(prompt_tokens)
            )
            skip_decode_prefix = 0
            if reset_kv_cache:
                self.ctx.kv_cache_clear()
                self._cached_prompt_tokens = list(primed)
            elif cache_prompt and self._cached_prompt_tokens:
                skip_decode_prefix = self._apply_prefix_reuse(primed)
            elif cache_prompt:
                self._cached_prompt_tokens = list(primed)
            else:
                self._invalidate_prompt_cache()
            try:
                token_probs = _llama.generate_tokens_with_details(
                    self.ctx,
                    sampler,
                    prompt_tokens,
                    int(max_tokens),
                    effective_add_bos,
                    eos,
                    stop_sequences,
                    int(logprobs),
                    bool(echo),
                    skip_decode_prefix,
                )
            except Exception:
                self._invalidate_prompt_cache()
                raise
            # generate_tokens_with_details may include echoed prompt tokens at
            # the head of the returned list (when echo=True). Generated tokens
            # are everything after the priming length. Commit only those to
            # the mirror to keep it aligned with KV.
            primed_len = len(primed)
            tail = (
                [tp.token for tp in token_probs[primed_len:]]
                if echo
                else [tp.token for tp in token_probs]
            )
            if cache_prompt or reset_kv_cache:
                self._commit_generation_to_cache(tail)
            output_tokens = [tp.token for tp in token_probs]
        else:
            # No-logprobs paths route through _generate_from_tokens, which
            # handles prefix reuse, KV clear, mirror updates, and dispatch.
            output_tokens = self._generate_from_tokens(
                prompt_tokens,
                max_tokens=max_tokens,
                sampler=sampler,
                stop_sequences=stop_sequences,
                grammar=None,
                reset_kv_cache=reset_kv_cache,
                cache_prompt=cache_prompt,
                speculative=speculative,
                n_draft_max=effective_n_draft_max,
            )
            if echo:
                output_tokens = list(prompt_tokens) + output_tokens

        if logprobs is not None:
            text = self.detokenize(
                output_tokens, remove_special=True, unparse_special=False
            )
            return {
                "text": text,
                "tokens": output_tokens,
                "token_probs": token_probs,
            }

        if stream:
            return self._token_to_text_incremental(iter(output_tokens))

        text = self.detokenize(
            output_tokens, remove_special=True, unparse_special=False
        )
        return text

    def _generate_with_token_count(
        self,
        prompt: str,
        *,
        max_tokens: int = 128,
        sampling: SamplingParams | None = None,
        stop: Sequence[str | int] | None = None,
        echo: bool = False,
        seed: int | None = None,
        reset_kv_cache: bool = True,
        cache_prompt: bool = True,
    ) -> tuple[str, int]:
        """Generate text and return (text, n_generated_tokens).

        Used by ``__call__`` to avoid the detokenize → re-tokenize round-trip
        (which isn't lossless for merged/special tokens and wastes CPU).
        Mirrors the non-logprobs path of :meth:`generate`.
        """
        self._check_closed()
        if not isinstance(prompt, str):
            raise ValidationError("prompt must be a string")
        if not prompt.strip():
            raise ValidationError("prompt cannot be empty")
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValidationError("max_tokens must be a positive integer")
        if len(prompt) > _MAX_PROMPT_LENGTH:
            raise ValidationError(
                f"prompt exceeds maximum length ({_MAX_PROMPT_LENGTH} chars)"
            )
        self._validate_stop_sequences(stop)

        sampler_params = sampling or self.sampling
        if seed is not None:
            sampler_params = dc_replace(sampler_params, seed=seed)
        sampler = self._build_sampler(sampler_params)
        # Tokenize without BOS — _generate_from_tokens decides whether C++
        # prepends BOS based on (reset_kv_cache, cache_prompt).
        prompt_tokens = self.tokenize(prompt, add_special=False)
        self._validate_prompt_token_count(len(prompt_tokens))

        stop_sequences = self._tokenize_stop_sequences(stop)

        generated = self._generate_from_tokens(
            prompt_tokens,
            max_tokens=max_tokens,
            sampler=sampler,
            stop_sequences=stop_sequences,
            grammar=None,
            reset_kv_cache=reset_kv_cache,
            cache_prompt=cache_prompt,
        )

        n_generated = len(generated)
        output_tokens = list(prompt_tokens) + generated if echo else generated
        text = self.detokenize(
            output_tokens, remove_special=True, unparse_special=False
        )
        return text, n_generated

    # llama-cpp-python compatibility - __call__ returns OpenAI-style dict
    def __call__(
        self,
        prompt: str,
        *,
        max_tokens: int = 128,
        stop: Sequence[str | int] | None = None,
        echo: bool = False,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any] | Iterator[dict[str, Any]]:
        """Generate completion with OpenAI-compatible response format.

        Note: Streaming yields chunks after full generation (not true streaming).
        """
        prompt_tokens = self.tokenize(prompt, add_special=self._effective_add_bos)
        prompt_tok_count = len(prompt_tokens)

        if stream:

            def stream_chunks() -> Iterator[dict[str, Any]]:
                created = int(time.time())
                cmpl_id = f"cmpl-{_uuid7_hex()}"
                for chunk in self.generate(
                    prompt,
                    max_tokens=max_tokens,
                    stop=stop,
                    echo=echo,
                    stream=True,
                    **kwargs,
                ):
                    yield {
                        "id": cmpl_id,
                        "object": "text_completion",
                        "created": created,
                        "choices": [{"text": chunk, "index": 0, "finish_reason": None}],
                    }
                yield {
                    "id": cmpl_id,
                    "object": "text_completion",
                    "created": created,
                    "choices": [{"text": "", "index": 0, "finish_reason": "stop"}],
                }

            return stream_chunks()

        logprobs = kwargs.get("logprobs")
        if logprobs is not None:
            # Logprobs path: generate() returns a dict with {text, tokens, ...}
            # — token count is exact.
            result = self.generate(
                prompt,
                max_tokens=max_tokens,
                stop=stop,
                echo=echo,
                stream=False,
                **kwargs,
            )
            if not isinstance(result, dict):
                raise TypeError(f"Unexpected generate() return type: {type(result)}")
            text = result["text"]
            completion_tokens = len(result.get("tokens", []))
        else:
            # Non-logprobs path: call the internal token-level helper so we get
            # the exact generated token count without a lossy detokenize →
            # tokenize round-trip (which can miss special/merged tokens and
            # wastes CPU). Only pass kwargs the helper accepts.
            helper_kwargs = {
                k: kwargs[k]
                for k in ("sampling", "seed", "reset_kv_cache", "cache_prompt")
                if k in kwargs
            }
            text, completion_tokens = self._generate_with_token_count(
                prompt,
                max_tokens=max_tokens,
                stop=stop,
                echo=echo,
                **helper_kwargs,
            )
        created = int(time.time())

        return {
            "id": f"cmpl-{_uuid7_hex()}",
            "object": "text_completion",
            "created": created,
            "model": os.path.basename(self.config.model_path),
            "choices": [
                {"text": text, "index": 0, "logprobs": None, "finish_reason": "stop"}
            ],
            "usage": {
                "prompt_tokens": prompt_tok_count,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tok_count + completion_tokens,
            },
        }

    def create_completion(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Create completion (delegates to __call__ for consistency)."""
        result = self(prompt, **kwargs)
        # Type-check the result directly rather than re-reading kwargs: __call__
        # returns a dict for non-stream and an iterator for stream.
        if isinstance(result, dict):
            return result
        chunks: list[dict[str, Any]] = list(result)
        text = "".join(c["choices"][0]["text"] for c in chunks)
        return {
            "id": chunks[0]["id"] if chunks else f"cmpl-{_uuid7_hex()}",
            "object": "text_completion",
            "created": chunks[0]["created"] if chunks else int(time.time()),
            "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
        }

    # OpenAI-style / llama-cpp-python compatible chat API
    def create_chat_completion(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        max_tokens: int = 128,
        stream: bool = False,
        stop: Sequence[str | int] | None = None,
        response_format: dict[str, Any] | None = None,
        grammar: LlamaGrammar | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        reset_kv_cache: bool = True,
        cache_prompt: bool = True,
        speculative: bool = False,
        n_draft_max: int | None = None,
        **sampling_overrides: Any,
    ) -> dict[str, Any] | Iterator[dict[str, Any]]:
        """Chat completions endpoint compatible with llama-cpp-python.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            max_tokens: Maximum tokens to generate.
            stream: If True, yield chunks.
            stop: Stop sequences.
            response_format: {"type": "json_object"} or {"type": "json_object", "schema": {...}}
            grammar: LlamaGrammar instance for constrained generation.
            tools: List of tool/function definitions for function calling.
            tool_choice: "auto", "none", or {"type": "function", "function": {"name": "..."}}
            reset_kv_cache: If True (default), clear KV cache before generation.
                Set to False for multi-turn session continuation.
            cache_prompt: When True (default) and reset_kv_cache=False, reuse
                the longest matching prefix of the previous turn's KV state.
                Decodes only the divergent suffix (typically the new user
                message). Ignored when reset_kv_cache=True.
            **sampling_overrides: Override sampling params (temperature, top_p, etc.)
        """
        # Handle function calling by injecting tools into messages
        effective_messages = list(messages)
        if tools and tool_choice != "none":
            tools_prompt = _format_tools_prompt(tools)
            # Prepend or append to system message
            if effective_messages and effective_messages[0].get("role") == "system":
                effective_messages[0] = {
                    "role": "system",
                    "content": effective_messages[0]["content"] + "\n\n" + tools_prompt,
                }
            else:
                effective_messages.insert(
                    0, {"role": "system", "content": tools_prompt}
                )

            # Force JSON output for function calling
            if response_format is None:
                response_format = {"type": "json_object"}

        # Validate stop sequences
        self._validate_stop_sequences(stop)
        # Validate sampling overrides at the public boundary so that a typo
        # (e.g. "tempeature=0.8") or a foreign kwarg (e.g. OpenAI-only
        # "logprobs") surfaces as a clear ValidationError here, not as a
        # confusing TypeError deep in SamplingParams.__init__.
        self._validate_sampling_overrides(sampling_overrides)
        self._validate_speculative(speculative)
        effective_n_draft_max = (
            int(n_draft_max)
            if n_draft_max is not None
            else int(sampling_overrides.get("n_draft_max", self.sampling.n_draft_max))
        )

        # Tokenize without BOS — the chat template may already include BOS
        # as a literal, and _generate_from_tokens applies its BOS rule based
        # on (reset_kv_cache, cache_prompt). Avoid double-tokenizing BOS here.
        _, prompt_tokens, n_prompt_tokens = self._prepare_chat(
            effective_messages, add_bos=False
        )
        # Same DoS guard as generate() / generate_stream(): reject high-
        # compression prompts that tokenize to > 2×n_ctx.
        self._validate_prompt_token_count(n_prompt_tokens)
        sampler = self._build_sampler(None, **sampling_overrides)

        stop_sequences = self._tokenize_stop_sequences(stop)

        # Determine grammar from response_format or explicit grammar
        use_grammar = None
        if grammar is not None:
            grammar._ensure_sampler(self.model)
            use_grammar = grammar._sampler
        elif response_format is not None:
            fmt_type = response_format.get("type", "")
            if fmt_type == "json_object":
                schema = response_format.get("schema")
                grammar_str = (
                    _json_schema_to_grammar(schema) if schema else JSON_GRAMMAR
                )
                use_grammar = _create_grammar_sampler(self.model, grammar_str, "root")

        # Use unified generation path
        generated = self._generate_from_tokens(
            prompt_tokens,
            max_tokens=max_tokens,
            sampler=sampler,
            stop_sequences=stop_sequences,
            grammar=use_grammar,
            reset_kv_cache=reset_kv_cache,
            cache_prompt=cache_prompt,
            speculative=speculative,
            n_draft_max=effective_n_draft_max,
        )

        created = int(time.time())
        cmpl_id = f"chatcmpl-{_uuid7_hex()}"
        model_id = os.path.basename(self.config.model_path)

        if stream:

            def stream_chunks() -> Iterator[dict[str, Any]]:
                # Stream text pieces with incremental UTF-8 decoding
                for text_piece in self._token_to_text_incremental(iter(generated)):
                    yield {
                        "id": cmpl_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": text_piece},
                                "finish_reason": None,
                            }
                        ],
                    }
                # Final chunk with finish_reason
                yield {
                    "id": cmpl_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }

            return stream_chunks()

        text = self.detokenize(generated, remove_special=True, unparse_special=False)
        prompt_tok_count = len(prompt_tokens)
        completion_tok_count = len(generated)

        # Parse tool calls if tools were provided
        message: dict[str, Any] = {"role": "assistant", "content": text}
        finish_reason = "stop"

        if tools and tool_choice != "none":
            tool_calls = _parse_tool_calls(text)
            if tool_calls:
                message["tool_calls"] = tool_calls
                message["content"] = None
                finish_reason = "tool_calls"

        return {
            "id": cmpl_id,
            "object": "chat.completion",
            "created": created,
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tok_count,
                "completion_tokens": completion_tok_count,
                "total_tokens": prompt_tok_count + completion_tok_count,
            },
        }

    def reset(self) -> None:
        """Reset context (recreates KV cache). Reapplies any loaded LoRA adapters."""
        self._check_closed()
        self.ctx.reset()
        self._invalidate_prompt_cache()
        self._reapply_lora_adapters()

    def kv_cache_seq_rm(self, seq_id: int = 0, p0: int = -1, p1: int = -1) -> bool:
        """Remove tokens from KV cache for sequence. Returns True if successful.

        Invalidates the prompt-cache mirror — direct seq_rm callers are
        treated as escape hatches, so the mirror's "matches KV exactly"
        invariant cannot be guaranteed. Subsequent generations will fall
        back to a full prompt decode.
        """
        self._check_closed()
        result: bool = self.ctx.kv_cache_seq_rm(seq_id, p0, p1)
        self._invalidate_prompt_cache()
        return result

    def kv_cache_seq_cp(
        self, seq_id_src: int, seq_id_dst: int, p0: int = -1, p1: int = -1
    ) -> None:
        """Copy KV cache from one sequence to another."""
        self._check_closed()
        self.ctx.kv_cache_seq_cp(seq_id_src, seq_id_dst, p0, p1)
        self._invalidate_prompt_cache()

    def kv_cache_seq_keep(self, seq_id: int) -> None:
        """Remove all tokens not belonging to the specified sequence."""
        self._check_closed()
        self.ctx.kv_cache_seq_keep(seq_id)
        self._invalidate_prompt_cache()

    def kv_cache_seq_add(self, seq_id: int, p0: int, p1: int, delta: int) -> None:
        """Add delta to positions in range [p0, p1] for sequence."""
        self._check_closed()
        self.ctx.kv_cache_seq_add(seq_id, p0, p1, delta)
        self._invalidate_prompt_cache()

    def kv_cache_seq_pos_max(self, seq_id: int = 0) -> int:
        """Return max position in KV cache for sequence. -1 if empty."""
        self._check_closed()
        result: int = self.ctx.kv_cache_seq_pos_max(seq_id)
        return result

    def save_state(self, path: str) -> bool:
        """Save KV cache state to file."""
        self._check_closed()
        result: bool = self.ctx.save_state(path)
        return result

    def load_state(self, path: str) -> int:
        """Load KV cache state from file. Returns token count.

        Invalidates the prompt-cache mirror: post-load KV contents are not
        what the mirror represents.
        """
        self._check_closed()
        result: int = self.ctx.load_state(path)
        self._invalidate_prompt_cache()
        return result

    def get_state(self) -> bytes:
        """Get KV cache state as bytes."""
        self._check_closed()
        data: bytes = self.ctx.get_state_data()
        return data

    def set_state(self, data: bytes) -> int:
        """Set KV cache state from bytes. Returns bytes read.

        Invalidates the prompt-cache mirror.
        """
        self._check_closed()
        result: int = self.ctx.set_state_data(data)
        self._invalidate_prompt_cache()
        return result

    def save_seq_state_on_device(self, seq_id: int = 0) -> bytes:
        """Save per-sequence state with the ON_DEVICE flag (llama.cpp 2026-04+).

        Tensor data stays in device buffers (GPU memory) instead of being
        copied to host. The returned ``bytes`` is an opaque handle/header
        that references the device-resident slot — it is **not** a host-
        serializable copy of the KV cache.

        For host-serializable / multi-snapshot state, use ``get_state()``
        (whole-context, returns real bytes).

        Invariant from llama.h:
            "Getting the state for a seq_id with this flag invalidates all
             prior states gotten for that seq_id with this flag."

        Only one on-device snapshot per ``seq_id`` may be live at a time;
        using a stale handle after re-saving the same ``seq_id`` is
        undefined behavior.

        Handle lifetime: the snapshot is also invalidated by any operation
        that clears KV memory (``reset()``, ``kv_cache_clear()``, or
        ``set_state_data()`` / ``load_state()``). Loading a stale handle
        terminates the process via ``ggml_abort`` — the C API performs no
        validation. Treat the handle as a short-lived reference, not a
        persistent snapshot; for durable state use ``get_state()``.

        Args:
            seq_id: Sequence id to snapshot (default 0, the only sequence
                used by single-stream Llama).

        Returns:
            Opaque handle bytes; pass to ``load_seq_state_on_device``.
        """
        self._check_closed()
        result: bytes = self.ctx.save_seq_state_on_device(seq_id)
        return result

    def load_seq_state_on_device(self, data: bytes, dest_seq_id: int = 0) -> int:
        """Restore an on-device snapshot from ``save_seq_state_on_device``.

        The handle is only valid on the same ``Llama`` instance that produced
        it (it references device buffers owned by that context). Using a
        handle from a different instance, after a context reset, or after a
        later save_seq_state_on_device for the same seq_id, is undefined.

        Invalidates the prompt-cache mirror when restoring into seq 0.

        Args:
            data: Opaque handle from ``save_seq_state_on_device``.
            dest_seq_id: Destination sequence id (default 0).

        Returns:
            Bytes read from the handle.
        """
        self._check_closed()
        result: int = self.ctx.load_seq_state_on_device(data, dest_seq_id)
        if dest_seq_id == 0:
            self._invalidate_prompt_cache()
        return result

    def load_lora(self, path: str, scale: float = 1.0) -> Any:
        """Load and apply a LoRA adapter.

        The adapter is stored internally to prevent garbage collection.
        Returns adapter handle for use with remove_lora().
        """
        self._check_closed()
        adapter = _llama.LoraAdapter(self.model, path)
        self._lora_adapters.append(adapter)
        self._lora_configs.append((path, scale))
        self._apply_adapters()
        return adapter

    def remove_lora(self, adapter: Any) -> None:
        """Remove a specific LoRA adapter."""
        if adapter in self._lora_adapters:
            idx = self._lora_adapters.index(adapter)
            self._lora_adapters.pop(idx)
            if idx < len(self._lora_configs):
                self._lora_configs.pop(idx)
            self._apply_adapters()

    def clear_lora(self) -> None:
        """Remove all LoRA adapters."""
        self._check_closed()
        self.ctx.clear_lora()
        self._lora_adapters.clear()
        self._lora_configs.clear()

    def perf(self) -> dict[str, Any]:
        """Get performance metrics (timing and token counts)."""
        self._check_closed()
        return dict(self.ctx.perf())

    def perf_reset(self) -> None:
        """Reset performance counters."""
        self._check_closed()
        self.ctx.perf_reset()

    @property
    def scores(self) -> list[float]:
        """Get raw logits from last decode. Returns empty list if unavailable."""
        self._check_closed()
        try:
            return list(self.ctx.logits())
        except RuntimeError:
            return []

    # Async API (thread-safe wrappers) --------------------------------------
    # Note: Async methods use a lock to ensure thread safety. This means
    # concurrent async calls will serialize (run one at a time), not in parallel.
    # This is a limitation of the underlying llama.cpp context which is not
    # thread-safe. For true parallelism, use multiple Llama instances.

    def _generate_locked(self, prompt: str, **kwargs: Any) -> Any:
        """Thread-safe wrapper for generate().

        For stream=True, the generator is eagerly consumed under the lock
        so that iteration does not happen without synchronization.
        """
        with self._lock:
            result = self.generate(prompt, **kwargs)
            if kwargs.get("stream"):
                return list(result)
            return result

    def _chat_locked(self, messages: Sequence[dict[str, Any]], **kwargs: Any) -> Any:
        """Thread-safe wrapper for create_chat_completion().

        For stream=True, the generator is eagerly consumed under the lock
        so that iteration does not happen without synchronization.
        """
        with self._lock:
            result = self.create_chat_completion(messages, **kwargs)
            if kwargs.get("stream"):
                return list(result)
            return result

    async def generate_async(
        self,
        prompt: str,
        *,
        max_tokens: int = 128,
        sampling: SamplingParams | None = None,
        stop: Sequence[str | int] | None = None,
        echo: bool = False,
        logprobs: int | None = None,
        stream: bool = False,
        seed: int | None = None,
        reset_kv_cache: bool = True,
        cache_prompt: bool = True,
    ) -> str | AsyncIterator[str] | dict[str, Any]:
        """Async version of generate(). Runs in thread pool.

        Note: Concurrent calls serialize due to thread safety lock.
        For true parallelism, use multiple Llama instances.
        """
        if stream:
            if logprobs is not None:
                raise ValueError(
                    "Streaming with logprobs is not supported; "
                    "set stream=False or logprobs=None"
                )
            # True incremental async streaming: bridge the sync generate_stream
            # generator (which runs its own background worker thread for C++
            # generation) into async via a queue. Each chunk is yielded as
            # soon as the worker produces it, matching the README's streaming
            # contract. This replaces the old behavior of eagerly buffering
            # the entire generator under self._lock before yielding.
            loop = asyncio.get_running_loop()
            out_queue: asyncio.Queue[str | None | BaseException] = asyncio.Queue()

            def run_stream() -> None:
                try:
                    for chunk in self.generate_stream(
                        prompt,
                        max_tokens=max_tokens,
                        sampling=sampling,
                        stop=stop,
                        seed=seed,
                        reset_kv_cache=reset_kv_cache,
                        cache_prompt=cache_prompt,
                    ):
                        asyncio.run_coroutine_threadsafe(out_queue.put(chunk), loop)
                    asyncio.run_coroutine_threadsafe(out_queue.put(None), loop)
                except BaseException as exc:  # noqa: BLE001
                    asyncio.run_coroutine_threadsafe(out_queue.put(exc), loop)

            async def async_stream() -> AsyncIterator[str]:
                # Start the pump thread (it holds self._lock for the duration
                # of generate_stream — no other Llama call will run until
                # this stream completes or is cancelled).
                pump = loop.run_in_executor(None, run_stream)
                try:
                    while True:
                        item = await out_queue.get()
                        if item is None:
                            break
                        if isinstance(item, BaseException):
                            raise item
                        yield item
                finally:
                    # Await the pump task so exceptions surface and the
                    # lock-release path in generate_stream runs. Swallow any
                    # exception here because the main yield loop already
                    # re-raised the meaningful error from the queue.
                    with contextlib.suppress(BaseException):
                        await pump

            return async_stream()

        return await asyncio.to_thread(
            self._generate_locked,
            prompt,
            max_tokens=max_tokens,
            sampling=sampling,
            stop=stop,
            echo=echo,
            logprobs=logprobs,
            stream=False,
            seed=seed,
            reset_kv_cache=reset_kv_cache,
            cache_prompt=cache_prompt,
        )

    async def create_chat_completion_async(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        max_tokens: int = 128,
        stream: bool = False,
        stop: Sequence[str | int] | None = None,
        response_format: dict[str, Any] | None = None,
        grammar: LlamaGrammar | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        reset_kv_cache: bool = True,
        cache_prompt: bool = True,
        **sampling_overrides: Any,
    ) -> dict[str, Any] | AsyncIterator[dict[str, Any]]:
        """Async version of create_chat_completion(). Runs in thread pool.

        Note: Concurrent calls serialize due to thread safety lock.
        For true parallelism, use multiple Llama instances.
        """
        if stream:

            async def async_stream() -> AsyncIterator[dict[str, Any]]:
                chunks = await asyncio.to_thread(
                    self._chat_locked,
                    messages,
                    max_tokens=max_tokens,
                    stream=True,
                    stop=stop,
                    response_format=response_format,
                    grammar=grammar,
                    tools=tools,
                    tool_choice=tool_choice,
                    reset_kv_cache=reset_kv_cache,
                    cache_prompt=cache_prompt,
                    **sampling_overrides,
                )
                for chunk in chunks:
                    yield chunk

            return async_stream()

        return await asyncio.to_thread(
            self._chat_locked,
            messages,
            max_tokens=max_tokens,
            stream=False,
            stop=stop,
            response_format=response_format,
            grammar=grammar,
            tools=tools,
            tool_choice=tool_choice,
            reset_kv_cache=reset_kv_cache,
            cache_prompt=cache_prompt,
            **sampling_overrides,
        )

    async def embed_async(self, text: str) -> list[float]:
        """Async version of embed(). Runs in thread pool."""
        return await asyncio.to_thread(self.embed, text)

    async def create_embedding_async(
        self, input: str | list[str], model: str | None = None
    ) -> dict[str, Any]:
        """Async version of create_embedding(). Runs in thread pool."""
        return await asyncio.to_thread(self.create_embedding, input, model)


# Logging helpers ------------------------------------------------------------

_LEVEL_MAP = {
    "none": 0,
    "debug": 1,
    "info": 2,
    "warn": 3,
    "warning": 3,
    "error": 4,
}


def set_log_level(level: str | int) -> None:
    """Set minimum ggml/llama.cpp log level (stderr)."""
    if isinstance(level, str):
        key = level.lower()
        if key not in _LEVEL_MAP:
            raise ValueError(f"Unknown log level '{level}'")
        level_int = _LEVEL_MAP[key]
    else:
        level_int = int(level)
    _llama.set_log_level(level_int)


def disable_logging() -> None:
    """Silence llama.cpp logging completely.

    This affects logging globally across all Llama instances (llama.cpp limitation).

    Note: This is a wrapper around _llama.disable_logging() (imported from C++ bindings).
    Logging can be re-enabled by creating a new Llama instance with verbose=True.
    """
    _llama.disable_logging()


def reset_logging() -> None:
    """Restore default llama.cpp logging callback.

    This re-enables logging after disable_logging() was called.
    Called automatically when creating a Llama instance with verbose=True
    after logging was previously disabled.
    """
    _llama.reset_logging()


def print_system_info() -> str:
    """Return llama.cpp system info (CPU features, build info, etc.)."""
    result: str = _llama.print_system_info()
    return result


# Function calling helpers ---------------------------------------------------


def _format_tools_prompt(tools: list[dict[str, Any]]) -> str:
    """Format tools as a system prompt for function calling."""
    tool_descs = []
    for tool in tools:
        if tool.get("type") == "function":
            func = tool.get("function", {})
            tool_descs.append(
                {
                    "name": func.get("name"),
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                }
            )
    return (
        "You have access to the following functions. "
        "To call a function, respond with a JSON object with 'name' and 'arguments' keys.\n\n"
        f"Functions: {json.dumps(tool_descs, indent=2)}"
    )


_TOOL_CALL_JSON_MAX_CHARS = 1_000_000  # 1MB hard cap on model-emitted JSON


def _parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse function calls from model output."""
    text = text.strip()
    tool_calls: list[dict[str, Any]] = []

    # Defense-in-depth: model output is bounded by max_tokens upstream, but
    # a runaway generation could still produce megabytes of JSON. Cap the
    # parse input so json.loads can't burn unbounded CPU/memory.
    if len(text) > _TOOL_CALL_JSON_MAX_CHARS:
        logging.debug(
            "Tool-call JSON rejected: %d chars exceeds %d-char cap",
            len(text),
            _TOOL_CALL_JSON_MAX_CHARS,
        )
        return tool_calls

    # Try to parse as JSON
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            if "name" in data and data.get("name"):
                # Single function call - validate required fields
                tool_calls.append(
                    {
                        "id": f"call_{_uuid7_hex()[:16]}",
                        "type": "function",
                        "function": {
                            "name": data.get("name"),
                            "arguments": json.dumps(data.get("arguments", {})),
                        },
                    }
                )
            elif "tool_calls" in data:
                # Multiple function calls
                for i, call in enumerate(data["tool_calls"]):
                    if isinstance(call, dict) and call.get("name"):
                        tool_calls.append(
                            {
                                "id": f"call_{_uuid7_hex()[:16]}",
                                "type": "function",
                                "function": {
                                    "name": call.get("name"),
                                    "arguments": json.dumps(call.get("arguments", {})),
                                },
                            }
                        )
                    else:
                        logging.debug(
                            "Skipping invalid tool call at index %d: %s",
                            i,
                            call,
                        )
    except json.JSONDecodeError as e:
        logging.debug("Failed to parse tool calls from response: %s", e)
    except (KeyError, TypeError, ValueError) as e:
        logging.debug("Invalid tool call structure: %s", e)

    return tool_calls


# JSON Grammar for constrained generation
JSON_GRAMMAR = r"""
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws

ws ::= ([ \t\n] ws)?
"""


def _json_schema_to_grammar(schema: dict[str, Any]) -> str:
    """Convert JSON schema to GBNF grammar (simplified).

    Supports: object with typed properties, string, number/integer,
    boolean, null, array (generic).

    Not supported: $ref, anyOf, oneOf, allOf, required field enforcement,
    nested array item types, enum, const, pattern, min/max constraints,
    additionalProperties. For complex schemas, use LlamaGrammar.from_string()
    with a hand-written GBNF grammar or llama.cpp's built-in JSON schema
    support.
    """

    def _type_to_rule(t: str, props: dict[str, Any] | None = None) -> str:
        if t == "string":
            return "string"
        elif t == "number" or t == "integer":
            return "number"
        elif t == "boolean":
            return '("true" | "false")'
        elif t == "null":
            return '"null"'
        elif t == "array":
            return "array"
        elif t == "object" and props:
            # Generate specific object structure
            parts = []
            for k, v in props.items():
                vtype = v.get("type", "string")
                parts.append(
                    f'"{k}" ":" ws {_type_to_rule(vtype, v.get("properties"))}'
                )
            if parts:
                return '"{" ws ' + ' "," ws '.join(parts) + ' "}" ws'
            return "object"
        return "value"

    schema_type = schema.get("type", "object")
    properties = schema.get("properties")

    if schema_type == "object" and properties:
        root_rule = _type_to_rule("object", properties)
        return f"""
root   ::= {root_rule}
value  ::= object | array | string | number | ("true" | "false" | "null") ws
object ::= "{{" ws (string ":" ws value ("," ws string ":" ws value)*)? "}}" ws
array  ::= "[" ws (value ("," ws value)*)? "]" ws
string ::= "\\"" ([^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F]{{4}}))* "\\"" ws
number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws
ws ::= ([ \\t\\n] ws)?
"""
    return JSON_GRAMMAR


def _create_grammar_sampler(model: Any, grammar_str: str, root: str = "root") -> Any:
    """Create a fresh grammar sampler.

    Grammar samplers are stateful (llama_sampler_accept mutates internal state),
    so a fresh instance must be created for each generation.

    Args:
        model: The llama model instance.
        grammar_str: GBNF grammar string.
        root: Root rule name.

    Returns:
        New GrammarSampler instance.
    """
    return _llama.GrammarSampler(model, grammar_str, root)


class LlamaGrammar:
    """Grammar for constrained text generation.

    Eager mode (default): the grammar constrains every sampled token from
    the first one — use for "pure" structured output.

    Lazy mode (``trigger_patterns`` / ``trigger_tokens``): the grammar
    stays inactive until the model emits text matching one of the trigger
    patterns (regex anchored at the start of the generated output) or one
    of the trigger token ids. Useful for tool-calling and mixed
    free-form/structured output where the model should emit a sentinel
    like ``<tool_call>`` before being constrained to a JSON schema.
    """

    def __init__(
        self,
        grammar_str: str,
        root: str = "root",
        *,
        trigger_patterns: list[str] | None = None,
        trigger_tokens: list[int] | None = None,
    ) -> None:
        self._grammar_str = grammar_str
        self._root = root
        self._trigger_patterns: list[str] = list(trigger_patterns or [])
        self._trigger_tokens: list[int] = list(trigger_tokens or [])
        self._sampler: Any | None = None  # Created lazily with model

    @property
    def is_lazy(self) -> bool:
        return bool(self._trigger_patterns or self._trigger_tokens)

    @classmethod
    def from_string(cls, grammar_str: str, root: str = "root") -> LlamaGrammar:
        """Create grammar from GBNF string."""
        return cls(grammar_str, root)

    @classmethod
    def from_json_schema(cls, schema: str | dict[str, Any]) -> LlamaGrammar:
        """Create grammar from JSON schema."""
        schema_dict: dict[str, Any] = (
            json.loads(schema) if isinstance(schema, str) else schema
        )
        grammar_str = _json_schema_to_grammar(schema_dict)
        return cls(grammar_str, "root")

    @classmethod
    def lazy(
        cls,
        grammar_str: str,
        root: str = "root",
        *,
        trigger_patterns: list[str] | None = None,
        trigger_tokens: list[int] | None = None,
    ) -> LlamaGrammar:
        """Create a lazy grammar that activates on a pattern or token trigger.

        At least one of ``trigger_patterns`` or ``trigger_tokens`` must be
        non-empty; otherwise the grammar would never activate.
        """
        if not (trigger_patterns or trigger_tokens):
            raise ValidationError(
                "lazy grammar requires at least one trigger_pattern or trigger_token"
            )
        return cls(
            grammar_str,
            root,
            trigger_patterns=trigger_patterns,
            trigger_tokens=trigger_tokens,
        )

    def _ensure_sampler(self, model: Any) -> None:
        """Create a fresh native sampler for this generation.

        Grammar samplers are stateful, so a new instance is created each time
        to avoid cross-generation state leakage.
        """
        if self.is_lazy:
            self._sampler = _llama.GrammarSampler(
                model,
                self._grammar_str,
                self._root,
                self._trigger_patterns,
                self._trigger_tokens,
            )
        else:
            self._sampler = _llama.GrammarSampler(model, self._grammar_str, self._root)
