"""Pythonic interface around the nanobind llama.cpp bindings."""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import gc
import hashlib
import json
import os
import queue
import sys
import threading
import time
import weakref
from collections import OrderedDict
from collections.abc import AsyncGenerator, Generator, Sequence
from dataclasses import dataclass
from typing import Any

from . import _about  # noqa: F401
from . import (
    _llama,
)

# ---------------------------------------------------------------------------
# Instance tracking for cleanup at exit
# ---------------------------------------------------------------------------
_instances: set[weakref.ref[Any]] = set()
_shutdown_called = False
_cleanup_registered = False
_llama_initialized = False
_cleanup_lock = threading.Lock()

# Grammar sampler cache: (schema_hash, model_id) -> GrammarSampler (LRU)
_grammar_cache: OrderedDict[tuple[str, int], Any] = OrderedDict()
_GRAMMAR_CACHE_MAX = 32

# Configuration constants
_ALL_GPU_LAYERS_SENTINEL = 1_000_000  # Special value meaning "offload all layers to GPU"
_MAX_PROMPT_LENGTH = 10_000_000  # Maximum prompt length in characters (10MB limit)
_MAX_STOP_SEQUENCES = 20  # Maximum number of stop sequences allowed
_MAX_STOP_SEQUENCE_LENGTH = 500  # Maximum length of each stop sequence in characters


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
    with _cleanup_lock:
        if _llama_initialized:
            return
        _llama_initialized = True
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


@dataclass
class SamplingParams:
    """Sampling configuration mirroring llama-cpp-python defaults."""

    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.0
    min_keep: int = 1
    repeat_penalty: float = 1.1
    repeat_last_n: int = 64
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate sampling parameters."""
        if self.temperature < 0:
            raise ValidationError("temperature must be non-negative")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValidationError("top_p must be between 0.0 and 1.0")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValidationError("min_p must be between 0.0 and 1.0")
        if self.top_k < 0:
            raise ValidationError("top_k must be non-negative (0 = disabled)")
        if self.repeat_penalty < 0:
            raise ValidationError("repeat_penalty must be non-negative")
        if self.min_keep < 1:
            raise ValidationError("min_keep must be at least 1")

    def to_native(self) -> _llama.SamplerParams:
        native = _llama.SamplerParams()
        native.top_k = self.top_k
        native.top_p = float(self.top_p)
        native.min_p = float(self.min_p)
        native.min_keep = int(self.min_keep)
        native.temp = float(self.temperature)
        native.penalty_last_n = int(self.repeat_last_n)
        native.repeat_penalty = float(self.repeat_penalty)
        native.freq_penalty = float(self.frequency_penalty)
        native.presence_penalty = float(self.presence_penalty)
        native.seed = -1 if self.seed is None else int(self.seed)
        return native


@dataclass
class LlamaConfig:
    model_path: str
    n_ctx: int = 4096
    n_batch: int = 2048
    n_ubatch: int = 512
    n_seq_max: int = 1  # Max parallel sequences (1 = single sequence, simplest)
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
    offload_kqv: bool = True
    embeddings: bool = False
    rope_freq_base: float = 0.0
    rope_freq_scale: float = 0.0
    add_bos: bool = True
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
    """

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
        self._lora_configs: list[tuple[str, float]] = (
            []
        )  # (path, scale) for reapplication

        # Apply verbose setting
        # WARNING: This affects logging globally, not per-instance.
        # In multi-instance apps, the last instance's verbose setting wins.
        if not cfg.verbose:
            import warnings

            warnings.warn(
                "verbose=False affects logging globally for all Llama instances. "
                "This is a limitation of the underlying llama.cpp library. "
                "In multi-instance applications, the last instance's setting applies to all.",
                RuntimeWarning,
                stacklevel=2,
            )
            disable_logging()

        # Apply seed to default sampling if specified
        if cfg.seed >= 0 and self.sampling.seed is None:
            self.sampling = SamplingParams(
                **{**self.sampling.__dict__, "seed": cfg.seed}
            )

        model_params = _llama.ModelParams()
        # llama.cpp treats negative n_gpu_layers as "all layers" in the CLI wrapper,
        # but the low-level API expects a non-negative count. Translate -1 to a large
        # sentinel so users can keep using -1 to mean "full offload".
        gpu_layers = cfg.n_gpu_layers if cfg.n_gpu_layers >= 0 else _ALL_GPU_LAYERS_SENTINEL
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
        ctx_params.n_threads = int(cfg.n_threads or os.cpu_count() or 1)
        ctx_params.n_threads_batch = int(cfg.n_threads_batch or ctx_params.n_threads)
        ctx_params.flash_attn_type = int(cfg.flash_attn)
        ctx_params.offload_kqv = bool(cfg.offload_kqv)
        ctx_params.embeddings = bool(cfg.embeddings)
        ctx_params.rope_freq_base = float(cfg.rope_freq_base)
        ctx_params.rope_freq_scale = float(cfg.rope_freq_scale)

        try:
            self.model = _llama.Model(cfg.model_path, model_params)
        except RuntimeError as e:
            raise ModelLoadError(f"Failed to load model: {cfg.model_path}") from e

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

    def __del__(self) -> None:
        # Avoid calling close() during interpreter shutdown to prevent segfault
        # when C++ destructors run after Python internals are torn down
        if not sys.is_finalizing() and getattr(self, "_closed", True) is False:
            self.close()

    def _check_closed(self) -> None:
        """Raise error if instance has been closed."""
        if self._closed:
            raise LlamaError("Llama instance has been closed")

    def close(self) -> None:
        """Release model and context resources."""
        if getattr(self, "_closed", True):
            return
        self._closed = True
        # Remove from instance tracking
        if hasattr(self, "_ref"):
            _instances.discard(self._ref)
        if hasattr(self, "_lora_adapters"):
            self._lora_adapters.clear()
        # Explicitly free context before model (C++ dependency)
        if getattr(self, "ctx", None) is not None:
            self.ctx.close()
            self.ctx = None
        if getattr(self, "model", None) is not None:
            self.model.close()
            self.model = None
        # Force GC to collect any reference cycles while interpreter is safe
        gc.collect()

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
                add_special=self.config.add_bos if add_special is None else add_special,
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
        result: str = self.model.detokenize(
            list(tokens), remove_special=remove_special, unparse_special=unparse_special
        )
        return result

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

    def token_bos(self) -> int:
        """Return BOS token id."""
        result: int = self.model.bos()
        return result

    def token_eos(self) -> int:
        """Return EOS token id."""
        result: int = self.model.eos()
        return result

    def token_eot(self) -> int:
        """Return EOT token id."""
        result: int = self.model.eot()
        return result

    def n_ctx(self) -> int:
        """Return context size."""
        result: int = self.ctx.n_ctx()
        return result

    def n_vocab(self) -> int:
        """Return vocabulary size."""
        result: int = self.model.n_vocab()
        return result

    def n_embd(self) -> int:
        """Return embedding dimension."""
        result: int = self.model.n_embd()
        return result

    def model_size(self) -> int:
        """Return total size of all tensors in bytes."""
        result: int = self.model.model_size()
        return result

    def n_params(self) -> int:
        """Return total number of parameters."""
        result: int = self.model.n_params()
        return result

    def n_layer(self) -> int:
        """Return number of layers."""
        result: int = self.model.n_layer()
        return result

    def get_chat_template(self, name: str = "") -> str:
        """Return model's chat template. Empty string if not available."""
        result: str = self.model.chat_template(name)
        return result

    def token_to_piece(self, token: int) -> str:
        """Return the text representation of a token."""
        result: str = self.model.token_to_piece(token)
        return result

    @property
    def metadata(self) -> dict[str, str]:
        """Return model metadata as a dictionary (cached)."""
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

    def _reapply_lora_adapters(self) -> None:
        """Reapply all loaded LoRA adapters after context reset."""
        if self._lora_configs:
            self._lora_adapters.clear()
            for path, scale in self._lora_configs:
                adapter = _llama.LoraAdapter(self.model, path)
                self.ctx.set_lora(adapter, scale)
                self._lora_adapters.append(adapter)

    def embed(self, text: str) -> list[float]:
        """Get embedding for text. Clears KV cache."""
        if not self.config.embeddings:
            raise ValidationError(
                "Embeddings not enabled. Set embeddings=True in LlamaConfig."
            )
        self.ctx.kv_cache_clear()
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
        """
        if not self.config.embeddings:
            raise ValidationError(
                "Embeddings not enabled. Set embeddings=True in LlamaConfig."
            )

        inputs = [input] if isinstance(input, str) else list(input)

        data = []
        total_tokens = 0
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
        if overrides:
            params_obj = SamplingParams(
                **{
                    **params_obj.__dict__,
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
        except Exception:
            # Fallback to simple format
            parts = [f"{role}: {content}" for role, content in msg_pairs]
            parts.append("assistant:")
            return "\n".join(parts)

    def _prepare_chat(
        self, messages: Sequence[dict[str, Any]]
    ) -> tuple[str, list[int], int]:
        """Prepare chat for generation: format and tokenize once.

        Args:
            messages: List of message dicts with 'role' and 'content'.

        Returns:
            Tuple of (formatted_prompt, prompt_tokens, token_count).
        """
        prompt = self._format_chat_messages(messages)
        tokens = self.tokenize(prompt, add_special=self.config.add_bos)
        return prompt, tokens, len(tokens)

    def _generate_from_tokens(
        self,
        prompt_tokens: list[int],
        *,
        max_tokens: int,
        sampler: _llama.SamplerChain,
        stop_sequences: list[list[int]] | None = None,
        grammar: Any | None = None,
        reset_kv_cache: bool = True,
    ) -> list[int]:
        """Internal generation from pre-tokenized input.

        Args:
            prompt_tokens: Pre-tokenized prompt.
            max_tokens: Maximum tokens to generate.
            sampler: SamplerChain instance.
            stop_sequences: Optional multi-token stop sequences.
            grammar: Optional GrammarSampler for constrained generation.
            reset_kv_cache: Whether to clear KV cache before generation.

        Returns:
            List of generated token IDs.
        """
        if reset_kv_cache:
            self.ctx.kv_cache_clear()

        eos = self.model.eos()
        stop_seqs = stop_sequences or []

        if grammar is not None:
            return list(
                _llama.generate_tokens_grammar_multi_stop(
                    self.ctx,
                    sampler,
                    grammar,
                    prompt_tokens,
                    int(max_tokens),
                    bool(self.config.add_bos),
                    eos,
                    stop_seqs,
                )
            )
        elif stop_seqs:
            return list(
                _llama.generate_tokens_multi_stop(
                    self.ctx,
                    sampler,
                    prompt_tokens,
                    int(max_tokens),
                    bool(self.config.add_bos),
                    eos,
                    stop_seqs,
                )
            )
        else:
            return list(
                _llama.generate_tokens(
                    self.ctx,
                    sampler,
                    prompt_tokens,
                    int(max_tokens),
                    bool(self.config.add_bos),
                    eos,
                    [],
                )
            )

    def generate_stream(
        self,
        prompt: str,
        *,
        max_tokens: int = 128,
        sampling: SamplingParams | None = None,
        stop: Sequence[str | int] | None = None,
        seed: int | None = None,
        reset_kv_cache: bool = True,
    ) -> Generator[str, None, None]:
        """True streaming generation - yields text as tokens are decoded.

        Unlike generate(..., stream=True) which buffers all tokens first,
        this yields each token immediately as it's generated in a background thread.

        Args:
            prompt: Input prompt string.
            max_tokens: Maximum tokens to generate.
            sampling: Optional sampling parameters.
            stop: Optional stop sequences.
            seed: Optional RNG seed.
            reset_kv_cache: Clear KV cache before generation (default True).

        Yields:
            Text chunks as they're generated.
        """
        self._check_closed()
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValidationError("prompt must be a non-empty string")
        if max_tokens < 1:
            raise ValidationError("max_tokens must be positive")

        sampler_params = sampling or self.sampling
        if seed is not None:
            sampler_params = SamplingParams(**{**sampler_params.__dict__, "seed": seed})
        sampler = self._build_sampler(sampler_params)

        if reset_kv_cache:
            self.ctx.kv_cache_clear()

        prompt_tokens = self.tokenize(prompt, add_special=self.config.add_bos)

        stop_sequences: list[list[int]] = []
        if stop:
            for item in stop:
                if isinstance(item, str):
                    tks = self.tokenize(item, add_special=False, parse_special=False)
                    if tks:
                        stop_sequences.append([int(t) for t in tks])
                else:
                    stop_sequences.append([int(item)])

        eos = self.model.eos()

        # Use queue for true streaming from background thread
        token_queue: queue.Queue[int | None | Exception] = queue.Queue()

        def worker() -> None:
            """Background thread that generates tokens and puts them in queue."""
            try:
                def on_token(token: int) -> bool:
                    token_queue.put(token)
                    return True

                _llama.generate_tokens_streaming(
                    self.ctx,
                    sampler,
                    prompt_tokens,
                    int(max_tokens),
                    bool(self.config.add_bos),
                    eos,
                    stop_sequences,
                    on_token,
                )
                token_queue.put(None)  # Sentinel: generation complete
            except Exception as e:
                token_queue.put(e)  # Propagate exception to main thread

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        # Yield tokens as they arrive from the background thread
        try:
            while True:
                item = token_queue.get()
                if item is None:
                    break  # Generation complete
                if isinstance(item, Exception):
                    raise item  # Propagate exception from worker thread
                # item is a token
                text = self.detokenize([item], remove_special=True, unparse_special=True)
                yield text
        finally:
            # Ensure thread completes even if generator is closed early
            thread.join(timeout=1.0)

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
    ) -> str | Generator[str, None, None] | dict[str, Any]:
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
            raise ValidationError(f"prompt exceeds maximum length ({_MAX_PROMPT_LENGTH} chars)")
        if stop:
            if len(stop) > _MAX_STOP_SEQUENCES:
                raise ValidationError(f"too many stop sequences (max {_MAX_STOP_SEQUENCES})")
            for item in stop:
                if isinstance(item, str) and len(item) > _MAX_STOP_SEQUENCE_LENGTH:
                    raise ValidationError(f"stop sequence too long (max {_MAX_STOP_SEQUENCE_LENGTH} chars)")

        sampler_params = sampling or self.sampling
        if seed is not None:
            sampler_params = SamplingParams(**{**sampler_params.__dict__, "seed": seed})
        sampler = self._build_sampler(sampler_params)
        # Optionally clear KV cache (default True for fresh generation)
        if reset_kv_cache:
            self.ctx.kv_cache_clear()
        prompt_tokens = self.tokenize(prompt, add_special=self.config.add_bos)

        # prepare stop sequences
        stop_sequences: list[list[int]] = []
        if stop:
            for item in stop:
                if isinstance(item, str):
                    tks = self.tokenize(item, add_special=False, parse_special=False)
                    if tks:
                        stop_sequences.append([int(t) for t in tks])
                else:
                    stop_sequences.append([int(item)])

        eos = self.model.eos()

        # Only use expensive details path when actually needed (echo or logprobs)
        need_details = echo or logprobs is not None

        if stream and logprobs is not None:
            raise ValueError(
                "Streaming with logprobs is not supported; set stream=False or logprobs=None"
            )

        if need_details:
            token_probs = _llama.generate_tokens_with_details(
                self.ctx,
                sampler,
                prompt_tokens,
                int(max_tokens),
                bool(self.config.add_bos),
                eos,
                stop_sequences,
                int(logprobs or 0),
                bool(echo),
            )
            output_tokens = [tp.token for tp in token_probs]
        elif stop_sequences:
            # Use fast C++ multi-stop helper (no O(n_vocab) per-token overhead)
            generated = _llama.generate_tokens_multi_stop(
                self.ctx,
                sampler,
                prompt_tokens,
                int(max_tokens),
                bool(self.config.add_bos),
                eos,
                stop_sequences,
            )
            output_tokens = list(generated)
        else:
            # No stop sequences - use simplest path
            generated = _llama.generate_tokens(
                self.ctx,
                sampler,
                prompt_tokens,
                int(max_tokens),
                bool(self.config.add_bos),
                eos,
                [],
            )
            output_tokens = list(generated)

        if logprobs is not None:
            text = self.detokenize(
                output_tokens, remove_special=True, unparse_special=False
            )
            return {
                "text": text,
                "tokens": output_tokens,
                "token_probs": token_probs,
            }

        def stream_chunks() -> Generator[str, None, None]:
            for tok in output_tokens:
                yield self.detokenize([tok], remove_special=True, unparse_special=True)

        if stream:
            return stream_chunks()

        text = self.detokenize(
            output_tokens, remove_special=True, unparse_special=False
        )
        return text

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
    ) -> dict[str, Any] | Generator[dict[str, Any], None, None]:
        """Generate completion with OpenAI-compatible response format.

        Note: Streaming yields chunks after full generation (not true streaming).
        """
        prompt_tokens = self.tokenize(prompt, add_special=self.config.add_bos)
        prompt_tok_count = len(prompt_tokens)

        if stream:

            def stream_chunks() -> Generator[dict[str, Any], None, None]:
                created = int(time.time())
                for chunk in self.generate(
                    prompt,
                    max_tokens=max_tokens,
                    stop=stop,
                    echo=echo,
                    stream=True,
                    **kwargs,
                ):
                    yield {
                        "id": f"cmpl-{created}",
                        "object": "text_completion",
                        "created": created,
                        "choices": [{"text": chunk, "index": 0, "finish_reason": None}],
                    }
                yield {
                    "id": f"cmpl-{created}",
                    "object": "text_completion",
                    "created": created,
                    "choices": [{"text": "", "index": 0, "finish_reason": "stop"}],
                }

            return stream_chunks()

        text = self.generate(
            prompt, max_tokens=max_tokens, stop=stop, echo=echo, stream=False, **kwargs
        )
        created = int(time.time())
        # Get completion token count from KV cache position instead of re-tokenizing
        completion_tokens = (
            max(0, self.kv_cache_seq_pos_max() - prompt_tok_count + 1)
            if isinstance(text, str)
            else 0
        )
        return {
            "id": f"cmpl-{created}",
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
        # Handle streaming case
        if hasattr(result, "__iter__") and not isinstance(result, dict):
            chunks = list(result)
            text = "".join(c["choices"][0]["text"] for c in chunks)
            return {
                "id": chunks[0]["id"] if chunks else f"cmpl-{int(time.time())}",
                "object": "text_completion",
                "created": chunks[0]["created"] if chunks else int(time.time()),
                "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
            }
        return dict(result)  # type: ignore[arg-type]

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
        **sampling_overrides: Any,
    ) -> dict[str, Any] | Generator[dict[str, Any], None, None]:
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

        # Tokenize once via _prepare_chat
        _, prompt_tokens, _ = self._prepare_chat(effective_messages)
        sampler = self._build_sampler(None, **sampling_overrides)

        stop_sequences: list[list[int]] = []
        if stop:
            for item in stop:
                if isinstance(item, str):
                    tks = self.tokenize(item, add_special=False, parse_special=False)
                    if tks:
                        stop_sequences.append([int(t) for t in tks])
                else:
                    stop_sequences.append([int(item)])

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
                use_grammar = _get_cached_grammar_sampler(
                    self.model, grammar_str, "root"
                )

        # Use unified generation path
        generated = self._generate_from_tokens(
            prompt_tokens,
            max_tokens=max_tokens,
            sampler=sampler,
            stop_sequences=stop_sequences,
            grammar=use_grammar,
            reset_kv_cache=reset_kv_cache,
        )

        created = int(time.time())
        model_id = os.path.basename(self.config.model_path)

        if stream:

            def stream_chunks() -> Generator[dict[str, Any], None, None]:
                for tok in generated:
                    text_piece = self.detokenize(
                        [tok], remove_special=True, unparse_special=True
                    )
                    yield {
                        "id": f"chatcmpl-{created}",
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
                yield {
                    "id": f"chatcmpl-{created}",
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
            "id": f"chatcmpl-{created}",
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
        self.ctx.reset()
        self._reapply_lora_adapters()

    def kv_cache_seq_rm(self, seq_id: int = 0, p0: int = -1, p1: int = -1) -> bool:
        """Remove tokens from KV cache for sequence. Returns True if successful."""
        result: bool = self.ctx.kv_cache_seq_rm(seq_id, p0, p1)
        return result

    def kv_cache_seq_cp(
        self, seq_id_src: int, seq_id_dst: int, p0: int = -1, p1: int = -1
    ) -> None:
        """Copy KV cache from one sequence to another."""
        self.ctx.kv_cache_seq_cp(seq_id_src, seq_id_dst, p0, p1)

    def kv_cache_seq_keep(self, seq_id: int) -> None:
        """Remove all tokens not belonging to the specified sequence."""
        self.ctx.kv_cache_seq_keep(seq_id)

    def kv_cache_seq_add(self, seq_id: int, p0: int, p1: int, delta: int) -> None:
        """Add delta to positions in range [p0, p1] for sequence."""
        self.ctx.kv_cache_seq_add(seq_id, p0, p1, delta)

    def kv_cache_seq_pos_max(self, seq_id: int = 0) -> int:
        """Return max position in KV cache for sequence. -1 if empty."""
        result: int = self.ctx.kv_cache_seq_pos_max(seq_id)
        return result

    def save_state(self, path: str) -> bool:
        """Save KV cache state to file."""
        result: bool = self.ctx.save_state(path)
        return result

    def load_state(self, path: str) -> int:
        """Load KV cache state from file. Returns token count."""
        result: int = self.ctx.load_state(path)
        return result

    def get_state(self) -> bytes:
        """Get KV cache state as bytes."""
        return bytes(self.ctx.get_state_data())

    def set_state(self, data: bytes) -> int:
        """Set KV cache state from bytes. Returns bytes read."""
        result: int = self.ctx.set_state_data(list(data))
        return result

    def load_lora(self, path: str, scale: float = 1.0) -> Any:
        """Load and apply a LoRA adapter.

        The adapter is stored internally to prevent garbage collection.
        Returns adapter handle for use with remove_lora().
        """
        adapter = _llama.LoraAdapter(self.model, path)
        self.ctx.set_lora(adapter, scale)
        self._lora_adapters.append(adapter)
        self._lora_configs.append((path, scale))
        return adapter

    def remove_lora(self, adapter: Any) -> None:
        """Remove a specific LoRA adapter."""
        self.ctx.remove_lora(adapter)
        if adapter in self._lora_adapters:
            idx = self._lora_adapters.index(adapter)
            self._lora_adapters.remove(adapter)
            if idx < len(self._lora_configs):
                self._lora_configs.pop(idx)

    def clear_lora(self) -> None:
        """Remove all LoRA adapters."""
        self.ctx.clear_lora()
        self._lora_adapters.clear()
        self._lora_configs.clear()

    def perf(self) -> dict[str, Any]:
        """Get performance metrics (timing and token counts)."""
        return dict(self.ctx.perf())

    def perf_reset(self) -> None:
        """Reset performance counters."""
        self.ctx.perf_reset()

    @property
    def scores(self) -> list[float]:
        """Get raw logits from last decode. Returns empty list if unavailable."""
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
        """Thread-safe wrapper for generate()."""
        with self._lock:
            return self.generate(prompt, **kwargs)

    def _chat_locked(self, messages: Sequence[dict[str, Any]], **kwargs: Any) -> Any:
        """Thread-safe wrapper for create_chat_completion()."""
        with self._lock:
            return self.create_chat_completion(messages, **kwargs)

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
    ) -> str | AsyncGenerator[str, None] | dict[str, Any]:
        """Async version of generate(). Runs in thread pool.

        Note: Concurrent calls serialize due to thread safety lock.
        For true parallelism, use multiple Llama instances.
        """
        if stream:

            async def async_stream() -> AsyncGenerator[str, None]:
                gen = await asyncio.to_thread(
                    self._generate_locked,
                    prompt,
                    max_tokens=max_tokens,
                    sampling=sampling,
                    stop=stop,
                    echo=echo,
                    logprobs=logprobs,
                    stream=True,
                    seed=seed,
                )
                for chunk in gen:
                    yield chunk

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
        **sampling_overrides: Any,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """Async version of create_chat_completion(). Runs in thread pool.

        Note: Concurrent calls serialize due to thread safety lock.
        For true parallelism, use multiple Llama instances.
        """
        if stream:

            async def async_stream() -> AsyncGenerator[dict[str, Any], None]:
                gen = await asyncio.to_thread(
                    self._chat_locked,
                    messages,
                    max_tokens=max_tokens,
                    stream=True,
                    stop=stop,
                    response_format=response_format,
                    grammar=grammar,
                    tools=tools,
                    tool_choice=tool_choice,
                    **sampling_overrides,
                )
                for chunk in gen:
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
    """Silence llama.cpp logging completely."""
    _llama.disable_logging()


def reset_logging() -> None:
    """Restore default llama.cpp logging callback."""
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


def _parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse function calls from model output."""
    import logging

    text = text.strip()
    tool_calls = []

    # Try to parse as JSON
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            if "name" in data and data.get("name"):
                # Single function call - validate required fields
                tool_calls.append(
                    {
                        "id": f"call_{hash(text) & 0xFFFFFFFF:08x}",
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
                                "id": f"call_{i}_{hash(text) & 0xFFFFFFFF:08x}",
                                "type": "function",
                                "function": {
                                    "name": call.get("name"),
                                    "arguments": json.dumps(call.get("arguments", {})),
                                },
                            }
                        )
                    else:
                        logging.debug(f"Skipping invalid tool call at index {i}: {call}")
    except json.JSONDecodeError as e:
        logging.debug(f"Failed to parse tool calls from response: {e}")
    except Exception as e:
        logging.warning(f"Unexpected error parsing tool calls: {e}")

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
    """Convert JSON schema to GBNF grammar (simplified)."""

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


def _get_cached_grammar_sampler(
    model: Any, grammar_str: str, root: str = "root"
) -> Any:
    """Get or create a cached grammar sampler (LRU cache).

    Args:
        model: The llama model instance.
        grammar_str: GBNF grammar string.
        root: Root rule name.

    Returns:
        GrammarSampler instance (cached if possible).
    """
    # Hash grammar string for cache key
    grammar_hash = hashlib.md5(grammar_str.encode()).hexdigest()[:16]
    model_id = id(model)
    cache_key = (grammar_hash, model_id)

    if cache_key in _grammar_cache:
        # LRU: move to end on access
        _grammar_cache.move_to_end(cache_key)
        return _grammar_cache[cache_key]

    # Evict least recently used if cache full (LRU)
    if len(_grammar_cache) >= _GRAMMAR_CACHE_MAX:
        _grammar_cache.popitem(last=False)  # Remove oldest (LRU)

    sampler = _llama.GrammarSampler(model, grammar_str, root)
    _grammar_cache[cache_key] = sampler
    return sampler


class LlamaGrammar:
    """Grammar for constrained text generation."""

    def __init__(self, grammar_str: str, root: str = "root") -> None:
        self._grammar_str = grammar_str
        self._root = root
        self._sampler: Any | None = None  # Created lazily with model

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

    def _ensure_sampler(self, model: Any) -> None:
        """Create native sampler if not already created."""
        if self._sampler is None:
            self._sampler = _llama.GrammarSampler(model, self._grammar_str, self._root)
