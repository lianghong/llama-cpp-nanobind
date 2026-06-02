"""Pool manager for parallel inference with multiple Llama instances.

This module provides utilities for true parallel processing by managing multiple
Llama instances. Each instance can process requests independently, enabling
concurrent inference on multi-core CPUs or GPUs with sufficient memory.

Example:
    >>> from llama_cpp import LlamaPool
    >>> import asyncio
    >>>
    >>> async def main():
    ...     async with LlamaPool("model.gguf", pool_size=4) as pool:
    ...         results = await pool.generate_batch([
    ...             "What is AI?",
    ...             "Explain quantum computing",
    ...             "Tell me about Python",
    ...         ])
    ...         print(results)
    >>>
    >>> asyncio.run(main())
"""

import asyncio
from dataclasses import replace as dc_replace
import logging
import threading
from typing import Any, cast

from .llama import Llama
from .llama import LlamaConfig
from .llama import SamplingParams


_POOL_CLOSED = object()  # Sentinel to unblock waiters on shutdown


class LlamaPool:
    """Pool of Llama instances for true parallel inference.

    Creates multiple independent Llama instances that can process requests
    concurrently. Each instance loads the model separately, so GPU memory
    requirements scale with pool_size.

    Thread Safety:
        This class is async-safe. Multiple coroutines can call methods
        concurrently, and requests are distributed across available instances.

    GPU Memory Planning:
        - Each instance loads the full model
        - Required VRAM ≈ model_size × pool_size
        - Example: 8GB model with pool_size=3 needs ~24GB VRAM
        - Adjust pool_size based on available GPU memory

    Example:
        >>> # For a 7B model (~8GB), use pool_size=2 on 24GB GPU
        >>> pool = LlamaPool("model.gguf", pool_size=2)
        >>> result = await pool.generate("Hello world")
        >>> pool.close()

    Attributes:
        model_path: Path to the model file.
        pool_size: Number of parallel worker instances.
        config: Configuration shared by all instances.
        instances: List of Llama instances in the pool.
    """

    def __init__(
        self,
        model_path: str,
        pool_size: int = 4,
        config: LlamaConfig | None = None,
        warmup: bool = False,
    ) -> None:
        """Initialize pool with multiple Llama instances.

        Args:
            model_path: Path to model file (.gguf).
            pool_size: Number of parallel workers. Each worker is an independent
                Llama instance that can process requests concurrently.
            config: Optional configuration for all instances. If None, uses
                defaults from LlamaConfig.
            warmup: If True, run a dummy inference on each instance after
                initialization to pre-load GPU caches and ensure CUDA kernels
                are compiled. This adds initialization time but ensures first
                real request has consistent latency. Recommended for production
                services with strict SLA requirements.

        Raises:
            ValueError: If pool_size < 1.
            ModelLoadError: If model fails to load.

        Example:
            >>> config = LlamaConfig(
            ...     model_path="model.gguf",
            ...     n_gpu_layers=-1,  # Full GPU offload
            ...     n_ctx=4096,
            ... )
            >>> pool = LlamaPool("model.gguf", pool_size=3, config=config)
        """
        if pool_size < 1:
            raise ValueError(f"pool_size must be >= 1, got {pool_size}")

        self.model_path = model_path
        self.pool_size = pool_size
        self.config = config or LlamaConfig(model_path=model_path)
        self._closed = False

        # Create worker instances
        logging.info("Initializing LlamaPool with %d instances...", pool_size)
        self.instances: list[Llama] = []
        for i in range(pool_size):
            logging.debug("Loading instance %d/%d...", i + 1, pool_size)
            # Give each instance its own LlamaConfig copy. Llama.__init__ stores
            # the config by reference (self.config = cfg) and LlamaConfig is a
            # mutable (non-frozen) dataclass — sharing one object across the pool
            # would couple all instances if any per-instance state were ever
            # written back to self.config. dc_replace() with no kwargs is a cheap
            # shallow copy that keeps them independent.
            instance = Llama(model_path, config=dc_replace(self.config))
            self.instances.append(instance)
        logging.info("LlamaPool initialized with %d instances", pool_size)

        # Warmup phase: run dummy inference to prime GPU caches
        if warmup:
            logging.info("Running warmup phase to pre-load GPU caches...")
            self._warmup_instances()
            logging.info("Warmup phase complete")

        # Queue of available instances - ensures each instance is used by
        # at most one request at a time (Llama is not thread-safe)
        # Created lazily on first async use to avoid event loop binding issues
        self._available: asyncio.Queue[Llama | object] | None = None
        self._queue_initialized = False
        self._queue_init_lock = threading.Lock()
        # Event loop the queue is bound to. asyncio.Queue is not portable
        # across event loops — subsequent checkouts from a different loop
        # would silently hang or raise. We record the loop at first-init
        # time and raise a clear error on any mismatch.
        self._bound_loop: asyncio.AbstractEventLoop | None = None
        # Serializes close()/close_graceful() so a concurrent second call
        # returns a no-op instead of re-entering the close path.
        self._close_lock = threading.Lock()

    def _ensure_queue_initialized(self) -> None:
        """Lazily initialize asyncio.Queue on first async use.

        This avoids creating the queue outside an event loop, which causes
        'RuntimeError: no running event loop' in Python 3.10+.

        Also records the binding event loop so we can detect cross-loop
        misuse (an asyncio.Queue belongs to one loop; using it from another
        would silently hang or raise deep in asyncio internals).
        """
        current_loop = asyncio.get_running_loop()
        if self._queue_initialized:
            if self._bound_loop is not current_loop:
                raise RuntimeError(
                    "LlamaPool was first used on a different event loop. "
                    "asyncio.Queue is bound to one loop; reusing this pool "
                    "from another loop would hang. Create a new LlamaPool "
                    "per event loop, or reuse the pool within a single loop."
                )
            return
        with self._queue_init_lock:
            if self._queue_initialized:
                if self._bound_loop is not current_loop:
                    raise RuntimeError(
                        "LlamaPool was first used on a different event loop. "
                        "asyncio.Queue is bound to one loop; reusing this "
                        "pool from another loop would hang. Create a new "
                        "LlamaPool per event loop, or reuse the pool within "
                        "a single loop."
                    )
                return
            queue: asyncio.Queue[Llama | object] = asyncio.Queue()
            for instance in self.instances:
                queue.put_nowait(instance)
            self._available = queue
            self._bound_loop = current_loop
            self._queue_initialized = True

    def _warmup_instances(self) -> None:
        """Run dummy inference on each instance to pre-load GPU caches.

        This performs a minimal inference (2-3 tokens) on each instance to:
        - Trigger CUDA kernel compilation/optimization
        - Initialize GPU memory pools
        - Prime various GPU caches
        - Ensure consistent latency for first real request

        The warmup prompt is short to minimize initialization time while
        still triggering all GPU initialization paths.
        """
        warmup_prompt = "Hi"  # Short prompt for minimal warmup time
        warmup_tokens = 3  # Just enough to trigger GPU paths

        for i, instance in enumerate(self.instances):
            try:
                logging.debug("Warming up instance %d/%d...", i + 1, self.pool_size)
                # Run minimal inference to trigger GPU initialization
                instance.generate(
                    warmup_prompt,
                    max_tokens=warmup_tokens,
                    reset_kv_cache=True,
                )
                # Clear KV cache after warmup to start fresh, keeping the
                # Python-side prompt-cache mirror and state-epoch bookkeeping
                # in sync (bypassing the wrapper would leave a stale mirror
                # that prefix-reuse could trust against a now-empty KV).
                instance.kv_cache_clear()
            except (RuntimeError, ValueError) as e:
                # Expected errors from model inference - warmup is optional optimization
                logging.warning(
                    "Warmup failed for instance %d (non-fatal): %s", i + 1, e
                )
            except Exception as e:
                # Unexpected errors during warmup
                logging.warning(
                    "Unexpected error during warmup for instance %d (non-fatal): %s",
                    i + 1,
                    e,
                )

    async def _checkout_instance(self, timeout: float | None = None) -> Llama:
        """Check out an available instance from the pool.

        Args:
            timeout: Maximum seconds to wait for an available instance.
                None means wait indefinitely.

        Raises:
            RuntimeError: If pool is closed.
            TimeoutError: If timeout expires before an instance is available.
        """
        self._ensure_queue_initialized()
        assert self._available is not None  # Type narrowing after init

        if self._closed:
            raise RuntimeError("LlamaPool is closed")
        try:
            if timeout is not None:
                item = await asyncio.wait_for(self._available.get(), timeout=timeout)
            else:
                item = await self._available.get()
        except TimeoutError:
            # Re-check closed state before propagating timeout
            if self._closed:
                raise RuntimeError("LlamaPool is closed") from None
            raise  # Legitimate timeout - pool busy
        if item is _POOL_CLOSED:
            # Re-inject sentinel so the next blocked waiter also wakes up
            self._available.put_nowait(_POOL_CLOSED)
            raise RuntimeError("LlamaPool is closed")
        return cast(Llama, item)

    def _return_instance(self, instance: Llama) -> None:
        """Return an instance to the pool after use."""
        assert self._available is not None  # Must be initialized after checkout
        self._available.put_nowait(instance)

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 128,
        sampling: SamplingParams | None = None,
        stop: list[str] | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate text using next available instance.

        Instances are checked out exclusively - each request gets its own
        instance, preventing concurrent access to non-thread-safe Llama objects.
        Multiple calls run in parallel up to pool_size.

        Args:
            prompt: Input prompt string.
            max_tokens: Maximum tokens to generate.
            sampling: Optional sampling parameters.
            stop: Optional stop sequences.
            timeout: Maximum seconds to wait for an available instance.
                None means wait indefinitely. Raises TimeoutError on expiry.
            **kwargs: Additional arguments passed to Llama.generate_async().

        Returns:
            Generated text response.

        Example:
            >>> results = await asyncio.gather(
            ...     pool.generate("Query 1"),
            ...     pool.generate("Query 2"),  # Runs in parallel!
            ...     pool.generate("Query 3"),
            ... )
        """
        instance = await self._checkout_instance(timeout=timeout)
        try:
            # Explicitly disable streaming - pool returns complete strings
            result = await instance.generate_async(
                prompt,
                max_tokens=max_tokens,
                sampling=sampling,
                stop=stop,
                stream=False,
                **kwargs,
            )
            return cast(str, result)  # generate_async returns str when stream=False
        finally:
            self._return_instance(instance)

    async def generate_batch(
        self,
        prompts: list[str],
        *,
        max_tokens: int = 128,
        sampling: SamplingParams | None = None,
        stop: list[str] | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Generate text for multiple prompts in parallel.

        Convenience method that creates concurrent tasks for all prompts and
        waits for all to complete. Automatically distributes load across pool.

        Args:
            prompts: List of input prompts.
            max_tokens: Maximum tokens per generation.
            sampling: Optional sampling parameters (same for all).
            stop: Optional stop sequences (same for all).
            timeout: Maximum seconds to wait for an available instance.
                None means wait indefinitely. Raises TimeoutError on expiry.
            **kwargs: Additional arguments passed to generate().

        Returns:
            List of generated texts in same order as prompts.

        Example:
            >>> prompts = ["Question 1", "Question 2", "Question 3"]
            >>> results = await pool.generate_batch(prompts, max_tokens=64)
            >>> for prompt, result in zip(prompts, results):
            ...     print(f"{prompt}: {result}")
        """
        tasks = [
            self.generate(
                prompt,
                max_tokens=max_tokens,
                sampling=sampling,
                stop=stop,
                timeout=timeout,
                **kwargs,
            )
            for prompt in prompts
        ]
        return await asyncio.gather(*tasks)

    async def create_chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 128,
        temperature: float | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create chat completion using next available instance.

        Args:
            messages: List of chat messages.
            max_tokens: Maximum tokens to generate.
            temperature: Optional temperature override.
            timeout: Maximum seconds to wait for an available instance.
                None means wait indefinitely. Raises TimeoutError on expiry.
            **kwargs: Additional arguments passed to create_chat_completion_async().

        Returns:
            Chat completion response dict.

        Example:
            >>> response = await pool.create_chat_completion(
            ...     [{"role": "user", "content": "Hello!"}],
            ...     max_tokens=32,
            ... )
            >>> print(response["choices"][0]["message"]["content"])
        """
        instance = await self._checkout_instance(timeout=timeout)
        try:
            # Explicitly disable streaming - pool returns complete dicts
            result = await instance.create_chat_completion_async(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
                **kwargs,
            )
            return cast(dict[str, Any], result)  # returns dict when stream=False
        finally:
            self._return_instance(instance)

    async def create_chat_completion_batch(
        self,
        message_lists: list[list[dict[str, str]]],
        *,
        max_tokens: int = 128,
        temperature: float | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Create chat completions for multiple conversations in parallel.

        Args:
            message_lists: List of message lists (one per conversation).
            max_tokens: Maximum tokens per generation.
            temperature: Optional temperature override.
            timeout: Maximum seconds to wait for an available instance.
                None means wait indefinitely. Raises TimeoutError on expiry.
            **kwargs: Additional arguments passed to create_chat_completion().

        Returns:
            List of chat completion responses in same order.

        Example:
            >>> conversations = [
            ...     [{"role": "user", "content": "Hi"}],
            ...     [{"role": "user", "content": "Hello"}],
            ... ]
            >>> responses = await pool.create_chat_completion_batch(conversations)
        """
        tasks = [
            self.create_chat_completion(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                **kwargs,
            )
            for messages in message_lists
        ]
        return await asyncio.gather(*tasks)

    def close(self) -> None:
        """Close all instances in the pool immediately.

        Releases all resources including GPU memory. Should be called when
        pool is no longer needed, or use async context manager. Safe to call
        multiple times (idempotent).

        Warning:
            Any instances currently checked out by in-flight requests will be
            closed while still in use. Use ``close_graceful()`` to wait for
            in-flight requests to finish first.

        Example:
            >>> pool = LlamaPool("model.gguf", pool_size=2)
            >>> # ... use pool ...
            >>> pool.close()
        """
        with self._close_lock:
            if self._closed:
                return
            self._closed = True

        # If queue was never initialized (pool never used), skip queue operations
        if self._available is not None:
            # Warn if instances are checked out (in-flight requests will be
            # disrupted). qsize() is a best-effort snapshot and close() is sync,
            # so this count is approximate — present it as such.
            in_flight = max(0, self.pool_size - self._available.qsize())
            if in_flight > 0:
                logging.warning(
                    "LlamaPool.close() called with ~%d in-flight request(s) "
                    "(approximate); use close_graceful() to wait for them. "
                    "In-flight requests may encounter errors.",
                    in_flight,
                )
            # Drain real instances from the queue
            while True:
                try:
                    self._available.get_nowait()
                except asyncio.QueueEmpty:
                    break
            # Inject sentinel to unblock any coroutines waiting on get().
            # Each woken waiter re-injects it, so one sentinel propagates
            # through all blocked waiters (poison-pill pattern).
            self._available.put_nowait(_POOL_CLOSED)
        logging.info("Closing LlamaPool with %d instances...", len(self.instances))
        for i, instance in enumerate(self.instances):
            logging.debug("Closing instance %d/%d...", i + 1, len(self.instances))
            instance.close()
        self.instances.clear()
        logging.info("LlamaPool closed")

    async def close_graceful(self, timeout: float = 30.0) -> None:
        """Gracefully close the pool, waiting for in-flight requests to finish.

        Stops accepting new requests immediately, then waits up to ``timeout``
        seconds for checked-out instances to be returned before force-closing.

        Args:
            timeout: Maximum seconds to wait for in-flight requests to complete.
                After this, any still-checked-out instances are force-closed.

        Example:
            >>> pool = LlamaPool("model.gguf", pool_size=2)
            >>> # ... use pool ...
            >>> await pool.close_graceful(timeout=10.0)
        """
        with self._close_lock:
            if self._closed:
                return
            self._closed = True

        # If queue was never initialized (pool never used), skip graceful wait
        if self._available is not None:
            # Inject sentinel to reject new checkout attempts
            self._available.put_nowait(_POOL_CLOSED)

            # Wait for checked-out instances to be returned to the queue.
            # Re-inject the sentinel whenever we encounter it so concurrent
            # waiters in _checkout_instance still wake up with RuntimeError.
            # To avoid a busy-loop when only the sentinel remains (e.g.
            # in-flight requests are stuck), sleep briefly after re-injecting
            # before retrying the get.
            returned = 0
            try:
                deadline = asyncio.get_running_loop().time() + timeout
                while returned < self.pool_size:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(
                            self._available.get(), timeout=remaining
                        )
                    except TimeoutError:
                        break
                    if item is _POOL_CLOSED:
                        # Put it back for other waiters, then yield so any
                        # woken waiter can consume it before we retry get().
                        # Without the sleep, dequeue-then-re-inject would
                        # burn the remaining timeout window in a tight loop
                        # when the sentinel is the only item in the queue.
                        self._available.put_nowait(_POOL_CLOSED)
                        await asyncio.sleep(0.01)
                        continue
                    returned += 1
            except Exception as e:
                # Best-effort drain; the force-close below handles the rest.
                # Log at debug so a stuck/erroring drain is diagnosable without
                # changing the best-effort behavior.
                logging.debug("close_graceful drain interrupted: %s", e)

            if returned < self.pool_size:
                logging.warning(
                    "Graceful shutdown timed out: %d/%d instances still checked out, "
                    "force-closing",
                    self.pool_size - returned,
                    self.pool_size,
                )

        # Force-close all instances regardless of checkout state
        logging.info("Closing LlamaPool with %d instances...", len(self.instances))
        for i, instance in enumerate(self.instances):
            logging.debug("Closing instance %d/%d...", i + 1, len(self.instances))
            instance.close()
        self.instances.clear()
        logging.info("LlamaPool closed")

    async def __aenter__(self) -> LlamaPool:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *_args: Any) -> None:
        """Async context manager exit — waits for in-flight requests."""
        await self.close_graceful()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"LlamaPool(model_path={self.model_path!r}, "
            f"pool_size={self.pool_size}, "
            f"active={len(self.instances)})"
        )
