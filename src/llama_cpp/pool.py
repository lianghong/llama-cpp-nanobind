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

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .llama import Llama, LlamaConfig, SamplingParams


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

        # Create worker instances
        logging.info(f"Initializing LlamaPool with {pool_size} instances...")
        self.instances: list[Llama] = []
        for i in range(pool_size):
            logging.debug(f"Loading instance {i+1}/{pool_size}...")
            instance = Llama(model_path, config=self.config)
            self.instances.append(instance)
        logging.info(f"LlamaPool initialized with {pool_size} instances")

        # Warmup phase: run dummy inference to prime GPU caches
        if warmup:
            logging.info("Running warmup phase to pre-load GPU caches...")
            self._warmup_instances()
            logging.info("Warmup phase complete")

        # Semaphore to limit concurrent access to pool_size
        self._semaphore = asyncio.Semaphore(pool_size)

        # Round-robin index for load balancing
        self._round_robin_index = 0
        self._index_lock = asyncio.Lock()

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
                logging.debug(f"Warming up instance {i+1}/{self.pool_size}...")
                # Run minimal inference to trigger GPU initialization
                instance.generate(
                    warmup_prompt,
                    max_tokens=warmup_tokens,
                    reset_kv_cache=True,
                )
                # Clear KV cache after warmup to start fresh
                instance.ctx.kv_cache_clear()
            except Exception as e:
                # Log but don't fail initialization - warmup is optional optimization
                logging.warning(
                    f"Warmup failed for instance {i+1} (non-fatal): {e}"
                )

    async def _get_instance(self) -> Llama:
        """Get next available instance using round-robin selection."""
        async with self._index_lock:
            instance = self.instances[self._round_robin_index]
            self._round_robin_index = (self._round_robin_index + 1) % self.pool_size
            return instance

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 128,
        sampling: SamplingParams | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate text using next available instance.

        This method distributes requests across instances in the pool using
        round-robin scheduling. Multiple calls run in parallel up to pool_size.

        Args:
            prompt: Input prompt string.
            max_tokens: Maximum tokens to generate.
            sampling: Optional sampling parameters.
            stop: Optional stop sequences.
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
        async with self._semaphore:
            instance = await self._get_instance()
            return await instance.generate_async(
                prompt,
                max_tokens=max_tokens,
                sampling=sampling,
                stop=stop,
                **kwargs,
            )

    async def generate_batch(
        self,
        prompts: list[str],
        *,
        max_tokens: int = 128,
        sampling: SamplingParams | None = None,
        stop: list[str] | None = None,
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
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create chat completion using next available instance.

        Args:
            messages: List of chat messages.
            max_tokens: Maximum tokens to generate.
            temperature: Optional temperature override.
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
        async with self._semaphore:
            instance = await self._get_instance()
            return await instance.create_chat_completion_async(
                messages, max_tokens=max_tokens, temperature=temperature, **kwargs
            )

    async def create_chat_completion_batch(
        self,
        message_lists: list[list[dict[str, str]]],
        *,
        max_tokens: int = 128,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Create chat completions for multiple conversations in parallel.

        Args:
            message_lists: List of message lists (one per conversation).
            max_tokens: Maximum tokens per generation.
            temperature: Optional temperature override.
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
                messages, max_tokens=max_tokens, temperature=temperature, **kwargs
            )
            for messages in message_lists
        ]
        return await asyncio.gather(*tasks)

    def close(self) -> None:
        """Close all instances in the pool.

        Releases all resources including GPU memory. Should be called when
        pool is no longer needed, or use async context manager.

        Example:
            >>> pool = LlamaPool("model.gguf", pool_size=2)
            >>> # ... use pool ...
            >>> pool.close()
        """
        logging.info(f"Closing LlamaPool with {len(self.instances)} instances...")
        for i, instance in enumerate(self.instances):
            logging.debug(f"Closing instance {i+1}/{len(self.instances)}...")
            instance.close()
        self.instances.clear()
        logging.info("LlamaPool closed")

    async def __aenter__(self) -> LlamaPool:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        self.close()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"LlamaPool(model_path={self.model_path!r}, "
            f"pool_size={self.pool_size}, "
            f"active={len(self.instances)})"
        )
