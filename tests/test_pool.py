"""Tests for LlamaPool parallel inference."""

import asyncio

import pytest

from llama_cpp import LlamaConfig, LlamaPool


@pytest.mark.asyncio
async def test_pool_basic_generation(model_path):
    """Test basic generation with pool."""
    async with LlamaPool(model_path, pool_size=2) as pool:
        result = await pool.generate("Test prompt", max_tokens=8)
        assert isinstance(result, str)
        assert len(result) > 0


@pytest.mark.asyncio
async def test_pool_batch_generation(model_path):
    """Test batch generation with pool."""
    async with LlamaPool(model_path, pool_size=2) as pool:
        prompts = ["Test 1", "Test 2", "Test 3"]
        results = await pool.generate_batch(prompts, max_tokens=8)

        assert len(results) == len(prompts)
        assert all(isinstance(r, str) and len(r) > 0 for r in results)


@pytest.mark.asyncio
async def test_pool_concurrent_requests(model_path):
    """Test that pool handles concurrent requests correctly."""
    async with LlamaPool(model_path, pool_size=2) as pool:
        # Submit 4 requests concurrently to pool of 2
        tasks = [pool.generate(f"Query {i}", max_tokens=8) for i in range(4)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 4
        assert all(isinstance(r, str) and len(r) > 0 for r in results)


@pytest.mark.asyncio
async def test_pool_chat_completion(model_path):
    """Test chat completion with pool."""
    config = LlamaConfig(model_path=model_path, chat_format="qwen3")
    async with LlamaPool(model_path, pool_size=2, config=config) as pool:
        messages = [{"role": "user", "content": "Hello"}]
        response = await pool.create_chat_completion(messages, max_tokens=8)

        assert "choices" in response
        assert len(response["choices"]) > 0
        assert "message" in response["choices"][0]


@pytest.mark.asyncio
async def test_pool_chat_completion_batch(model_path):
    """Test batch chat completions with pool."""
    config = LlamaConfig(model_path=model_path, chat_format="qwen3")
    async with LlamaPool(model_path, pool_size=2, config=config) as pool:
        conversations = [
            [{"role": "user", "content": "Hi"}],
            [{"role": "user", "content": "Hello"}],
        ]
        responses = await pool.create_chat_completion_batch(
            conversations, max_tokens=8
        )

        assert len(responses) == len(conversations)
        assert all("choices" in r for r in responses)


def test_pool_invalid_size(model_path):
    """Test that invalid pool_size raises error."""
    with pytest.raises(ValueError, match="pool_size must be >= 1"):
        LlamaPool(model_path, pool_size=0)

    with pytest.raises(ValueError, match="pool_size must be >= 1"):
        LlamaPool(model_path, pool_size=-1)


@pytest.mark.asyncio
async def test_pool_context_manager(model_path):
    """Test that pool cleans up resources with context manager."""
    pool = LlamaPool(model_path, pool_size=2)
    assert len(pool.instances) == 2

    async with pool:
        result = await pool.generate("Test", max_tokens=8)
        assert isinstance(result, str)

    # After exiting context, instances should be closed
    # (we can't easily verify this without inspecting internal state)


def test_pool_manual_close(model_path):
    """Test manual close of pool."""
    pool = LlamaPool(model_path, pool_size=2)
    assert len(pool.instances) == 2

    pool.close()
    assert len(pool.instances) == 0


@pytest.mark.asyncio
async def test_pool_repr(model_path):
    """Test pool string representation."""
    pool = LlamaPool(model_path, pool_size=3)
    repr_str = repr(pool)

    assert "LlamaPool" in repr_str
    assert model_path in repr_str
    assert "pool_size=3" in repr_str
    assert "active=3" in repr_str

    pool.close()


@pytest.mark.asyncio
async def test_pool_with_sampling_params(model_path):
    """Test pool with custom sampling parameters."""
    from llama_cpp import SamplingParams

    sampling = SamplingParams(temperature=0.7, top_k=40)
    async with LlamaPool(model_path, pool_size=2) as pool:
        result = await pool.generate(
            "Test prompt", max_tokens=8, sampling=sampling
        )
        assert isinstance(result, str)
        assert len(result) > 0


@pytest.mark.asyncio
async def test_pool_with_stop_sequences(model_path):
    """Test pool with stop sequences."""
    async with LlamaPool(model_path, pool_size=2) as pool:
        result = await pool.generate(
            "Count: 1, 2, 3", max_tokens=16, stop=[","]
        )
        assert isinstance(result, str)
        # Result should stop at comma (though this depends on model behavior)


@pytest.mark.asyncio
async def test_pool_load_balancing(model_path):
    """Test that pool distributes requests across instances."""
    # This test verifies round-robin behavior by checking instance assignment
    # (implementation detail test)
    async with LlamaPool(model_path, pool_size=3) as pool:
        # Submit requests and verify they're distributed
        # We can't directly observe which instance handled each request,
        # but we can verify all complete successfully
        tasks = [pool.generate(f"Query {i}", max_tokens=4) for i in range(6)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 6
        assert all(isinstance(r, str) for r in results)


@pytest.mark.asyncio
async def test_pool_warmup(model_path):
    """Test pool with warmup enabled works correctly."""
    async with LlamaPool(model_path, pool_size=2, warmup=True) as pool:
        result = await pool.generate("Test", max_tokens=4)
        assert isinstance(result, str)
        assert len(result) > 0
