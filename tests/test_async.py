"""Tests for async API."""

import pytest

from conftest import requires_model


@pytest.mark.asyncio
@requires_model
async def test_generate_async(llm):
    """Test basic async generation."""
    text = await llm.generate_async("Hello", max_tokens=8)
    assert isinstance(text, str)
    assert len(text.strip()) > 0


@pytest.mark.asyncio
@requires_model
async def test_generate_async_stream(llm):
    """Test async streaming generation."""
    chunks = []
    async for chunk in await llm.generate_async("Test", max_tokens=8, stream=True):
        chunks.append(chunk)
        assert isinstance(chunk, str)
    assert len(chunks) > 0


@pytest.mark.asyncio
@requires_model
async def test_chat_completion_async(llm):
    """Test async chat completion."""
    response = await llm.create_chat_completion_async(
        [{"role": "user", "content": "Hi"}],
        max_tokens=8,
    )
    assert response["object"] == "chat.completion"
    assert isinstance(response["choices"][0]["message"]["content"], str)


@pytest.mark.asyncio
@requires_model
async def test_chat_completion_async_stream(llm):
    """Test async streaming chat completion."""
    chunks = []
    async for chunk in await llm.create_chat_completion_async(
        [{"role": "user", "content": "Hi"}],
        max_tokens=8,
        stream=True,
    ):
        chunks.append(chunk)
        assert chunk["object"] == "chat.completion.chunk"
    assert len(chunks) > 0


@pytest.mark.asyncio
@requires_model
async def test_embed_async(llm_embed):
    """Test async embedding."""
    embedding = await llm_embed.embed_async("test")
    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(x, float) for x in embedding)


@pytest.mark.asyncio
@requires_model
async def test_concurrent_async(llm):
    """Test concurrent async operations."""
    import asyncio

    results = await asyncio.gather(
        llm.generate_async("Hello", max_tokens=5),
        llm.generate_async("World", max_tokens=5),
    )
    assert len(results) == 2
    assert all(isinstance(r, str) for r in results)
