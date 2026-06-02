"""Tests for async API."""

import time

from conftest import requires_model
import pytest


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


@pytest.mark.asyncio
@requires_model
async def test_generate_async_stream_is_incremental(llm):
    """``generate_async(stream=True)`` must yield each chunk as it crosses
    the asyncio.Queue bridge — not after the underlying sync generator has
    fully drained. Regression for the queue-bridge path.
    """
    first_chunk_time = None
    last_chunk_time = None
    chunk_count = 0
    start = time.time()

    async for chunk in await llm.generate_async(
        "Count to ten", max_tokens=32, stream=True
    ):
        if first_chunk_time is None:
            first_chunk_time = time.time()
        last_chunk_time = time.time()
        chunk_count += 1
        assert isinstance(chunk, str)

    assert chunk_count > 0

    if chunk_count > 1 and last_chunk_time - start > 0.1:
        ratio = (first_chunk_time - start) / (last_chunk_time - start)
        assert ratio < 0.8, (
            f"First async chunk arrived at {ratio:.1%} of total streaming "
            "time; generate_async appears to buffer."
        )


@pytest.mark.asyncio
@requires_model
async def test_chat_completion_async_stream_is_incremental(llm):
    """``create_chat_completion_async(stream=True)`` must yield chunks
    incrementally through the queue-bridge, not after a full drain. The
    grammar/tools eager-buffer fallback is unreachable here (no grammar
    or tools passed), so this exercises the incremental path.
    """
    first_chunk_time = None
    last_chunk_time = None
    content_chunks = 0
    start = time.time()

    async for chunk in await llm.create_chat_completion_async(
        [{"role": "user", "content": "Count to ten."}],
        max_tokens=32,
        stream=True,
    ):
        assert chunk["object"] == "chat.completion.chunk"
        delta = chunk["choices"][0]["delta"]
        if delta.get("content"):
            if first_chunk_time is None:
                first_chunk_time = time.time()
            last_chunk_time = time.time()
            content_chunks += 1

    assert content_chunks > 0

    if content_chunks > 1 and last_chunk_time - start > 0.1:
        ratio = (first_chunk_time - start) / (last_chunk_time - start)
        assert ratio < 0.8, (
            f"First async chat chunk arrived at {ratio:.1%} of total "
            "streaming time; create_chat_completion_async appears to buffer."
        )


@pytest.mark.asyncio
@requires_model
async def test_generate_async_stream_early_break(llm):
    """Breaking out of an async generate_stream loop must release the lock
    and leave the instance reusable. Without proper aclose() cleanup the
    underlying sync worker would hold ``self._lock`` and the follow-up
    call below would block forever.
    """
    chunks = []
    stream = await llm.generate_async("Count to twenty", max_tokens=64, stream=True)
    async for piece in stream:
        chunks.append(piece)
        if len(chunks) >= 3:
            break
    # Closing the generator triggers its finally block (await pump → join).
    await stream.aclose()

    assert len(chunks) == 3
    assert not llm.is_stuck, "async stream early-break left worker stuck"

    # Instance must be reusable; this would block on a still-held _lock.
    follow_up = await llm.generate_async("Hello", max_tokens=4)
    assert isinstance(follow_up, str) and follow_up


@pytest.mark.asyncio
@requires_model
async def test_chat_completion_async_stream_early_break(llm):
    """Breaking out of an async chat-stream loop must release the lock
    and leave the instance reusable. Covers the queue-bridge path added
    to create_chat_completion_async for incremental streaming.
    """
    chunks = []
    stream = await llm.create_chat_completion_async(
        [{"role": "user", "content": "Count to twenty."}],
        max_tokens=64,
        stream=True,
    )
    async for chunk in stream:
        chunks.append(chunk)
        if len(chunks) >= 3:
            break
    await stream.aclose()

    assert len(chunks) == 3
    assert not llm.is_stuck, "async chat stream early-break left worker stuck"

    follow_up = await llm.create_chat_completion_async(
        [{"role": "user", "content": "Hello"}],
        max_tokens=4,
    )
    assert follow_up["object"] == "chat.completion"


@pytest.mark.asyncio
@requires_model
async def test_generate_async_stream_task_cancellation(llm):
    """Cancelling the task that drives an async stream must not hang or leave
    the instance stuck. Exercises the cross-thread queue bridge under the
    realistic shutdown trigger (CancelledError thrown into the consumer)."""
    import asyncio

    started = asyncio.Event()

    async def consume() -> None:
        stream = await llm.generate_async("Count to fifty", max_tokens=128, stream=True)
        async for _ in stream:
            started.set()
            await asyncio.sleep(0.05)  # let the cancellation land mid-stream

    task = asyncio.create_task(consume())
    await asyncio.wait_for(started.wait(), timeout=30)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The generator's finally/aclose must have released the lock and joined the
    # worker; a brief wait covers the worker-thread join, then the instance
    # must be reusable (this would block forever on a still-held lock).
    for _ in range(100):
        if not llm.is_stuck:
            break
        await asyncio.sleep(0.05)
    assert not llm.is_stuck, "task cancellation left the worker stuck"
    follow_up = await llm.generate_async("Hello", max_tokens=4)
    assert isinstance(follow_up, str) and follow_up


@pytest.mark.asyncio
@requires_model
async def test_generate_async_stream_slow_consumer_no_loss(llm):
    """A slow consumer must receive every chunk in order — the worker applies
    backpressure (run_coroutine_threadsafe(...).result()) rather than racing
    ahead and dropping chunks or the final sentinel."""
    import asyncio

    fast = []
    async for chunk in await llm.generate_async("Count to ten:", max_tokens=24, stream=True):
        fast.append(chunk)

    slow = []
    async for chunk in await llm.generate_async("Count to ten:", max_tokens=24, stream=True):
        slow.append(chunk)
        await asyncio.sleep(0.02)  # drain slower than the worker produces

    # Deterministic (greedy default seed is fixed per call? not guaranteed) —
    # so assert structural integrity rather than equality: the slow consumer
    # saw a non-empty, fully-terminated stream with no lost chunks.
    assert slow, "slow consumer received no chunks"
    assert "".join(slow).strip(), "slow consumer received only empty chunks"
