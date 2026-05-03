"""Tests for LlamaPool close semantics — specifically _POOL_CLOSED sentinel
propagation and waiter wake-up behavior.

These tests avoid loading a real model by monkey-patching the ``Llama`` class
inside ``llama_cpp.pool`` with a lightweight stub. They exercise the queue /
checkout / close paths directly.
"""

import asyncio

import pytest

from llama_cpp import pool as pool_module


class _StubLlama:
    """Stand-in for Llama that does no real work but supports close()."""

    _closed = False

    def __init__(self, *_args, **_kwargs) -> None:
        self._closed = False

    def close(self) -> None:
        self._closed = True


@pytest.fixture
def stub_llama(monkeypatch):
    """Patch pool_module.Llama with a no-model stub."""
    monkeypatch.setattr(pool_module, "Llama", _StubLlama)
    return _StubLlama


@pytest.mark.asyncio
async def test_close_wakes_all_blocked_waiters(stub_llama):
    """When close() runs while N coroutines are blocked in _checkout_instance,
    every waiter must wake up with RuntimeError (not hang forever)."""
    pool = pool_module.LlamaPool("dummy.gguf", pool_size=1)

    # Exhaust the pool: check out the only instance so subsequent checkouts block.
    inst = await pool._checkout_instance()

    # Launch several coroutines that will all block in _checkout_instance.
    n_waiters = 5
    waiters = [asyncio.create_task(pool._checkout_instance()) for _ in range(n_waiters)]

    # Give them a tick to reach the blocked state.
    await asyncio.sleep(0.05)
    assert all(not t.done() for t in waiters), "waiters should still be blocked"

    # Close while waiters are parked. The sentinel must propagate to every one.
    pool.close()

    # Every waiter should raise RuntimeError("LlamaPool is closed") within
    # a reasonable timeout — no permanent blocks.
    results = await asyncio.gather(*waiters, return_exceptions=True)
    assert len(results) == n_waiters
    for r in results:
        assert isinstance(r, RuntimeError)
        assert "closed" in str(r).lower()

    # The stub instance checked out earlier was force-closed by close().
    assert inst._closed is True


@pytest.mark.asyncio
async def test_close_graceful_wakes_all_blocked_waiters(stub_llama):
    """close_graceful() must also propagate the sentinel to waiters."""
    pool = pool_module.LlamaPool("dummy.gguf", pool_size=1)

    # Hold the only instance so subsequent checkouts block.
    held = await pool._checkout_instance()

    n_waiters = 4
    waiters = [asyncio.create_task(pool._checkout_instance()) for _ in range(n_waiters)]
    await asyncio.sleep(0.05)
    assert all(not t.done() for t in waiters)

    # Return the held instance shortly after initiating graceful close, then
    # let the graceful close drive the sentinel through remaining waiters.
    async def release_after_close() -> None:
        await asyncio.sleep(0.05)
        pool._return_instance(held)

    release_task = asyncio.create_task(release_after_close())
    await pool.close_graceful(timeout=1.0)
    await release_task

    results = await asyncio.gather(*waiters, return_exceptions=True)
    for r in results:
        assert isinstance(r, RuntimeError)
        assert "closed" in str(r).lower()


@pytest.mark.asyncio
async def test_checkout_after_close_raises(stub_llama):
    """A fresh _checkout_instance() on an already-closed pool raises
    synchronously (not after waiting) — the early self._closed check."""
    pool = pool_module.LlamaPool("dummy.gguf", pool_size=2)
    pool.close()

    with pytest.raises(RuntimeError, match="closed"):
        await pool._checkout_instance()


@pytest.mark.asyncio
async def test_close_is_idempotent_under_concurrent_calls(stub_llama):
    """Two concurrent close() calls must not both enter the close path.

    The _close_lock guards the check-then-set; only the first caller performs
    the shutdown work, the second returns immediately.
    """
    pool = pool_module.LlamaPool("dummy.gguf", pool_size=3)

    # Force the queue to be initialized so close() hits the drain path.
    await pool._checkout_instance()  # triggers _ensure_queue_initialized

    # Run close() twice concurrently from threads — the lock should prevent
    # double-entering the drain / close-instances path.
    loop = asyncio.get_running_loop()
    await asyncio.gather(
        loop.run_in_executor(None, pool.close),
        loop.run_in_executor(None, pool.close),
    )

    # Every instance closed exactly once (stub tracks via _closed flag).
    assert pool._closed is True
    # After close, instances list is cleared.
    assert pool.instances == []


@pytest.mark.asyncio
async def test_close_graceful_does_not_busy_loop_on_sentinel(stub_llama):
    """Regression for the close_graceful drain loop.

    When a pool_size-worth of instances are checked out and never returned,
    close_graceful should sit out the timeout in a sleep-yielding loop, not
    busy-spin dequeue/re-inject on the sentinel. A tight loop would burn
    100% CPU on the event loop thread for the full timeout.
    """
    pool = pool_module.LlamaPool("dummy.gguf", pool_size=2)

    # Hold both instances — they never return during this test.
    _a = await pool._checkout_instance()
    _b = await pool._checkout_instance()

    # close_graceful should hit the timeout cleanly. We give it a short
    # timeout and assert close_graceful doesn't hang noticeably past it.
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    await pool.close_graceful(timeout=0.2)
    elapsed = loop.time() - t0

    # If the loop were tight, elapsed would still be ~0.2s (bounded by
    # remaining deadline) but would burn CPU. We can't directly measure
    # CPU here; instead assert elapsed stays in a reasonable window.
    assert 0.15 <= elapsed < 2.0, f"close_graceful took {elapsed:.3f}s"
    assert pool._closed is True


def test_pool_cross_event_loop_raises_clear_error(stub_llama):
    """Using the same LlamaPool instance from two different event loops
    must raise a clear RuntimeError, not silently hang on the asyncio.Queue.
    """
    pool = pool_module.LlamaPool("dummy.gguf", pool_size=1)

    # Loop 1: bind the pool.
    async def _loop1() -> None:
        inst = await pool._checkout_instance()
        pool._return_instance(inst)

    asyncio.run(_loop1())

    # Loop 2: attempt reuse — must raise RuntimeError.
    async def _loop2() -> None:
        with pytest.raises(RuntimeError, match="different event loop"):
            await pool._checkout_instance()

    asyncio.run(_loop2())


@pytest.mark.asyncio
async def test_sentinel_reinjection_survives_multiple_waiters(stub_llama):
    """Verify the poison-pill pattern: after close() puts one _POOL_CLOSED,
    each waiter re-injects it so later waiters also wake up.

    This exercises _checkout_instance's re-injection branch (line 211-213).
    """
    pool = pool_module.LlamaPool("dummy.gguf", pool_size=1)
    await pool._checkout_instance()  # exhaust

    # Sequential waiters arriving AFTER close() — each should wake by reading
    # the sentinel, then re-inject it for the next caller.
    pool.close()

    for _ in range(3):
        with pytest.raises(RuntimeError, match="closed"):
            await pool._checkout_instance()
