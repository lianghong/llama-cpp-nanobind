"""Tests for per-sequence on-device state save/load (llama.cpp 2026-04+).

Uses LLAMA_STATE_SEQ_FLAGS_ON_DEVICE — tensor data stays in device buffers
instead of round-tripping through host memory.

CRITICAL invariants (from llama.h + observed behavior):
  1. Only one on-device snapshot per seq_id may be live at a time. Re-saving
     the same seq_id invalidates the prior handle.
  2. The snapshot is also invalidated by anything that clears KV memory:
     ``reset()``, ``kv_cache_clear()``, ``set_state_data()``, ``load_state()``.
     Loading a stale handle calls ``ggml_abort`` (terminates the process)
     — the C API performs no validation.

Tests below respect both invariants: they never reset/clear KV between a
save and the matching load.
"""

import pytest

from llama_cpp import LlamaError
from conftest import requires_model


@requires_model
def test_save_seq_state_on_device_returns_bytes(llm):
    """Smoke test: save returns bytes (the opaque handle)."""
    llm.reset()
    llm.generate("Hello", max_tokens=4)
    handle = llm.save_seq_state_on_device(seq_id=0)
    assert isinstance(handle, bytes)
    assert len(handle) > 0


@requires_model
def test_on_device_round_trip_preserves_generation(llm):
    """Save → extend KV (without clearing) → load handle → next generation
    must match the pre-extension continuation. Verifies the on-device
    snapshot captures sufficient state to roll the sequence back.
    """
    llm.reset()
    # Prime KV cache; record the deterministic continuation we want to recover.
    llm.generate("Once upon a time", max_tokens=4, seed=42)
    handle = llm.save_seq_state_on_device(seq_id=0)
    expected = llm.generate(
        ", there was",
        max_tokens=8,
        seed=42,
        reset_kv_cache=False,
        cache_prompt=False,
    )

    # Restore the snapshot and re-generate. We must NOT call reset() here:
    # any KV-clearing op invalidates the on-device handle. Loading directly
    # over the current KV state rolls the sequence back to the snapshot.
    llm.load_seq_state_on_device(handle, dest_seq_id=0)
    actual = llm.generate(
        ", there was",
        max_tokens=8,
        seed=42,
        reset_kv_cache=False,
        cache_prompt=False,
    )
    assert actual == expected


@requires_model
def test_on_device_load_invalidates_prompt_cache(llm):
    """Loading on-device state into seq 0 must invalidate the prompt cache
    mirror — same invariant as set_state / load_state.
    """
    llm.reset()
    llm.generate("Cached prefix here", max_tokens=2)
    assert len(llm._cached_prompt_tokens) > 0  # type: ignore[attr-defined]

    handle = llm.save_seq_state_on_device(seq_id=0)
    # Extend the sequence (without clearing KV — that would invalidate the handle)
    llm.generate(" and more", max_tokens=2, reset_kv_cache=False, cache_prompt=False)
    llm.load_seq_state_on_device(handle, dest_seq_id=0)

    assert len(llm._cached_prompt_tokens) == 0  # type: ignore[attr-defined]


@requires_model
def test_on_device_state_after_close_raises(model_path):
    """Using on-device state methods on a closed instance must raise a
    clear LlamaError (close-guard), not an AttributeError.
    """
    from llama_cpp import Llama, LlamaConfig

    llm = Llama(model_path, config=LlamaConfig(model_path=str(model_path), n_ctx=512))
    llm.close()
    with pytest.raises(LlamaError):
        llm.save_seq_state_on_device()
    with pytest.raises(LlamaError):
        llm.load_seq_state_on_device(b"x")


# NOTE: there is intentionally no "garbage handle raises" test for the
# on-device path. Upstream `llama_state_seq_set_data_ext` with the
# LLAMA_STATE_SEQ_FLAGS_ON_DEVICE flag calls ggml_abort() on invalid handle
# bytes rather than returning 0, which terminates the process. The opaque
# handle has no public format we could pre-validate in the binding either,
# so this failure mode is unreachable from a well-behaved caller. Round-trip
# integrity is covered by test_on_device_round_trip_preserves_generation.
