"""Memory-safety regression scenarios.

Exercises resource cleanup paths that historically caused double-free
crashes:

  1. Explicit double close() on a single instance
  2. Context manager followed by explicit close()
  3. Sequential create/destroy cycles (backend lifecycle)
  4. State save/load then close (stale pointer risk)
  5. GC collection racing with explicit close
  6. Rapid create-close loops
  7. Multiple instances closed in different orders

A successful run finishes with all tests passing. Any double-free will
manifest as SIGABRT / SIGSEGV — pytest reports those as test failures.

For allocator-level corruption detection, run under glibc's heap
checker:

    MALLOC_CHECK_=3 uv run pytest tests/test_double_free_scenarios.py -v
"""

import gc
import os
import tempfile

import pytest

from llama_cpp import Llama
from llama_cpp import LlamaConfig
from llama_cpp import LlamaError
from llama_cpp.unified import UnifiedLLM

from conftest import MODEL_PATH, requires_model


def _make_llm(**kwargs):
    """Create a small-context Llama instance for testing."""
    defaults = {
        "model_path": MODEL_PATH,
        "n_ctx": 512,
        "n_gpu_layers": 99,
        "verbose": False,
    }
    defaults.update(kwargs)
    cfg = LlamaConfig(**defaults)
    return Llama(MODEL_PATH, config=cfg)


def _make_unified(**kwargs):
    """Create a small-context UnifiedLLM instance for testing.

    Skips automatically when MODEL_PATH is not a UnifiedLLM-supported
    family (Qwen 3.5 / 3.6, Gemma 4, Granite 4.1).
    """
    try:
        return UnifiedLLM(
            MODEL_PATH, n_ctx=512, n_gpu_layers=99, verbose=False, **kwargs
        )
    except Exception as e:
        pytest.skip(f"UnifiedLLM rejected MODEL_PATH: {e}")


# ── Llama scenarios ─────────────────────────────────────────────────────────


@requires_model
def test_llama_double_close():
    """Calling close() twice must not crash."""
    llm = _make_llm()
    llm.close()
    llm.close()


@requires_model
def test_llama_context_manager_then_close():
    """close() after __exit__ must not crash."""
    with _make_llm() as llm:
        llm.generate("Hi", max_tokens=4)
    llm.close()


@requires_model
def test_llama_sequential_cycles():
    """Repeated create-use-destroy must not leak or double-free backend."""
    for _ in range(3):
        with _make_llm() as llm:
            llm.generate("test", max_tokens=4)


@requires_model
def test_llama_state_roundtrip_close():
    """get_state / set_state followed by close must not crash."""
    with _make_llm() as llm:
        llm.generate("Hello world", max_tokens=8)
        state = llm.get_state()
        llm.set_state(state)
        llm.generate("Continue", max_tokens=4, reset_kv_cache=False)


@requires_model
def test_llama_state_file_close():
    """save_state / load_state to file followed by close must not crash."""
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        state_path = tmp.name
    try:
        with _make_llm() as llm:
            llm.generate("Hello", max_tokens=8)
            llm.save_state(state_path)

        with _make_llm() as llm:
            llm.load_state(state_path)
            llm.generate("Continue", max_tokens=4, reset_kv_cache=False)
    finally:
        os.unlink(state_path)


@requires_model
def test_llama_gc_pressure():
    """Force GC while close is releasing resources."""
    llm = _make_llm()
    llm.generate("test", max_tokens=4)
    a: dict = {}
    b: dict = {"ref": a}
    a["ref"] = b
    del a, b
    gc.collect()
    llm.close()
    gc.collect()


@requires_model
def test_llama_use_after_close_raises():
    """Using a closed instance must raise LlamaError, not crash."""
    llm = _make_llm()
    llm.close()
    with pytest.raises(LlamaError):
        llm.generate("Should fail", max_tokens=4)


@requires_model
def test_llama_multi_instance_close_creation_order():
    """Close instances in creation order (non-LIFO)."""
    a = _make_llm()
    b = _make_llm(n_gpu_layers=0)
    a.close()
    b.close()


@requires_model
def test_llama_multi_instance_close_reverse_order():
    """Close instances in reverse-creation (LIFO) order."""
    a = _make_llm()
    b = _make_llm(n_gpu_layers=0)
    b.close()
    a.close()


@requires_model
def test_llama_rapid_create_close():
    """Rapid allocation/deallocation loop to stress the allocator."""
    for _ in range(10):
        llm = _make_llm()
        llm.close()
    gc.collect()


@requires_model
def test_llama_del_without_close():
    """Dropping reference without close() must not crash (RAII handles it)."""
    llm = _make_llm()
    llm.generate("test", max_tokens=4)
    del llm
    gc.collect()


# ── UnifiedLLM scenarios ────────────────────────────────────────────────────


@requires_model
def test_unified_double_close():
    """UnifiedLLM.close() twice must not crash (cascades to Llama.close())."""
    llm = _make_unified()
    llm.close()
    llm.close()


@requires_model
def test_unified_context_manager_then_close():
    """UnifiedLLM __exit__ then explicit close() must not crash."""
    with _make_unified() as llm:
        llm.generate("Hi", max_tokens=4)
    llm.close()


@requires_model
def test_unified_sequential_cycles():
    """Repeated UnifiedLLM create-use-destroy cycles."""
    for _ in range(3):
        with _make_unified() as llm:
            llm.generate("test", max_tokens=4)


@requires_model
def test_unified_del():
    """UnifiedLLM __del__ must clean up without crash."""
    llm = _make_unified()
    llm.generate("test", max_tokens=4)
    del llm
    gc.collect()


@requires_model
def test_unified_close_then_del():
    """close() then __del__ via GC must not double-free."""
    llm = _make_unified()
    llm.generate("test", max_tokens=4)
    llm.close()
    del llm
    gc.collect()


@requires_model
def test_unified_use_after_close_raises():
    """Operations on closed UnifiedLLM must raise, not crash."""
    llm = _make_unified()
    llm.close()
    # AttributeError because UnifiedLLM sets self.llm = None on close.
    with pytest.raises((LlamaError, AttributeError)):
        llm.generate("Should fail", max_tokens=4)


@requires_model
def test_unified_and_llama_mixed_close_order():
    """Create both UnifiedLLM and Llama, close in interleaved order."""
    u = _make_unified()
    standalone = _make_llm(n_gpu_layers=0)
    u.close()
    standalone.close()


@requires_model
def test_unified_rapid_create_close():
    """Rapid UnifiedLLM allocation/deallocation loop."""
    for _ in range(5):
        llm = _make_unified()
        llm.close()
    gc.collect()


@requires_model
def test_unified_inner_llm_is_none_after_close():
    """After close(), inner llm is None; accessing it must not crash."""
    llm = _make_unified()
    llm.close()
    assert llm.llm is None
