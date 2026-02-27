#!/usr/bin/env python3
"""Verify that double-free issues do not occur.

Exercises resource cleanup paths that historically caused double-free crashes:
  1. Explicit double close() on a single instance
  2. Context manager followed by explicit close()
  3. Sequential create/destroy cycles (backend lifecycle)
  4. State save/load then close (stale pointer risk)
  5. GC collection racing with explicit close
  6. Rapid create-close loops
  7. Multiple instances closed in different orders

A successful run prints PASS for each scenario and exits cleanly (rc 0).
Any double-free will manifest as SIGABRT / SIGSEGV and a non-zero exit.
"""

import gc
import signal
import sys
import tempfile

from llama_cpp import Llama
from llama_cpp import LlamaConfig
from llama_cpp import LlamaError
from llama_cpp import ModelLoadError
from llama_cpp import shutdown
from llama_cpp.unified import UnifiedLLM


MODEL_PATH = "models/Qwen3-8B-Q6_K.gguf"

passed = 0
failed = 0


def report(name: str, ok: bool, detail: str = "") -> None:
    global passed, failed
    status = "PASS" if ok else "FAIL"
    suffix = f" -- {detail}" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    if ok:
        passed += 1
    else:
        failed += 1


def make_llm(**kwargs):
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


def make_unified(**kwargs):
    """Create a small-context UnifiedLLM instance for testing."""
    return UnifiedLLM(MODEL_PATH, n_ctx=512, n_gpu_layers=99, verbose=False, **kwargs)


# ── Scenario 1: Double close() ──────────────────────────────────────────────
def test_double_close():
    """Calling close() twice must not crash."""
    llm = make_llm()
    llm.close()
    llm.close()  # Second close should be a no-op
    report("double close()", True)


# ── Scenario 2: Context manager + explicit close() ──────────────────────────
def test_context_manager_then_close():
    """close() after __exit__ must not crash."""
    with make_llm() as llm:
        llm.generate("Hi", max_tokens=4)
    llm.close()  # Already closed by __exit__
    report("context manager + close()", True)


# ── Scenario 3: Sequential create/destroy cycles ────────────────────────────
def test_sequential_cycles():
    """Repeated create-use-destroy must not leak or double-free backend."""
    for _i in range(3):
        with make_llm() as llm:
            llm.generate("test", max_tokens=4)
    report("sequential create/destroy x3", True)


# ── Scenario 4: State round-trip then close ─────────────────────────────────
def test_state_roundtrip_close():
    """get_state / set_state followed by close must not crash."""
    with make_llm() as llm:
        llm.generate("Hello world", max_tokens=8)
        state = llm.get_state()
        llm.set_state(state)
        llm.generate("Continue", max_tokens=4, reset_kv_cache=False)
    # __exit__ calls close; verify no crash
    report("state round-trip + close", True)


# ── Scenario 5: State save/load via file then close ─────────────────────────
def test_state_file_close():
    """save_state / load_state to file followed by close must not crash."""
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        state_path = tmp.name

    with make_llm() as llm:
        llm.generate("Hello", max_tokens=8)
        llm.save_state(state_path)

    with make_llm() as llm:
        llm.load_state(state_path)
        llm.generate("Continue", max_tokens=4, reset_kv_cache=False)

    import os

    os.unlink(state_path)
    report("state file save/load + close", True)


# ── Scenario 6: GC pressure during close ────────────────────────────────────
def test_gc_pressure():
    """Force GC while close is releasing resources."""
    llm = make_llm()
    llm.generate("test", max_tokens=4)
    # Create reference cycles that GC must collect
    a: dict = {}
    b: dict = {"ref": a}
    a["ref"] = b
    del a, b
    gc.collect()
    llm.close()
    gc.collect()
    report("GC pressure during close", True)


# ── Scenario 7: Operations on closed instance ───────────────────────────────
def test_use_after_close():
    """Using a closed instance must raise LlamaError, not crash."""
    llm = make_llm()
    llm.close()
    ok = False
    try:
        llm.generate("Should fail", max_tokens=4)
    except LlamaError:
        ok = True
    except Exception as e:
        report(
            "use-after-close", False, f"unexpected exception: {type(e).__name__}: {e}"
        )
        return
    report("use-after-close raises LlamaError", ok)


# ── Scenario 8: Multiple instances closed in different orders ────────────────
def test_multi_instance_close_order():
    """Close instances in non-LIFO order (reverse of creation)."""
    # Second instance uses CPU to avoid GPU OOM with large models
    a = make_llm()
    b = make_llm(n_gpu_layers=0)
    # Close in creation order (not reverse)
    a.close()
    b.close()
    report("multi-instance close (creation order)", True)

    a = make_llm()
    b = make_llm(n_gpu_layers=0)
    # Close in reverse order
    b.close()
    a.close()
    report("multi-instance close (reverse order)", True)


# ── Scenario 9: Rapid create-close loop ─────────────────────────────────────
def test_rapid_create_close():
    """Rapid allocation/deallocation loop to stress the allocator."""
    for _ in range(10):
        llm = make_llm()
        llm.close()
    gc.collect()
    report("rapid create-close x10", True)


# ── Scenario 10: del without close ──────────────────────────────────────────
def test_del_without_close():
    """Dropping reference without close() must not crash (RAII handles it)."""
    llm = make_llm()
    llm.generate("test", max_tokens=4)
    del llm
    gc.collect()
    report("del without close (RAII)", True)


# ═══════════════════════════════════════════════════════════════════════════════
# UnifiedLLM scenarios
# ═══════════════════════════════════════════════════════════════════════════════


# ── Scenario 11: UnifiedLLM double close() ───────────────────────────────────
def test_unified_double_close():
    """UnifiedLLM.close() twice must not crash (cascades to Llama.close())."""
    llm = make_unified()
    llm.close()
    llm.close()
    report("UnifiedLLM double close()", True)


# ── Scenario 12: UnifiedLLM context manager + close() ───────────────────────
def test_unified_context_manager_then_close():
    """UnifiedLLM __exit__ then explicit close() must not crash."""
    with make_unified() as llm:
        llm.generate("Hi", max_tokens=4)
    llm.close()
    report("UnifiedLLM context manager + close()", True)


# ── Scenario 13: UnifiedLLM sequential create/destroy ───────────────────────
def test_unified_sequential_cycles():
    """Repeated UnifiedLLM create-use-destroy cycles."""
    for _ in range(3):
        with make_unified() as llm:
            llm.generate("test", max_tokens=4)
    report("UnifiedLLM sequential create/destroy x3", True)


# ── Scenario 14: UnifiedLLM del triggers __del__ ────────────────────────────
def test_unified_del():
    """UnifiedLLM __del__ must clean up without crash."""
    llm = make_unified()
    llm.generate("test", max_tokens=4)
    del llm
    gc.collect()
    report("UnifiedLLM del (__del__ path)", True)


# ── Scenario 15: UnifiedLLM __del__ after close() ───────────────────────────
def test_unified_close_then_del():
    """close() then __del__ via GC must not double-free."""
    llm = make_unified()
    llm.generate("test", max_tokens=4)
    llm.close()
    del llm
    gc.collect()
    report("UnifiedLLM close() then del", True)


# ── Scenario 16: UnifiedLLM use-after-close ──────────────────────────────────
def test_unified_use_after_close():
    """Operations on closed UnifiedLLM must raise, not crash."""
    llm = make_unified()
    llm.close()
    ok = False
    try:
        llm.generate("Should fail", max_tokens=4)
    except LlamaError, AttributeError:
        # AttributeError because UnifiedLLM sets self.llm = None
        ok = True
    except Exception as e:
        report(
            "UnifiedLLM use-after-close", False, f"unexpected: {type(e).__name__}: {e}"
        )
        return
    report("UnifiedLLM use-after-close", ok)


# ── Scenario 17: UnifiedLLM + Llama mixed close order ───────────────────────
def test_unified_and_llama_mixed():
    """Create both UnifiedLLM and Llama, close in interleaved order."""
    u = make_unified()
    standalone = make_llm(n_gpu_layers=0)  # CPU to avoid GPU OOM with two models loaded
    # Close UnifiedLLM first (which closes its inner Llama), then standalone Llama
    u.close()
    standalone.close()
    report("UnifiedLLM + Llama mixed close", True)


# ── Scenario 18: UnifiedLLM rapid create-close ──────────────────────────────
def test_unified_rapid_create_close():
    """Rapid UnifiedLLM allocation/deallocation loop."""
    for _ in range(5):
        llm = make_unified()
        llm.close()
    gc.collect()
    report("UnifiedLLM rapid create-close x5", True)


# ── Scenario 19: UnifiedLLM access inner llm after close ────────────────────
def test_unified_inner_llm_after_close():
    """After close(), inner llm is None; accessing it must not crash."""
    llm = make_unified()
    llm.close()
    ok = llm.llm is None
    report("UnifiedLLM inner llm is None after close", ok)


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    # Catch fatal signals that indicate a double-free
    def on_fatal(signum, _frame):
        name = signal.Signals(signum).name
        print(f"\nFATAL: Caught {name} -- likely double-free or use-after-free!")
        sys.exit(128 + signum)

    for sig in (signal.SIGABRT, signal.SIGSEGV):
        signal.signal(sig, on_fatal)

    print(f"Model: {MODEL_PATH}")
    print("Running double-free verification scenarios...\n")

    try:
        make_llm().close()  # Sanity check: model loads at all
    except ModelLoadError as e:
        print(f"Cannot load model: {e}")
        print("Place a GGUF model at the expected path and re-run.")
        sys.exit(1)

    llama_scenarios = [
        test_double_close,
        test_context_manager_then_close,
        test_sequential_cycles,
        test_state_roundtrip_close,
        test_state_file_close,
        test_gc_pressure,
        test_use_after_close,
        test_multi_instance_close_order,
        test_rapid_create_close,
        test_del_without_close,
    ]

    unified_scenarios = [
        test_unified_double_close,
        test_unified_context_manager_then_close,
        test_unified_sequential_cycles,
        test_unified_del,
        test_unified_close_then_del,
        test_unified_use_after_close,
        test_unified_and_llama_mixed,
        test_unified_rapid_create_close,
        test_unified_inner_llm_after_close,
    ]

    print("── Llama ──")
    for fn in llama_scenarios:
        try:
            fn()
        except Exception as e:
            report(fn.__name__, False, f"unhandled: {type(e).__name__}: {e}")

    print("\n── UnifiedLLM ──")
    for fn in unified_scenarios:
        try:
            fn()
        except Exception as e:
            report(fn.__name__, False, f"unhandled: {type(e).__name__}: {e}")

    print(f"\nResults: {passed} passed, {failed} failed")

    # Explicit shutdown to verify final backend cleanup
    shutdown()
    print("shutdown() completed cleanly.")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
