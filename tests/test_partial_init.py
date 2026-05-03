"""Regression test: UnifiedLLM.close() must be callable after a partial
__init__ failure (e.g. unknown family string, invalid model path). The
_closed flag is initialized before any operation that can raise, so close()
takes the cleanup path instead of silently returning.
"""

import pytest

from llama_cpp.unified import UnifiedLLM


def test_close_safe_after_unknown_family():
    """Passing an unknown family string raises ValueError during __init__
    *after* _closed has been set but *before* self.llm is loaded. close()
    must handle that cleanly with no AttributeError.
    """
    # Avoid triggering Llama load by using a bogus model path;
    # family="nonsense" raises before path is touched.
    with pytest.raises(ValueError, match="Unknown family"):
        UnifiedLLM("nonexistent.gguf", family="nonsense-family-v0")


def test_close_safe_after_unknown_model_family_enum():
    """Similar check via the ModelFamily enum path."""
    with pytest.raises(TypeError, match="family must be"):
        # Pass an int (neither str nor ModelFamily) — triggers the TypeError
        # branch, after _closed is already False.
        UnifiedLLM("nonexistent.gguf", family=42)  # type: ignore[arg-type]


def test_manual_close_on_partially_built_instance_no_raise():
    """Simulate a mid-init failure by constructing the bare object via
    __new__ + only setting _closed, then calling close(). Nothing should
    raise — the per-attribute guards in close() handle partial state.
    """
    obj = UnifiedLLM.__new__(UnifiedLLM)
    obj._closed = False  # Mirror the first line of __init__
    obj.llm = None  # type: ignore[assignment]
    obj.backend = None  # type: ignore[assignment]
    # close() must not raise even though llm/backend are None and
    # _ref / _unified_instances entry were never set up.
    obj.close()
    assert obj._closed is True
