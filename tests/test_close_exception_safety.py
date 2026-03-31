"""Test close() exception safety and resource cleanup."""

import pytest
from unittest.mock import Mock, patch

from llama_cpp import Llama, LlamaConfig, LlamaError


def test_close_handles_ctx_close_exception(model_path):
    """Test that close() handles exceptions from ctx.close() gracefully."""
    llm = Llama(model_path, config=LlamaConfig(model_path=model_path, n_ctx=512))

    # Mock ctx.close() to raise an exception
    original_ctx_close = llm.ctx.close
    llm.ctx.close = Mock(side_effect=RuntimeError("ctx close failed"))

    # close() should catch the exception and still mark instance as closed
    with pytest.raises(LlamaError, match="Errors during close"):
        llm.close()

    # Instance should be marked closed despite exception
    assert llm._closed is True
    # ctx should be set to None even after exception
    assert llm.ctx is None
    # model should still be closed
    assert llm.model is None


def test_close_handles_model_close_exception(model_path):
    """Test that close() handles exceptions from model.close() gracefully."""
    llm = Llama(model_path, config=LlamaConfig(model_path=model_path, n_ctx=512))

    # Mock model.close() to raise an exception
    original_model_close = llm.model.close
    llm.model.close = Mock(side_effect=RuntimeError("model close failed"))

    # close() should catch the exception and still mark instance as closed
    with pytest.raises(LlamaError, match="Errors during close"):
        llm.close()

    # Instance should be marked closed despite exception
    assert llm._closed is True
    # ctx should be closed successfully
    assert llm.ctx is None
    # model should be set to None even after exception
    assert llm.model is None


def test_close_handles_both_exceptions(model_path):
    """Test that close() handles exceptions from both ctx and model close."""
    llm = Llama(model_path, config=LlamaConfig(model_path=model_path, n_ctx=512))

    # Mock both to raise exceptions
    llm.ctx.close = Mock(side_effect=RuntimeError("ctx close failed"))
    llm.model.close = Mock(side_effect=RuntimeError("model close failed"))

    # close() should catch both exceptions and report the first one
    with pytest.raises(LlamaError, match="Errors during close.*ctx close failed"):
        llm.close()

    # Instance should be marked closed despite exceptions
    assert llm._closed is True
    # Both should be set to None
    assert llm.ctx is None
    assert llm.model is None


def test_close_idempotent_after_exception(model_path):
    """Test that close() is idempotent even after an exception."""
    llm = Llama(model_path, config=LlamaConfig(model_path=model_path, n_ctx=512))

    # Mock ctx.close() to raise on first call
    call_count = 0

    def raise_once(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("First call fails")

    llm.ctx.close = Mock(side_effect=raise_once)

    # First close() raises
    with pytest.raises(LlamaError):
        llm.close()

    assert llm._closed is True
    assert llm.ctx is None

    # Second close() should return early (idempotent)
    llm.close()  # Should not raise

    # ctx.close was only called once (not called again on second close())
    assert call_count == 1


def test_close_normal_path_still_works(model_path):
    """Test that close() still works normally without exceptions."""
    llm = Llama(model_path, config=LlamaConfig(model_path=model_path, n_ctx=512))

    # Normal close should not raise
    llm.close()

    assert llm._closed is True
    assert llm.ctx is None
    assert llm.model is None

    # Second close should be idempotent
    llm.close()


def test_close_clears_lora_adapters(model_path):
    """Test that close() clears LoRA adapters even on exception."""
    llm = Llama(model_path, config=LlamaConfig(model_path=model_path, n_ctx=512))

    # Add mock adapter tracking
    llm._lora_adapters = ["adapter1", "adapter2"]

    # Mock ctx.close() to raise
    llm.ctx.close = Mock(side_effect=RuntimeError("close failed"))

    # close() should still clear adapters
    with pytest.raises(LlamaError):
        llm.close()

    assert len(llm._lora_adapters) == 0
