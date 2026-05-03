"""Test close() exception safety and resource cleanup.

Note: Direct mocking of nanobind C++ methods is not possible due to read-only attributes.
These tests verify the close() logic without artificial exception injection.
"""

from llama_cpp import Llama, LlamaConfig


def test_close_normal_path_works(model_path):
    """Test that close() works normally without exceptions."""
    llm = Llama(model_path, config=LlamaConfig(model_path=model_path, n_ctx=512))

    # Normal close should not raise
    llm.close()

    assert llm._closed is True
    assert llm.ctx is None
    assert llm.model is None


def test_close_is_idempotent(model_path):
    """Test that close() can be called multiple times safely."""
    llm = Llama(model_path, config=LlamaConfig(model_path=model_path, n_ctx=512))

    # First close
    llm.close()
    assert llm._closed is True

    # Second close should not raise
    llm.close()
    assert llm._closed is True


def test_close_in_context_manager(model_path):
    """Test that close() works correctly with context manager."""
    with Llama(model_path, config=LlamaConfig(model_path=model_path, n_ctx=512)) as llm:
        assert not llm._closed

    # After exiting context, should be closed
    assert llm._closed
    assert llm.ctx is None
    assert llm.model is None


def test_close_clears_lora_tracking(model_path):
    """Test that close() clears internal LoRA adapter tracking."""
    llm = Llama(model_path, config=LlamaConfig(model_path=model_path, n_ctx=512))

    # Simulate adapter tracking (internal state)
    llm._lora_adapters = [("adapter1", 1.0), ("adapter2", 0.5)]

    llm.close()

    # Adapters list should be cleared
    assert len(llm._lora_adapters) == 0
    assert llm._closed is True
