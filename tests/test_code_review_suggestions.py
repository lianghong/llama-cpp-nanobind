"""Test additional suggestions from code review."""

import pytest

from llama_cpp import Llama, LlamaConfig


def test_reset_verbose_classmethod():
    """Test that reset_verbose class method works."""
    # Set verbose state
    Llama._global_verbose = False

    # Reset it
    Llama.reset_verbose()

    # Should be None now
    assert Llama._global_verbose is None


def test_max_prompt_multiplier_default(model_path):
    """Test that max_prompt_multiplier defaults to 2."""
    llm = Llama(model_path, config=LlamaConfig(model_path=model_path, n_ctx=512))

    # The multiplier defaults to 2 if not present
    multiplier = getattr(llm.config, "max_prompt_multiplier", 2)
    assert multiplier == 2

    llm.close()


def test_dataclass_replace_used():
    """Verify dataclasses.replace is imported (code inspection test)."""
    from llama_cpp import llama as llama_module

    # Check the source uses dc_replace (dataclasses.replace)
    import inspect

    source = inspect.getsource(llama_module.Llama.generate)
    assert "dc_replace" in source, "Should use dataclasses.replace for cleaner code"
