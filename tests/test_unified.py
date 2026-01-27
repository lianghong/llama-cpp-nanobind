"""Tests for UnifiedLLM wrapper."""

import os

import pytest

from llama_cpp.unified import (
    MODEL_CONFIGS,
    ModelFamily,
    UnifiedLLM,
    detect_model_family,
)

from conftest import MODEL_PATH, requires_model


# Model family detection tests (no model loading required)
def test_detect_model_family_qwen():
    config = detect_model_family("models/Qwen3-8B-Q6_K.gguf")
    assert config.family == ModelFamily.QWEN3


def test_detect_model_family_gemma():
    config = detect_model_family("models/gemma-2-9b-it-Q6_K.gguf")
    assert config.family == ModelFamily.GEMMA


def test_detect_model_family_gpt_oss():
    config = detect_model_family("models/gpt-oss-20b-Q4_K_M.gguf")
    assert config.family == ModelFamily.GPT_OSS


def test_detect_model_family_phi():
    config = detect_model_family("models/phi-4-Q6_K.gguf")
    assert config.family == ModelFamily.PHI


def test_detect_model_family_unknown():
    with pytest.raises(ValueError) as exc_info:
        detect_model_family("unknown_model.gguf")
    assert "Supported:" in str(exc_info.value)


def test_model_configs_exist():
    """Verify all expected model families have configs."""
    expected = [
        "aya",
        "gemma",
        "granite",
        "minicpm",
        "ministral-instruct",
        "ministral-reasoning",
        "phi-4",
        "qwen3",
        "gpt-oss",
    ]
    for key in expected:
        assert key in MODEL_CONFIGS


def test_detect_ministral():
    """Test Ministral model detection."""
    config = detect_model_family("models/Ministral-3-14B-Reasoning-2512-Q6_K.gguf")
    assert config.family == ModelFamily.MISTRAL


# Integration tests (require model)
@pytest.fixture(scope="module")
def unified_llm():
    """Shared UnifiedLLM instance for tests."""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("test model not found")
    instance = UnifiedLLM(MODEL_PATH, verbose=False)
    yield instance
    instance.close()


@requires_model
def test_unified_llm_family(unified_llm):
    assert unified_llm.family == ModelFamily.QWEN3


@requires_model
def test_unified_llm_generate(unified_llm):
    response = unified_llm.generate("Hello", max_tokens=32)
    assert isinstance(response, str)


@requires_model
def test_unified_llm_context_manager():
    """Test context manager protocol."""
    with UnifiedLLM(MODEL_PATH, verbose=False) as llm:
        response = llm.generate("Hi", max_tokens=5)
        assert isinstance(response, str)
    assert llm.llm is None
    assert llm.backend is None


@requires_model
def test_unified_llm_close():
    """Test explicit close() method."""
    llm = UnifiedLLM(MODEL_PATH, verbose=False)
    assert llm.llm is not None
    assert llm.backend is not None
    llm.close()
    assert llm.llm is None
    assert llm.backend is None


@requires_model
def test_unified_llm_double_close():
    """Test that calling close() twice is safe."""
    llm = UnifiedLLM(MODEL_PATH, verbose=False)
    llm.close()
    llm.close()
    assert llm.llm is None


@requires_model
def test_unified_llm_kv_cache_clear():
    """Test kv_cache_clear works correctly."""
    with UnifiedLLM(MODEL_PATH, verbose=False) as llm:
        llm.generate("Hello", max_tokens=5)
        llm.kv_cache_clear()
        response = llm.generate("World", max_tokens=5)
        assert isinstance(response, str)


@requires_model
def test_unified_llm_invalid_max_tokens():
    """Test that invalid max_tokens raises ValueError."""
    with UnifiedLLM(MODEL_PATH, verbose=False) as llm:
        with pytest.raises(ValueError, match="must be positive"):
            llm.backend._calc_max_tokens("test", 0)
        with pytest.raises(ValueError, match="must be positive"):
            llm.backend._calc_max_tokens("test", -5)
