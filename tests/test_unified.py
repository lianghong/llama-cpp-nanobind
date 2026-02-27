"""Tests for UnifiedLLM wrapper."""

import os

import pytest
from conftest import MODEL_PATH, requires_model

from llama_cpp.unified import (
    MODEL_CONFIGS,
    ModelFamily,
    UnifiedLLM,
    detect_model_family,
)


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


def test_detect_model_family_aya():
    config = detect_model_family("models/tiny-aya-global-q8_0.gguf")
    assert config.family == ModelFamily.AYA


def test_detect_model_family_glm47():
    config = detect_model_family("models/GLM-4.7-Flash-REAP-23B-A3B-Q4_K_M.gguf")
    assert config.family == ModelFamily.GLM4
    assert config.supports_thinking is True


def test_detect_model_family_glm4_legacy():
    """Older GLM-4 models still match glm-4 config."""
    config = detect_model_family("models/glm-4-9b-chat-Q6_K.gguf")
    assert config.family == ModelFamily.GLM4
    assert config.supports_thinking is False


def test_detect_model_family_unknown():
    with pytest.raises(ValueError) as exc_info:
        detect_model_family("unknown_model.gguf")
    assert "Supported:" in str(exc_info.value)


def test_model_configs_exist():
    """Verify all expected model families have configs."""
    expected = [
        "aya",
        "gemma",
        "glm-4",
        "glm-4.7",
        "granite",
        "minicpm",
        "ministral-instruct",
        "ministral-reasoning",
        "phi-4",
        "qwen3",
        "gpt-oss",
        "translategemma",
    ]
    for key in expected:
        assert key in MODEL_CONFIGS


def test_aya_config_values():
    """Verify Aya config has correct stop sequences and params."""
    config = MODEL_CONFIGS["aya"]
    assert config.temperature == 0.3
    assert config.top_p == 0.95
    assert "<|END_OF_TURN_TOKEN|>" in config.stop_sequences
    assert "<|END_RESPONSE|>" in config.stop_sequences
    assert config.supports_thinking is False


def test_glm47_config_values():
    """Verify GLM-4.7 config has thinking support and correct params."""
    config = MODEL_CONFIGS["glm-4.7"]
    assert config.supports_thinking is True
    assert config.temperature == 1.0
    assert config.top_p == 0.95
    assert config.min_p == 0.01
    assert "<|endoftext|>" in config.stop_sequences
    assert "<|user|>" in config.stop_sequences
    assert "<|observation|>" in config.stop_sequences


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
