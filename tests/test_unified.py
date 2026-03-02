"""Tests for UnifiedLLM wrapper."""

import os

from conftest import MODEL_PATH
from conftest import requires_model
from llama_cpp.unified import detect_model_family
from llama_cpp.unified import MODEL_CONFIGS
from llama_cpp.unified import ModelFamily
from llama_cpp.unified import UnifiedLLM
import pytest


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
        "qwen3-instruct-2507",
        "qwen3-thinking-2507",
        "qwen3.5",
        "qwen3.5-small",
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
    assert config.repeat_penalty == 1.0  # Z.ai: must be disabled
    assert "<|endoftext|>" in config.stop_sequences
    assert "<|user|>" in config.stop_sequences
    assert "<|observation|>" in config.stop_sequences


def test_detect_model_family_qwen35():
    config = detect_model_family("models/Qwen3.5-27B-Q4_K_M.gguf")
    assert config.family == ModelFamily.QWEN3_5


def test_detect_model_family_qwen3_not_qwen35():
    """Qwen3 filenames must NOT match qwen3.5 config."""
    config = detect_model_family("models/Qwen3-8B-Q6_K.gguf")
    assert config.family == ModelFamily.QWEN3


def test_qwen35_config_values():
    """Verify Qwen3.5 config has correct params from model card."""
    config = MODEL_CONFIGS["qwen3.5"]
    assert config.family == ModelFamily.QWEN3_5
    assert config.supports_thinking is True
    assert config.temperature == 1.0
    assert config.top_p == 0.95
    assert config.top_k == 20
    assert config.presence_penalty == 1.5
    assert config.max_ctx == 262144
    assert "<|im_end|>" in config.stop_sequences
    assert "<|endoftext|>" in config.stop_sequences


def test_qwen35_no_think_suffix():
    """Qwen3.5 must NOT append /think or /nothink suffixes."""
    config = MODEL_CONFIGS["qwen3.5"]
    # The _build_messages guard only triggers for ModelFamily.QWEN3,
    # so QWEN3_5 naturally skips the /think suffix
    assert config.family != ModelFamily.QWEN3


def test_detect_ignores_directory_names():
    """Directory names must not trigger false-positive detection."""
    with pytest.raises(ValueError):
        detect_model_family("/home/user/phi-experiments/llama-model.gguf")
    with pytest.raises(ValueError):
        detect_model_family("/data/qwen3-finetuning/my_custom_model.gguf")


def test_detect_ministral():
    """Test Ministral model detection."""
    config = detect_model_family("models/Ministral-3-14B-Reasoning-2512-Q6_K.gguf")
    assert config.family == ModelFamily.MISTRAL


def test_detect_qwen3_instruct_2507():
    """Qwen3-Instruct-2507 should be non-thinking with Instruct defaults."""
    config = detect_model_family("models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf")
    assert config.family == ModelFamily.QWEN3
    assert config.supports_thinking is False
    assert config.temperature == 0.7
    assert config.top_p == 0.8
    assert config.top_k == 20
    assert config.presence_penalty == 1.0


def test_detect_qwen3_thinking_2507():
    """Qwen3-Thinking-2507 should be thinking-enabled with Thinking defaults."""
    config = detect_model_family("models/Qwen3-8B-Thinking-2507-Q6_K.gguf")
    assert config.family == ModelFamily.QWEN3
    assert config.supports_thinking is True
    assert config.temperature == 0.6
    assert config.top_p == 0.95
    assert config.top_k == 20
    assert config.presence_penalty == 1.0


def test_qwen3_2507_configs_in_model_configs():
    """Verify both 2507 configs are registered."""
    assert "qwen3-instruct-2507" in MODEL_CONFIGS
    assert "qwen3-thinking-2507" in MODEL_CONFIGS


def test_detect_qwen35_small_9b():
    """Qwen3.5-9B should detect as small (thinking disabled)."""
    config = detect_model_family("models/Qwen3.5-9B-Q6_K.gguf")
    assert config.family == ModelFamily.QWEN3_5
    assert config.supports_thinking is False
    assert config.temperature == 0.7
    assert config.top_p == 0.8


def test_detect_qwen35_small_sizes():
    """All Qwen3.5 small sizes (0.8B, 2B, 4B, 9B) should detect as small."""
    for size in ["0.8B", "2B", "4B", "9B"]:
        config = detect_model_family(f"models/Qwen3.5-{size}-Q4_K_M.gguf")
        assert config.supports_thinking is False, (
            f"Qwen3.5-{size} should be non-thinking"
        )


def test_detect_qwen35_large_still_thinking():
    """Large Qwen3.5 models (27B, 35B, 122B, 397B) keep thinking enabled."""
    for name in [
        "Qwen3.5-27B-Q4_K_M.gguf",
        "Qwen3.5-35B-A3B-Q4_K_M.gguf",
        "Qwen3.5-122B-A10B-Q4_K_M.gguf",
        "Qwen3.5-397B-A17B-Q4_K_M.gguf",
    ]:
        config = detect_model_family(f"models/{name}")
        assert config.supports_thinking is True, f"{name} should have thinking"


def test_qwen35_repeat_penalty_disabled():
    """Qwen3.5 configs should have repeat_penalty=1.0 (disabled per Unsloth guide)."""
    assert MODEL_CONFIGS["qwen3.5"].repeat_penalty == 1.0
    assert MODEL_CONFIGS["qwen3.5-small"].repeat_penalty == 1.0


def test_qwen35_config_presence_penalty():
    """Both Qwen3.5 configs should have presence_penalty=1.5."""
    assert MODEL_CONFIGS["qwen3.5"].presence_penalty == 1.5
    assert MODEL_CONFIGS["qwen3.5-small"].presence_penalty == 1.5


def test_gpt_oss_config_values():
    """Verify GPT-OSS config matches OpenAI recommended settings."""
    config = MODEL_CONFIGS["gpt-oss"]
    assert config.family == ModelFamily.GPT_OSS
    assert config.temperature == 1.0
    assert config.top_p == 1.0
    assert config.top_k == 0  # OpenAI: disable top-k
    assert config.min_p == 0.0
    assert config.max_ctx == 131072  # 128K context window
    assert config.supports_thinking is True
    assert config.repeat_penalty == 1.0  # RL-trained, disable penalty


def test_gpt_oss_stop_sequences():
    """GPTOSSBackend must include <|return|> EOS token."""
    from llama_cpp.unified import GPTOSSBackend

    assert "<|return|>" in GPTOSSBackend.STOP
    assert "<|start|>user" in GPTOSSBackend.STOP
    assert "<|end|><|end|>" in GPTOSSBackend.STOP


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
