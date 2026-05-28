"""Tests for UnifiedLLM (Qwen 3.5, Qwen 3.6, Gemma 4, IBM Granite 4.1)."""

import os

from conftest import MODEL_PATH
from conftest import requires_model
from llama_cpp.unified import detect_model_family
from llama_cpp.unified import MODEL_CONFIGS
from llama_cpp.unified import ModelFamily
from llama_cpp.unified import UnifiedLLM
from llama_cpp.unified import UnsupportedModelError
import pytest


# ---------------------------------------------------------------------------
# Family enum membership — guards against accidental re-introduction of the
# legacy families we explicitly dropped.
# ---------------------------------------------------------------------------


def test_supported_families_only():
    """ModelFamily must contain exactly the four supported architectures."""
    assert {m.name for m in ModelFamily} == {
        "QWEN3_5",
        "QWEN3_6",
        "GEMMA4",
        "GRANITE",
    }


def test_model_configs_keys():
    """MODEL_CONFIGS must contain only presets for supported families."""
    expected = {
        "qwen3.5",
        "qwen3.5-small",
        "qwen3.5-coding",
        "qwen3.6",
        "qwen3.6-coding",
        "qwen3.6-instruct",
        "qwen3.6-instruct-reasoning",
        "gemma-4",
        "gemma-4-large",
        "granite",
    }
    assert set(MODEL_CONFIGS.keys()) == expected


# ---------------------------------------------------------------------------
# Filename detection
# ---------------------------------------------------------------------------


def test_detect_qwen35_default_thinking():
    config = detect_model_family("models/Qwen3.5-27B-Q4_K_M.gguf")
    assert config.family == ModelFamily.QWEN3_5
    assert config.supports_thinking is True


def test_detect_qwen35_small_sizes_skip_thinking():
    """0.8B / 2B / 4B / 9B Qwen 3.5 default to thinking off per Unsloth."""
    for size in ("0.8B", "2B", "4B", "9B"):
        config = detect_model_family(f"models/Qwen3.5-{size}-Q4_K_M.gguf")
        assert config.supports_thinking is False, (
            f"Qwen3.5-{size} should be non-thinking"
        )
        assert config.temperature == 0.7
        assert config.top_p == 0.8


def test_detect_qwen35_large_keeps_thinking():
    """27B / 35B-A3B / 122B-A10B / 397B-A17B keep thinking on."""
    for name in (
        "Qwen3.5-27B-Q4_K_M.gguf",
        "Qwen3.5-35B-A3B-Q4_K_M.gguf",
        "Qwen3.5-122B-A10B-Q4_K_M.gguf",
        "Qwen3.5-397B-A17B-Q4_K_M.gguf",
    ):
        config = detect_model_family(f"models/{name}")
        assert config.supports_thinking is True, f"{name} should have thinking"


def test_detect_qwen36_default_thinking():
    config = detect_model_family("models/Qwen3.6-27B-Q4_K_M.gguf")
    assert config.family == ModelFamily.QWEN3_6
    assert config.supports_thinking is True
    assert config.temperature == 1.0


def test_detect_qwen36_instruct():
    """Qwen 3.6 *Instruct* filenames pick the non-thinking instruct preset."""
    config = detect_model_family("models/Qwen3.6-35B-A3B-Instruct-Q4_K_M.gguf")
    assert config.family == ModelFamily.QWEN3_6
    assert config.supports_thinking is False
    assert config.temperature == 0.7  # general-instruct preset


def test_detect_qwen36_instruct_reasoning():
    """Filenames with both 'Instruct' and 'Reasoning' pick the higher-temp preset."""
    config = detect_model_family(
        "models/Qwen3.6-35B-A3B-Instruct-Reasoning-Q4_K_M.gguf"
    )
    assert config.family == ModelFamily.QWEN3_6
    assert config.supports_thinking is False
    assert config.temperature == 1.0


def test_detect_gemma4_small_128k():
    config = detect_model_family("models/gemma-4-e4b-it-Q8_0.gguf")
    assert config.family == ModelFamily.GEMMA4
    assert config.max_ctx == 131072
    assert config.supports_thinking is True


def test_detect_gemma4_large_256k():
    for name in (
        "gemma-4-26b-a4b-it-Q4_K_XL.gguf",
        "gemma-4-31b-it-Q4_K_XL.gguf",
    ):
        config = detect_model_family(f"models/{name}")
        assert config.family == ModelFamily.GEMMA4
        assert config.max_ctx == 262144


def test_detect_granite_4_1():
    config = detect_model_family("models/granite-4.1-3b-Q6_K.gguf")
    assert config.family == ModelFamily.GRANITE
    assert config.temperature == 0.0
    assert config.top_p == 1.0
    assert config.top_k == 0
    assert config.supports_thinking is False


# ---------------------------------------------------------------------------
# Negative detection — unsupported families must raise loudly.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename",
    [
        # Removed families
        "Qwen3-8B-Q6_K.gguf",
        "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf",
        "gemma-2-9b-it-Q6_K.gguf",
        "gemma-3-27b-it-Q4_K_M.gguf",
        "GLM-4.7-Flash-23B-Q4_K_M.gguf",
        "glm-4-9b-chat-Q6_K.gguf",
        "MiniCPM-3-4B-Q6_K.gguf",
        "Phi-4-Q6_K.gguf",
        "Mistral-Small-Q5_K_M.gguf",
        "Ministral-3-14B-Reasoning-Q6_K.gguf",
        "gpt-oss-20b-Q4_K_M.gguf",
        # Generic / unknown
        "llama-3.1-8B-instruct-Q5_K_M.gguf",
        "unknown_model.gguf",
        # Granite 3.x is not in scope (only 4.1)
        "granite-3.2-8b-Q6_K.gguf",
    ],
)
def test_unsupported_models_raise(filename):
    with pytest.raises(UnsupportedModelError) as exc_info:
        detect_model_family(f"models/{filename}")
    msg = str(exc_info.value)
    assert "Qwen 3.5" in msg
    assert "Granite 4.1" in msg


def test_detect_ignores_directory_names():
    """Directory names must not trigger false-positive detection."""
    with pytest.raises(UnsupportedModelError):
        detect_model_family("/home/user/qwen-finetuning/llama-model.gguf")
    with pytest.raises(UnsupportedModelError):
        detect_model_family("/data/gemma-2-experiments/my_custom_model.gguf")


# ---------------------------------------------------------------------------
# Preset values track Unsloth recipes — these are regression guards.
# ---------------------------------------------------------------------------


def test_qwen35_default_preset():
    cfg = MODEL_CONFIGS["qwen3.5"]
    assert cfg.family == ModelFamily.QWEN3_5
    assert cfg.temperature == 1.0
    assert cfg.top_p == 0.95
    assert cfg.top_k == 20
    assert cfg.min_p == 0.0
    assert cfg.presence_penalty == 1.5
    assert cfg.repeat_penalty == 1.0  # Unsloth: disabled
    assert cfg.max_ctx == 262144


def test_qwen35_coding_preset():
    cfg = MODEL_CONFIGS["qwen3.5-coding"]
    assert cfg.temperature == 0.6
    assert cfg.presence_penalty == 0.0
    assert cfg.repeat_penalty == 1.0


def test_qwen36_default_preset():
    cfg = MODEL_CONFIGS["qwen3.6"]
    assert cfg.family == ModelFamily.QWEN3_6
    assert cfg.temperature == 1.0
    assert cfg.top_p == 0.95
    assert cfg.top_k == 20
    assert cfg.presence_penalty == 1.5


def test_qwen36_instruct_presets():
    """Qwen 3.6 instruct presets follow Unsloth's general/reasoning split."""
    instr = MODEL_CONFIGS["qwen3.6-instruct"]
    assert instr.temperature == 0.7
    assert instr.top_p == 0.8
    assert instr.supports_thinking is False

    reason = MODEL_CONFIGS["qwen3.6-instruct-reasoning"]
    assert reason.temperature == 1.0
    assert reason.top_p == 0.95
    assert reason.supports_thinking is False


def test_gemma4_presets():
    small = MODEL_CONFIGS["gemma-4"]
    assert small.temperature == 1.0
    assert small.top_p == 0.95
    assert small.top_k == 64
    assert small.repeat_penalty == 1.0
    assert small.max_ctx == 131072
    assert "<turn|>" in small.stop_sequences

    large = MODEL_CONFIGS["gemma-4-large"]
    assert large.max_ctx == 262144


def test_granite_preset_deterministic():
    cfg = MODEL_CONFIGS["granite"]
    assert cfg.family == ModelFamily.GRANITE
    assert cfg.temperature == 0.0
    assert cfg.top_p == 1.0
    assert cfg.top_k == 0
    assert cfg.repeat_penalty == 1.0
    assert cfg.max_ctx == 131072


# ---------------------------------------------------------------------------
# Integration tests — require a real model.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def unified_llm():
    if not os.path.exists(MODEL_PATH):
        pytest.skip("test model not found")
    instance = UnifiedLLM(MODEL_PATH, verbose=False)
    yield instance
    instance.close()


@requires_model
def test_unified_llm_family_detected(unified_llm):
    """Whatever the test model is, family must be one of the supported four."""
    assert unified_llm.family in {
        ModelFamily.QWEN3_5,
        ModelFamily.QWEN3_6,
        ModelFamily.GEMMA4,
        ModelFamily.GRANITE,
    }


@requires_model
def test_unified_llm_generate(unified_llm):
    response = unified_llm.generate("Hello", max_tokens=32)
    assert isinstance(response, str)


@requires_model
def test_unified_llm_chat_basic(unified_llm):
    """chat() returns a non-empty assistant response and strips thinking."""
    messages = [{"role": "user", "content": "Say hi in one word."}]
    response = unified_llm.chat(messages, max_tokens=16)
    assert isinstance(response, str)
    # Thinking blocks must not leak through.
    assert "<think>" not in response
    assert "<|channel>" not in response


@requires_model
def test_unified_llm_chat_streaming(unified_llm):
    """chat(stream=True) yields raw text deltas as an iterator."""
    messages = [{"role": "user", "content": "Count from one to three."}]
    stream = unified_llm.chat(messages, max_tokens=24, stream=True)
    # Must be an iterator (not a string), and yield at least one non-empty chunk.
    assert not isinstance(stream, str)
    chunks = list(stream)
    assert len(chunks) >= 1
    assert all(isinstance(c, str) for c in chunks)
    joined = "".join(chunks)
    assert joined  # non-empty
    # Stream is *raw* — caller is responsible for stripping thinking blocks.
    # We don't assert their absence here; just that we got text.


@requires_model
def test_unified_llm_chat_history_sanitization(unified_llm):
    """Prior assistant turns with thinking blocks are stripped automatically."""
    messages = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": "<think>working on it</think>Hi!",
        },
        {"role": "user", "content": "Who are you?"},
    ]
    # Should run without raising; sanitize_history defaults True for thinking
    # families.  We can't assert what the model returns, only that the call
    # completes and doesn't choke on the embedded thinking block.
    response = unified_llm.chat(messages, max_tokens=24)
    assert isinstance(response, str)


@requires_model
def test_unified_llm_context_manager():
    with UnifiedLLM(MODEL_PATH, verbose=False) as llm:
        response = llm.generate("Hi", max_tokens=5)
        assert isinstance(response, str)
    assert llm.llm is None
    assert llm.backend is None


@requires_model
def test_unified_llm_close_idempotent():
    llm = UnifiedLLM(MODEL_PATH, verbose=False)
    assert llm.llm is not None
    llm.close()
    llm.close()  # safe to call twice
    assert llm.llm is None


@requires_model
def test_unified_llm_kv_cache_clear():
    with UnifiedLLM(MODEL_PATH, verbose=False) as llm:
        llm.generate("Hello", max_tokens=5)
        llm.kv_cache_clear()
        response = llm.generate("World", max_tokens=5)
        assert isinstance(response, str)


@requires_model
def test_unified_llm_invalid_max_tokens():
    with UnifiedLLM(MODEL_PATH, verbose=False) as llm:
        with pytest.raises(ValueError, match="must be positive"):
            llm.backend._calc_max_tokens("test", 0)
        with pytest.raises(ValueError, match="must be positive"):
            llm.backend._calc_max_tokens("test", -5)


# ---------------------------------------------------------------------------
# Speculative decoding (Option B: auto-detect from MTP capability).
# Pure tests run against the regular test model, which doesn't expose an
# MTP graph — the auto path therefore resolves to disabled, and force=True
# must raise.
# ---------------------------------------------------------------------------


@requires_model
def test_unified_llm_speculative_auto_disabled_on_non_mtp():
    """speculative='auto' on a non-MTP checkpoint must resolve to False."""
    with UnifiedLLM(MODEL_PATH, verbose=False) as llm:
        assert llm.speculative_enabled is False
        # _sampling_kwargs must NOT inject speculative when disabled.
        kwargs = llm.backend._sampling_kwargs(None)
        assert "speculative" not in kwargs
        assert "n_draft_max" not in kwargs


@requires_model
def test_unified_llm_speculative_force_true_rejects_non_mtp():
    """speculative=True on a non-MTP checkpoint must raise at construction."""
    with pytest.raises(ValueError, match="MTP graph"):
        UnifiedLLM(MODEL_PATH, verbose=False, speculative=True)


@requires_model
def test_unified_llm_speculative_explicit_false():
    """speculative=False must disable even if MTP would be available."""
    with UnifiedLLM(MODEL_PATH, verbose=False, speculative=False) as llm:
        assert llm.speculative_enabled is False


@requires_model
def test_unified_llm_speculative_invalid_mode():
    """Garbage values must raise a clear error."""
    with pytest.raises(ValueError, match="speculative must be"):
        UnifiedLLM(MODEL_PATH, verbose=False, speculative="yes")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Positive speculative path lives in tests/test_unified_speculative.py — that
# file is isolated so the module-scoped `unified_llm` fixture above does not
# hold the small test model in VRAM while the large MTP model is loaded.
# ---------------------------------------------------------------------------
