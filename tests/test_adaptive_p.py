"""Tests for adaptive-p sampler binding (llama.cpp PR #17927) and new ggml_type constants."""

import pytest

import llama_cpp
from llama_cpp.llama import SamplingParams, ValidationError, _VALID_CACHE_TYPES

from conftest import requires_model


# --- Pure validation (no model required) ---


def test_adaptive_p_disabled_by_default():
    params = SamplingParams()
    assert params.adaptive_p_target == -1.0
    assert params.adaptive_p_decay == 0.85


def test_adaptive_p_to_native_passes_through():
    params = SamplingParams(adaptive_p_target=0.1, adaptive_p_decay=0.9)
    native = params.to_native()
    assert native.adaptive_p_target == pytest.approx(0.1)
    assert native.adaptive_p_decay == pytest.approx(0.9)


def test_adaptive_p_negative_target_disables():
    """Negative target disables — must not raise even with arbitrary decay."""
    params = SamplingParams(adaptive_p_target=-0.5, adaptive_p_decay=0.5)
    native = params.to_native()
    assert native.adaptive_p_target < 0


def test_adaptive_p_target_above_one_rejected():
    with pytest.raises(ValidationError, match="adaptive_p_target"):
        SamplingParams(adaptive_p_target=1.5)


def test_adaptive_p_decay_out_of_range_rejected():
    with pytest.raises(ValidationError, match="adaptive_p_decay"):
        SamplingParams(adaptive_p_decay=1.0)
    with pytest.raises(ValidationError, match="adaptive_p_decay"):
        SamplingParams(adaptive_p_decay=-0.1)


def test_adaptive_p_target_zero_allowed():
    """Target = 0 is the lower edge of [0, 1] and must be accepted."""
    params = SamplingParams(adaptive_p_target=0.0)
    native = params.to_native()
    assert native.adaptive_p_target == 0.0


def test_adaptive_p_target_one_allowed():
    """Target = 1.0 is the upper edge of [0, 1] and must be accepted."""
    params = SamplingParams(adaptive_p_target=1.0)
    native = params.to_native()
    assert native.adaptive_p_target == pytest.approx(1.0)


# --- ggml_type constants (forward-compat exports) ---


def test_new_ggml_type_constants_exposed():
    """MXFP4/NVFP4/Q1_0 must be importable from the package root."""
    assert llama_cpp.GGML_TYPE_MXFP4 == 39
    assert llama_cpp.GGML_TYPE_NVFP4 == 40
    assert llama_cpp.GGML_TYPE_Q1_0 == 41


def test_new_ggml_types_NOT_in_kv_cache_whitelist():
    """MXFP4/NVFP4/Q1_0 are weight-only; rejecting them for KV cache is the
    safe default until upstream llama.cpp supports them in llama_kv_cache_unified.
    """
    assert llama_cpp.GGML_TYPE_MXFP4 not in _VALID_CACHE_TYPES
    assert llama_cpp.GGML_TYPE_NVFP4 not in _VALID_CACHE_TYPES
    assert llama_cpp.GGML_TYPE_Q1_0 not in _VALID_CACHE_TYPES


# --- End-to-end with a model ---


@requires_model
def test_adaptive_p_generation_smoke(llm):
    """Adaptive-p enabled must produce non-empty output without crashing.

    Adaptive-p is a terminal sampler; passed through the explicit
    SamplingParams override path (Llama.generate's `sampling` kwarg).
    """
    sampling = SamplingParams(
        adaptive_p_target=0.1,
        adaptive_p_decay=0.85,
        seed=42,
    )
    out = llm.generate(
        "The quick brown fox",
        max_tokens=12,
        sampling=sampling,
    )
    assert isinstance(out, str)
    assert len(out) > 0


@requires_model
def test_adaptive_p_disabled_matches_normal_path(llm):
    """Negative target = disabled. Output should follow the normal dist sampler."""
    sampling_disabled = SamplingParams(adaptive_p_target=-1.0, seed=123)
    sampling_default = SamplingParams(seed=123)
    out_disabled = llm.generate("Hello, ", max_tokens=8, sampling=sampling_disabled)
    out_default = llm.generate("Hello, ", max_tokens=8, sampling=sampling_default)
    assert out_disabled == out_default
