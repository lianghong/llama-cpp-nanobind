"""Tests for locally-typical sampling (llama_sampler_init_typical)."""

import pytest

from llama_cpp.llama import SamplingParams, ValidationError

from conftest import requires_model


# --- Pure validation (no model required) ---


def test_typical_p_disabled_by_default():
    params = SamplingParams()
    assert params.typical_p == 1.0
    native = params.to_native()
    assert native.typical_p == 1.0


def test_typical_p_to_native_passes_through():
    params = SamplingParams(typical_p=0.7)
    native = params.to_native()
    assert native.typical_p == pytest.approx(0.7)


def test_typical_p_zero_rejected():
    """0.0 would discard everything; lower edge of (0, 1] is open."""
    with pytest.raises(ValidationError, match="typical_p"):
        SamplingParams(typical_p=0.0)


def test_typical_p_negative_rejected():
    with pytest.raises(ValidationError, match="typical_p"):
        SamplingParams(typical_p=-0.1)


def test_typical_p_above_one_rejected():
    with pytest.raises(ValidationError, match="typical_p"):
        SamplingParams(typical_p=1.5)


def test_typical_p_one_allowed():
    """1.0 is the disabled sentinel and must be accepted."""
    params = SamplingParams(typical_p=1.0)
    assert params.typical_p == 1.0


# --- End-to-end with a model ---


@requires_model
def test_typical_p_disabled_matches_default_path(llm):
    """typical_p = 1.0 (default sentinel) must match the no-typical-p path."""
    a = llm.generate("Hello, ", max_tokens=8, sampling=SamplingParams(seed=11))
    b = llm.generate(
        "Hello, ", max_tokens=8, sampling=SamplingParams(seed=11, typical_p=1.0)
    )
    assert a == b


@requires_model
def test_typical_p_smoke_generation(llm):
    """Enabling typical_p must produce non-empty output without crashing."""
    out = llm.generate(
        "The quick brown fox",
        max_tokens=12,
        sampling=SamplingParams(seed=42, typical_p=0.5),
    )
    assert isinstance(out, str)
    assert len(out) > 0
