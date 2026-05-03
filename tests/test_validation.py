"""Tests for public-boundary validation helpers that don't require a model."""

import pytest

from llama_cpp.llama import Llama, SamplingParams, ValidationError


def test_validate_sampling_overrides_unknown_raises():
    """Unknown kwargs must surface as a clear ValidationError at the public
    boundary, not as a confusing TypeError deep in SamplingParams.__init__.
    """
    with pytest.raises(ValidationError, match="unknown sampling override"):
        Llama._validate_sampling_overrides({"logprobs": 5})


def test_validate_sampling_overrides_typo_raises():
    with pytest.raises(ValidationError, match="tempeature"):
        Llama._validate_sampling_overrides({"tempeature": 0.8})


def test_validate_sampling_overrides_accepts_valid():
    """Every actual SamplingParams field should pass through cleanly."""
    valid_keys = {
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 20,
        "min_p": 0.05,
        "repeat_penalty": 1.1,
        "seed": 42,
    }
    # Should not raise
    Llama._validate_sampling_overrides(valid_keys)

    # Sanity: they actually build a SamplingParams
    params = SamplingParams(**valid_keys)
    assert params.temperature == 0.5
    assert params.top_k == 20


def test_validate_sampling_overrides_empty_ok():
    Llama._validate_sampling_overrides({})


def test_validate_sampling_overrides_reports_multiple_unknowns():
    with pytest.raises(ValidationError) as excinfo:
        Llama._validate_sampling_overrides({"foo": 1, "bar": 2, "temperature": 0.5})
    msg = str(excinfo.value)
    assert "foo" in msg
    assert "bar" in msg
    # Valid key not flagged
    assert (
        "'temperature'"
        not in msg.split("unknown sampling override(s):")[1].split(".")[0]
    )
