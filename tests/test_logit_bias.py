"""Tests for logit_bias on SamplingParams (llama_sampler_init_logit_bias)."""

import math

import pytest

from llama_cpp.llama import SamplingParams, ValidationError

from conftest import requires_model


# --- Pure validation (no model required) ---


def test_logit_bias_disabled_by_default():
    params = SamplingParams()
    assert params.logit_bias is None
    native = params.to_native()
    assert list(native.logit_bias) == []


def test_logit_bias_dict_round_trips_to_native():
    params = SamplingParams(logit_bias={1: -10.0, 5: 2.5})
    native = params.to_native()
    # Order is dict-iteration-order (insertion order on CPython 3.7+).
    assert sorted(native.logit_bias) == [(1, -10.0), (5, 2.5)]


def test_logit_bias_empty_dict_is_disabled():
    """Empty dict should round-trip to an empty list (sampler not added)."""
    params = SamplingParams(logit_bias={})
    native = params.to_native()
    assert list(native.logit_bias) == []


def test_logit_bias_neg_inf_for_ban_accepted():
    """Banning a token uses -inf as the bias — must pass validation."""
    params = SamplingParams(logit_bias={42: float("-inf")})
    native = params.to_native()
    assert native.logit_bias[0][0] == 42
    assert math.isinf(native.logit_bias[0][1]) and native.logit_bias[0][1] < 0


def test_logit_bias_negative_token_id_rejected():
    with pytest.raises(ValidationError, match="logit_bias token id"):
        SamplingParams(logit_bias={-1: 1.0})


def test_logit_bias_nan_rejected():
    with pytest.raises(ValidationError, match="logit_bias"):
        SamplingParams(logit_bias={1: float("nan")})


def test_logit_bias_non_numeric_rejected():
    with pytest.raises(ValidationError, match="logit_bias"):
        SamplingParams(logit_bias={1: "high"})  # type: ignore[dict-item]


# --- End-to-end with a model ---


@requires_model
def test_logit_bias_does_not_affect_default_path(llm):
    """Empty / unset logit_bias must produce identical output to baseline."""
    p_none = SamplingParams(seed=123)
    p_empty = SamplingParams(seed=123, logit_bias={})
    out_none = llm.generate("Once upon", max_tokens=8, sampling=p_none)
    out_empty = llm.generate("Once upon", max_tokens=8, sampling=p_empty)
    assert out_none == out_empty


@requires_model
def test_logit_bias_changes_sampling_decisions(llm):
    """Heavily biasing a single token away from the baseline pick should
    change the generated output. Strategy: capture the first sampled token
    of a baseline run and re-run with that token banned via -inf bias.
    """
    seed = 42
    prompt = "The capital of France is"
    baseline = llm.generate(prompt, max_tokens=1, sampling=SamplingParams(seed=seed))
    assert baseline  # non-empty

    # Tokenize the baseline single-token output and ban it.
    banned_ids = llm.tokenize(baseline, add_special=False)
    assert banned_ids, "baseline produced no tokens — fixture issue"
    biased = SamplingParams(
        seed=seed, logit_bias={tok: float("-inf") for tok in banned_ids}
    )
    altered = llm.generate(prompt, max_tokens=1, sampling=biased)
    assert altered != baseline


@requires_model
def test_logit_bias_token_id_out_of_range_raises(llm):
    """Token id >= n_vocab must raise from C++ (out_of_range -> Python exception)."""
    n_vocab = llm.n_vocab()
    # Use n_vocab itself (one past the last valid id).
    bad = SamplingParams(seed=1, logit_bias={n_vocab: -5.0})
    with pytest.raises(IndexError, match="logit_bias token id out of range"):
        llm.generate("Hello", max_tokens=1, sampling=bad)
