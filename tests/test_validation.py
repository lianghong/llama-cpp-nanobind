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


def test_to_native_sets_every_native_field():
    """Drift guard: every writable field on the C++ ``SamplerParams`` must be
    populated by ``SamplingParams.to_native()``.

    The Python ``SamplingParams`` dataclass, the nanobind ``SamplerParams``
    class, and the C++ ``SamplerChain::Params`` struct are three hand-kept
    definitions with non-matching field names (``temperature``→``temp``,
    ``repeat_last_n``→``penalty_last_n``, …). A field-name match check would
    false-positive on those renames, so instead we assert behavioural
    completeness: build a SamplingParams whose every field differs from the
    native default, run to_native(), and require that EVERY native field moved
    off its default. If a future C++ field is added but to_native() forgets to
    set it, that field stays at its default and this test fails — without
    needing to encode the name mapping.
    """
    import llama_cpp._llama as _llama

    # Native defaults (a fresh native object — what an unset field would keep).
    default_native = _llama.SamplerParams()
    native_fields = [a for a in dir(default_native) if not a.startswith("_")]
    defaults = {f: getattr(default_native, f) for f in native_fields}

    # A SamplingParams with every field set to a value distinct from the native
    # default so a "got set" check is unambiguous. logit_bias and
    # dry_seq_breakers are containers — give them non-empty, non-default values.
    populated = SamplingParams(
        temperature=0.123,
        top_k=7,
        top_p=0.456,
        min_p=0.0789,
        typical_p=0.654,
        min_keep=3,
        repeat_penalty=1.234,
        repeat_last_n=55,
        presence_penalty=0.321,
        frequency_penalty=0.213,
        seed=98765,
        temp_delta=0.111,
        temp_exponent=1.222,
        xtc_probability=0.333,
        xtc_threshold=0.444,
        top_n_sigma=2.5,
        dry_multiplier=0.555,
        dry_base=1.875,
        dry_allowed_length=9,
        dry_penalty_last_n=77,
        dry_seq_breakers=["XX", "YY"],
        adaptive_p_target=0.666,
        adaptive_p_decay=0.777,
        logit_bias={123: 4.5},
    )
    native = populated.to_native()

    unset = []
    for field in native_fields:
        value = getattr(native, field)
        # A field to_native() forgot to set keeps its native default.
        if value == defaults[field]:
            unset.append(field)

    assert not unset, (
        "SamplingParams.to_native() did not populate native SamplerParams "
        f"field(s) {sorted(unset)!r} — either a new C++ field needs wiring "
        "in to_native(), or its sentinel above accidentally equals the native "
        "default."
    )
