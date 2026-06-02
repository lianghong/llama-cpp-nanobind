"""Tests for Llama inference API."""

from conftest import MODEL_PATH
from conftest import requires_model
from llama_cpp import disable_logging
from llama_cpp import Llama
from llama_cpp import LlamaConfig
from llama_cpp import LlamaGrammar
from llama_cpp import ModelLoadError
from llama_cpp import SamplingParams
from llama_cpp import ValidationError
import pytest


# Basic generation tests
@requires_model
def test_short_generation(llm):
    text = llm.generate("The quick brown fox", max_tokens=12)
    assert isinstance(text, str)
    assert len(text.strip()) > 0


@requires_model
def test_custom_sampling(llm):
    params = SamplingParams(
        temperature=0.7, top_p=0.9, repeat_penalty=1.05, repeat_last_n=32
    )
    out = llm.generate("List two colors:", max_tokens=8, sampling=params)
    assert isinstance(out, str)
    assert len(out) > 0


@requires_model
def test_chat_completion_api(llm):
    resp = llm.create_chat_completion(
        [{"role": "user", "content": "Answer with a single word animal"}],
        max_tokens=4,
    )
    assert resp["object"] == "chat.completion"
    assert resp["choices"][0]["message"]["role"] == "assistant"
    assert isinstance(resp["choices"][0]["message"]["content"], str)


# Token helper tests
@requires_model
def test_tokenize_detokenize(llm):
    text = "Hello world"
    tokens = llm.tokenize(text)
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)
    decoded = llm.detokenize(tokens)
    assert text in decoded or decoded.strip() == text.strip()


@requires_model
def test_n_tokens(llm):
    count = llm.n_tokens("Hello world")
    assert isinstance(count, int)
    assert count > 0


@requires_model
def test_special_tokens(llm):
    assert isinstance(llm.token_bos(), int)
    assert isinstance(llm.token_eos(), int)
    assert isinstance(llm.token_eot(), int)


# Model info tests
@requires_model
def test_model_info(llm):
    assert llm.n_ctx() > 0
    assert llm.n_vocab() > 0
    assert llm.n_embd() > 0
    assert llm.model_size() > 0
    assert llm.n_params() > 0
    assert llm.n_layer() > 0


@requires_model
def test_metadata_cached(llm):
    m1 = llm.metadata
    m2 = llm.metadata
    assert m1 is m2  # Same object (cached)
    assert isinstance(m1, dict)


# KV cache tests
@requires_model
def test_kv_cache_operations(llm):
    llm.reset()
    assert llm.kv_cache_seq_pos_max() == -1  # Empty
    llm.generate("Hello", max_tokens=5)
    assert llm.kv_cache_seq_pos_max() > 0  # Has content
    llm.kv_cache_seq_rm(0)
    assert llm.kv_cache_seq_pos_max() == -1  # Cleared


# State save/load tests
@requires_model
def test_state_bytes(llm):
    llm.reset()
    llm.generate("Test", max_tokens=3)
    state = llm.get_state()
    assert isinstance(state, bytes)
    assert len(state) > 0


# OpenAI-compatible API tests
@requires_model
def test_call_returns_dict(llm):
    result = llm("Hello", max_tokens=5)
    assert isinstance(result, dict)
    assert "choices" in result
    assert "usage" in result
    assert result["usage"]["prompt_tokens"] > 0


@requires_model
def test_call_logprobs_echo_completion_tokens_excludes_prompt(llm):
    """Regression: with logprobs + echo=True, usage.completion_tokens must
    count only generated tokens, not prompt+generated. The logprobs path
    returns echoed prompt tokens prepended in result['tokens'], so a naive
    len(tokens) over-counted completions by the prompt length.
    """
    prompt = "The capital of France is"
    max_new = 6
    res = llm(prompt, max_tokens=max_new, logprobs=1, echo=True)
    usage = res["usage"]
    # completion_tokens reflects only generated tokens (<= max_tokens), never
    # the echoed prompt.
    assert 0 < usage["completion_tokens"] <= max_new
    assert usage["prompt_tokens"] > 0
    # total = prompt + completion (no double-counting of the echoed prompt).
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    # echo=False must agree on the completion count (echo only changes the
    # echoed text/prompt accounting, not the number of generated tokens).
    res_no_echo = llm(prompt, max_tokens=max_new, logprobs=1, echo=False)
    assert res_no_echo["usage"]["completion_tokens"] <= max_new


@requires_model
def test_streaming(llm):
    chunks = list(llm.generate("Hello", max_tokens=5, stream=True))
    assert len(chunks) > 0
    assert all(isinstance(c, str) for c in chunks)


# Grammar tests (no model required)
def test_grammar_from_string():
    grammar = LlamaGrammar.from_string('root ::= "yes" | "no"')
    assert grammar._grammar_str == 'root ::= "yes" | "no"'


def test_grammar_from_json_schema():
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    grammar = LlamaGrammar.from_json_schema(schema)
    assert "root" in grammar._grammar_str


def test_grammar_from_json_schema_warns_on_top_level_unsupported(caplog):
    """Top-level unsupported keys (anyOf, enum, ...) must surface a warning."""
    import logging

    schema = {
        "type": "object",
        "properties": {"x": {"type": "string"}},
        "required": ["x"],
        "anyOf": [{"type": "string"}, {"type": "number"}],
    }
    with caplog.at_level(logging.WARNING, logger="root"):
        LlamaGrammar.from_json_schema(schema)
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "anyOf" in msgs
    assert "required" in msgs


def test_grammar_from_json_schema_warns_on_nested_unsupported(caplog):
    """Unsupported keys inside `properties.<name>` must surface a warning."""
    import logging

    schema = {
        "type": "object",
        "properties": {
            "color": {"type": "string", "enum": ["red", "blue"]},
        },
    }
    with caplog.at_level(logging.WARNING, logger="root"):
        LlamaGrammar.from_json_schema(schema)
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "color" in msgs
    assert "enum" in msgs


def test_grammar_from_json_schema_warns_on_generic_fallback(caplog):
    """Schemas without typed properties fall back to JSON_GRAMMAR with a warning."""
    import logging

    schema = {"anyOf": [{"type": "string"}, {"type": "number"}]}
    with caplog.at_level(logging.WARNING, logger="root"):
        LlamaGrammar.from_json_schema(schema)
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "fall" in msgs.lower()  # "falling back" / "fallback"


def test_grammar_from_json_schema_quiet_for_supported(caplog):
    """A clean, fully-supported schema must not emit any warnings."""
    import logging

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "active": {"type": "boolean"},
        },
    }
    with caplog.at_level(logging.WARNING, logger="root"):
        LlamaGrammar.from_json_schema(schema)
    assert not any(r.levelno >= logging.WARNING for r in caplog.records), (
        "supported schema should not emit warnings"
    )


@requires_model
def test_json_mode(llm):
    resp = llm.create_chat_completion(
        [{"role": "user", "content": "Return JSON with key 'x' value 1"}],
        max_tokens=20,
        response_format={"type": "json_object"},
    )
    content = resp["choices"][0]["message"]["content"]
    assert "{" in content  # Should contain JSON


# Performance metrics tests
@requires_model
def test_perf_metrics(llm):
    llm.perf_reset()
    llm.generate("Test", max_tokens=3)
    perf = llm.perf()
    assert isinstance(perf, dict)
    assert "n_eval" in perf


# Config tests
@requires_model
def test_config_verbose_false():
    disable_logging()
    config = LlamaConfig(model_path=MODEL_PATH, verbose=False, n_ctx=512)
    with Llama(MODEL_PATH, config=config) as llm:
        assert llm.n_ctx() == 512


# Validation tests
@requires_model
def test_invalid_max_tokens(llm):
    with pytest.raises(ValidationError):
        llm.generate("test", max_tokens=0)
    with pytest.raises(ValidationError):
        llm.generate("test", max_tokens=-1)


@requires_model
def test_invalid_prompt(llm):
    with pytest.raises(ValidationError):
        llm.generate(123, max_tokens=5)  # type: ignore
    with pytest.raises(ValidationError):
        llm.generate("", max_tokens=5)


# Config validation tests (no model required)
def test_invalid_config_n_ctx():
    with pytest.raises(ValidationError):
        LlamaConfig(model_path="dummy.gguf", n_ctx=0)


def test_invalid_config_n_batch():
    with pytest.raises(ValidationError):
        LlamaConfig(model_path="dummy.gguf", n_batch=0)


def test_invalid_sampling_params():
    with pytest.raises(ValidationError):
        SamplingParams(temperature=-1.0)
    with pytest.raises(ValidationError):
        SamplingParams(top_p=1.5)
    with pytest.raises(ValidationError):
        SamplingParams(top_k=-1)


def test_model_load_error():
    disable_logging()
    with pytest.raises(ModelLoadError):
        Llama("nonexistent_model.gguf")


# Context manager and cleanup tests
@requires_model
def test_context_manager():
    disable_logging()
    with Llama(MODEL_PATH) as llm:
        text = llm.generate("Hello", max_tokens=5)
        assert isinstance(text, str)
    assert llm._closed
    assert llm.ctx is None
    assert llm.model is None


@requires_model
def test_double_close_safe():
    disable_logging()
    llm = Llama(MODEL_PATH)
    llm.close()
    llm.close()  # Should not raise
    assert llm._closed


# Stop sequence tests
@requires_model
def test_stop_sequences(llm):
    stops = ["END", "STOP", ".", "!", "?"]
    result = llm.generate("Hello", max_tokens=20, stop=stops)
    assert isinstance(result, str)


@requires_model
def test_stop_sequence_validation(llm):
    # Too many stop sequences
    stops = [f"stop{i}" for i in range(25)]
    with pytest.raises(ValidationError):
        llm.generate("test", max_tokens=5, stop=stops)

    # Stop sequence too long
    with pytest.raises(ValidationError):
        llm.generate("test", max_tokens=5, stop=["x" * 600])


# Logprobs tests
@requires_model
def test_logprobs_basic(llm):
    """Test logprobs returns valid structure."""
    result = llm.generate("The capital of France is", max_tokens=8, logprobs=3)
    assert isinstance(result, dict)
    assert "text" in result
    assert "tokens" in result
    assert "token_probs" in result
    assert len(result["tokens"]) > 0
    assert len(result["token_probs"]) == len(result["tokens"])
    for tp in result["token_probs"]:
        assert isinstance(tp.token, int)
        assert isinstance(tp.logprob, float)
        assert tp.logprob <= 0.0  # Log-probs are non-positive
        assert len(tp.top_logprobs) <= 3


@requires_model
def test_logprobs_short_prompt(llm):
    """Test logprobs with minimal prompt to cover edge token values."""
    result = llm.generate("Hi", max_tokens=4, logprobs=1)
    assert isinstance(result, dict)
    assert len(result["tokens"]) > 0
    for tp in result["token_probs"]:
        assert tp.token >= 0  # No NULL/-1 tokens in output
        assert isinstance(tp.logprob, float)


@requires_model
def test_logprobs_with_stop_sequences(llm):
    """Test logprobs combined with stop sequences."""
    result = llm.generate("Count: 1, 2, 3", max_tokens=16, logprobs=1, stop=["."])
    assert isinstance(result, dict)
    assert "tokens" in result
    # All tokens should be valid (no out-of-range from stop handling)
    for tp in result["token_probs"]:
        assert tp.token >= 0


# Model architecture introspection tests
@requires_model
def test_model_architecture_introspection(llm):
    """Test that model introspection methods return sensible values."""
    assert isinstance(llm.n_head(), int)
    assert llm.n_head() > 0
    assert isinstance(llm.has_encoder(), bool)
    assert isinstance(llm.has_decoder(), bool)
    assert isinstance(llm.is_recurrent(), bool)
    assert isinstance(llm.is_hybrid(), bool)
    # Standard decoder-only model (Qwen3) should have decoder but no encoder
    assert llm.has_decoder() is True
    assert llm.has_encoder() is False


# Special tokens extended tests
@requires_model
def test_special_tokens_extended(llm):
    """Test extended special token access (sep, nl, pad)."""
    assert isinstance(llm.token_sep(), int)
    assert isinstance(llm.token_nl(), int)
    assert isinstance(llm.token_pad(), int)
    # Newline token should exist for text models
    assert llm.token_nl() >= 0


# Auto-detect BOS test
@requires_model
def test_auto_detect_bos(llm):
    """Test that add_bos is auto-detected from model preference.

    The user-supplied config is not mutated; the resolved effective value
    lives on the instance as ``_effective_add_bos``.
    """
    assert isinstance(llm.get_add_bos(), bool)
    # User-supplied config is not mutated (None = "auto-detect").
    assert llm.config.add_bos is None
    # Resolved effective value is exposed on the instance.
    assert isinstance(llm._effective_add_bos, bool)
    assert llm._effective_add_bos == llm.get_add_bos()


# Memory introspection tests
@requires_model
def test_memory_introspection(llm):
    """Test memory introspection methods."""
    assert isinstance(llm.memory_can_shift(), bool)
    llm.reset()
    # After reset, seq_pos_min should indicate empty
    min_pos = llm.kv_cache_seq_pos_min()
    assert isinstance(min_pos, int)


# Runtime context toggle tests
@requires_model
def test_set_causal_attn(llm):
    """Test runtime causal attention toggle."""
    # Should not raise
    llm.set_causal_attn(True)
    llm.set_causal_attn(False)
    llm.set_causal_attn(True)


# New sampler tests
def test_dry_sampler_params():
    """Test DRY sampler parameter validation."""
    p = SamplingParams(dry_multiplier=0.8)
    assert p.dry_multiplier == 0.8
    assert p.dry_base == 1.75
    assert p.dry_allowed_length == 2
    with pytest.raises(ValidationError):
        SamplingParams(dry_multiplier=-1.0)
    with pytest.raises(ValidationError):
        SamplingParams(dry_base=0.0)
    with pytest.raises(ValidationError):
        SamplingParams(dry_allowed_length=0)


def test_xtc_sampler_params():
    """Test XTC sampler parameter validation."""
    p = SamplingParams(xtc_probability=0.5, xtc_threshold=0.2)
    assert p.xtc_probability == 0.5
    assert p.xtc_threshold == 0.2
    with pytest.raises(ValidationError):
        SamplingParams(xtc_probability=1.5)
    with pytest.raises(ValidationError):
        SamplingParams(xtc_threshold=-0.1)


def test_dynamic_temp_params():
    """Test dynamic temperature parameter validation."""
    p = SamplingParams(temp_delta=0.5, temp_exponent=2.0)
    assert p.temp_delta == 0.5
    assert p.temp_exponent == 2.0
    with pytest.raises(ValidationError):
        SamplingParams(temp_delta=-1.0)
    with pytest.raises(ValidationError):
        SamplingParams(temp_exponent=0.0)


def test_top_n_sigma_params():
    """Test top-n-sigma parameter works."""
    p = SamplingParams(top_n_sigma=2.0)
    assert p.top_n_sigma == 2.0
    # Negative means disabled (valid)
    p2 = SamplingParams(top_n_sigma=-1.0)
    assert p2.top_n_sigma == -1.0


@requires_model
def test_dry_sampler_generation(llm):
    """Test generation with DRY sampler enabled."""
    params = SamplingParams(dry_multiplier=0.8, temperature=0.7)
    out = llm.generate("Hello world", max_tokens=8, sampling=params)
    assert isinstance(out, str)
    assert len(out) > 0


@requires_model
def test_xtc_sampler_generation(llm):
    """Test generation with XTC sampler enabled."""
    params = SamplingParams(xtc_probability=0.5, xtc_threshold=0.1, temperature=0.7)
    out = llm.generate("Hello world", max_tokens=8, sampling=params)
    assert isinstance(out, str)
    assert len(out) > 0


@requires_model
def test_dynamic_temp_generation(llm):
    """Test generation with dynamic temperature."""
    params = SamplingParams(temperature=0.8, temp_delta=0.3, temp_exponent=1.5)
    out = llm.generate("Hello world", max_tokens=8, sampling=params)
    assert isinstance(out, str)
    assert len(out) > 0


@requires_model
def test_top_n_sigma_generation(llm):
    """Test generation with top-n-sigma sampler."""
    params = SamplingParams(top_n_sigma=2.0, temperature=0.7)
    out = llm.generate("Hello world", max_tokens=8, sampling=params)
    assert isinstance(out, str)
    assert len(out) > 0


# LoRA lifecycle test
@requires_model
def test_lora_clear(llm):
    assert len(llm._lora_adapters) == 0
    llm.clear_lora()
    assert len(llm._lora_adapters) == 0
