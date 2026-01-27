"""Tests for Llama inference API."""

import pytest

from llama_cpp import (
    Llama,
    LlamaConfig,
    LlamaGrammar,
    ModelLoadError,
    SamplingParams,
    ValidationError,
    disable_logging,
)

from conftest import MODEL_PATH, requires_model


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


# LoRA lifecycle test
@requires_model
def test_lora_clear(llm):
    assert len(llm._lora_adapters) == 0
    llm.clear_lora()
    assert len(llm._lora_adapters) == 0
