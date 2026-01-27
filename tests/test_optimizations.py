"""Tests for optimization features."""

import pytest

from llama_cpp import Llama, LlamaConfig, ValidationError, disable_logging

from conftest import MODEL_PATH, requires_model


@requires_model
def test_embedding_validation_disabled():
    """Embeddings raise ValidationError when embeddings=False."""
    disable_logging()
    config = LlamaConfig(model_path=MODEL_PATH, embeddings=False)
    with Llama(model_path=MODEL_PATH, config=config) as llm:
        with pytest.raises(ValidationError, match="Embeddings not enabled"):
            llm.embed("test")


@requires_model
def test_embedding_batch(llm_embed):
    """Batch embeddings work correctly."""
    result = llm_embed.create_embedding(["test 1", "test 2", "test 3"])
    assert "data" in result
    assert len(result["data"]) == 3
    for item in result["data"]:
        assert len(item["embedding"]) > 0


@requires_model
def test_reset_kv_cache_option(llm):
    """Test reset_kv_cache=False preserves KV cache."""
    llm.generate("Hello", max_tokens=5, reset_kv_cache=True)
    pos1 = llm.kv_cache_seq_pos_max()
    assert pos1 > 0

    llm.generate("World", max_tokens=5, reset_kv_cache=False)
    pos2 = llm.kv_cache_seq_pos_max()
    assert pos2 > pos1  # Cache grew


@requires_model
def test_generate_stream(llm):
    """Test generate_stream yields tokens incrementally."""
    chunks = list(llm.generate_stream("Hello", max_tokens=5))
    assert len(chunks) > 0
    assert all(isinstance(c, str) for c in chunks)
    combined = "".join(chunks)
    assert len(combined) > 0


@requires_model
def test_n_seq_max_config():
    """Test n_seq_max is configurable in LlamaConfig."""
    disable_logging()
    config = LlamaConfig(model_path=MODEL_PATH, n_seq_max=2)
    assert config.n_seq_max == 2

    with pytest.raises(ValidationError, match="n_seq_max"):
        LlamaConfig(model_path=MODEL_PATH, n_seq_max=0)


@requires_model
def test_grammar_cache_reuse(llm):
    """Test grammar samplers are cached for repeated schemas."""
    from llama_cpp.llama import _grammar_cache

    initial_size = len(_grammar_cache)

    llm.create_chat_completion(
        [{"role": "user", "content": "Return JSON"}],
        max_tokens=10,
        response_format={"type": "json_object"},
    )
    assert len(_grammar_cache) > initial_size

    cached_size = len(_grammar_cache)

    llm.create_chat_completion(
        [{"role": "user", "content": "More JSON"}],
        max_tokens=10,
        response_format={"type": "json_object"},
    )
    assert len(_grammar_cache) == cached_size  # Reused


def test_backend_guard_functions():
    """Test backend guard helper functions exist."""
    from llama_cpp import _llama

    assert hasattr(_llama, "backend_can_free")
    assert hasattr(_llama, "model_count")
    count = _llama.model_count()
    assert isinstance(count, int)
    assert count >= 0
