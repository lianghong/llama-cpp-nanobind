"""Shared pytest fixtures for llama-cpp-nanobind tests."""

import os

import pytest

from llama_cpp import Llama, LlamaConfig, disable_logging

MODEL_PATH = os.environ.get(
    "LLAMA_TEST_MODEL",
    os.path.join(os.path.dirname(__file__), "..", "models", "Qwen3-8B-Q6_K.gguf"),
)

requires_model = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH), reason="test model not found"
)


@pytest.fixture
def model_path():
    """Fixture providing model path, skips if not found."""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("test model not found")
    return MODEL_PATH


@pytest.fixture(scope="module")
def llm():
    """Shared Llama instance for tests."""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("test model not found")
    disable_logging()
    instance = Llama(model_path=MODEL_PATH)
    yield instance
    instance.close()


@pytest.fixture(scope="module")
def llm_embed():
    """Llama instance with embeddings enabled."""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("test model not found")
    disable_logging()
    config = LlamaConfig(model_path=MODEL_PATH, embeddings=True)
    instance = Llama(model_path=MODEL_PATH, config=config)
    yield instance
    instance.close()
