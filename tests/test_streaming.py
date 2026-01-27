"""Tests for true incremental streaming API."""

import time

import pytest

from llama_cpp import Llama, SamplingParams

from conftest import requires_model


@requires_model
def test_generate_stream_incremental(model_path):
    """Test that generate_stream yields tokens incrementally, not buffered."""
    llm = Llama(model_path)

    first_yield_time = None
    last_yield_time = None
    chunk_count = 0
    start_time = time.time()

    for chunk in llm.generate_stream("Count to 5", max_tokens=32):
        if first_yield_time is None:
            first_yield_time = time.time()
        last_yield_time = time.time()
        chunk_count += 1

    llm.close()

    assert chunk_count > 0, "Should yield at least one chunk"

    # If truly incremental, first yield should happen before generation completes
    time_to_first_yield = first_yield_time - start_time
    total_time = last_yield_time - start_time

    if chunk_count > 1 and total_time > 0.1:
        ratio = time_to_first_yield / total_time
        assert ratio < 0.8, f"First yield at {ratio:.1%} suggests buffering"


@requires_model
def test_generate_stream_basic(model_path):
    """Test basic generate_stream functionality."""
    llm = Llama(model_path)

    chunks = []
    for chunk in llm.generate_stream("Say hello", max_tokens=16):
        assert isinstance(chunk, str)
        chunks.append(chunk)

    result = "".join(chunks)
    assert len(result) > 0
    assert len(chunks) > 0

    llm.close()


@requires_model
def test_generate_stream_stop_sequences(model_path):
    """Test generate_stream with stop sequences."""
    llm = Llama(model_path)

    chunks = list(llm.generate_stream("Count: 1, 2, 3", max_tokens=32, stop=[","]))
    result = "".join(chunks)
    assert isinstance(result, str)

    llm.close()


@requires_model
def test_generate_stream_early_termination(model_path):
    """Test that generator can be closed early without hanging."""
    llm = Llama(model_path)

    chunks = []
    for i, chunk in enumerate(llm.generate_stream("Test prompt", max_tokens=64)):
        chunks.append(chunk)
        if i >= 2:
            break

    assert len(chunks) == 3

    # Verify model still works after early termination
    more_chunks = list(llm.generate_stream("Another test", max_tokens=8))
    assert len(more_chunks) > 0

    llm.close()


@requires_model
def test_generate_stream_vs_generate_consistency(model_path):
    """Test that generate_stream produces same output as generate."""
    llm = Llama(model_path)

    sampling = SamplingParams(temperature=0.0, seed=42)
    prompt = "The capital of France is"

    streamed = "".join(llm.generate_stream(prompt, max_tokens=16, sampling=sampling))

    llm.ctx.kv_cache_clear()
    normal = llm.generate(prompt, max_tokens=16, sampling=sampling)

    assert streamed == normal, f"Streamed: {streamed!r}\nNormal: {normal!r}"

    llm.close()
