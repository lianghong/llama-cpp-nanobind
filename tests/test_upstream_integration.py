"""Behavioral regression coverage for the October upstream integration."""

import math
import os

import pytest
from conftest import MODEL_PATH, requires_model

from llama_cpp import Llama, LlamaConfig, SamplingParams, ValidationError, _llama


@pytest.mark.parametrize("value", [-1, True, 1.5, "1024"])
def test_moe_cache_rejects_invalid_byte_counts(value):
    with pytest.raises(ValidationError, match="moe_cache_size"):
        LlamaConfig(model_path="unused.gguf", moe_cache_size=value)


@requires_model
def test_prompt_batch_reuse_after_growth_clear_and_reset():
    """A reused batch must not retain tokens/positions/output flags from a prior call."""
    config = LlamaConfig(
        model_path=MODEL_PATH,
        n_ctx=512,
        n_batch=32,
        n_ubatch=32,
        verbose=False,
    )
    with Llama(MODEL_PATH, config=config) as llm:
        short = llm.tokenize("Hello world")
        long = llm.tokenize("A red ball and a blue box. " * 10)
        assert len(short) < config.n_batch < len(long)
        llm.ctx.decode(short)
        expected = llm.ctx.logits()
        llm.kv_cache_clear()
        llm.ctx.decode(long)
        assert llm.kv_cache_seq_pos_max() == len(long) - 1
        for reset in (llm.kv_cache_clear, llm.reset):
            reset()
            llm.ctx.decode(short)
            assert llm.kv_cache_seq_pos_max() == len(short) - 1
            assert llm.ctx.logits() == pytest.approx(expected, abs=1e-4)


@requires_model
def test_fixed_greedy_matches_logprobs_and_streaming(llm):
    """The greedy terminal keeps token choices and post-sampler probabilities."""
    params = SamplingParams(temperature=0.0, top_k=0, top_p=1.0, seed=42)
    prompt = "The capital of France is"
    fast = llm.generate(prompt, max_tokens=8, sampling=params)
    details = llm.generate(prompt, max_tokens=8, sampling=params, logprobs=3)
    assert details["tokens"]
    assert details["text"] == fast
    assert all(
        math.isfinite(item.logprob) and item.logprob == 0.0
        for item in details["token_probs"]
    )
    streamed = "".join(llm.generate_stream(prompt, max_tokens=8, sampling=params))
    assert streamed == fast


@requires_model
def test_moe_cache_budget_reaches_native_context(monkeypatch):
    """Capture the configured budget; dense fixtures cannot allocate a MoE cache."""
    original = _llama.Context
    seen = []

    def capture_context(model, params):
        seen.append(params.as_dict()["moe_cache_size"])
        params.moe_cache_size = 0
        return original(model, params)

    monkeypatch.setattr(_llama, "Context", capture_context)
    config = LlamaConfig(
        model_path=MODEL_PATH,
        n_ctx=512,
        cpu_moe=True,
        moe_cache_size=1024 * 1024,
        verbose=False,
    )
    with Llama(MODEL_PATH, config=config) as llm:
        assert llm.generate("Hello", max_tokens=2)
    assert seen == [config.moe_cache_size]


def test_moe_cache_generates_on_host_experts():
    """Optional live coverage of host expert placement and GPU cache allocation."""
    path = os.environ.get("LLAMA_MOE_TEST_MODEL")
    if not path or not os.path.isfile(path):
        pytest.skip("set LLAMA_MOE_TEST_MODEL to a MoE GGUF for cache validation")
    config = LlamaConfig(
        model_path=path,
        cpu_moe=True,
        moe_cache_size=512 * 1024 * 1024,
        n_ctx=256,
        n_batch=64,
        n_ubatch=64,
        n_rs_seq=0,
        verbose=False,
    )
    with Llama(path, config=config) as llm:
        assert llm.generate("The capital of France is", max_tokens=4)
        llm.reset()
        assert llm.generate("Hello", max_tokens=2)
