"""End-to-end tests for draft-MTP speculative decoding.

These tests require an MTP-capable Qwen3.6-MoE checkpoint
(`nextn_predict_layers > 0` + `blk.*.nextn.*` tensors). The default
location is `models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`, overridable via
`LLAMA_MTP_TEST_MODEL`. Tests skip cleanly when absent.
"""

import os

import pytest

from llama_cpp import (
    Llama,
    LlamaConfig,
    SamplingParams,
)

MTP_MODEL_PATH = os.environ.get(
    "LLAMA_MTP_TEST_MODEL",
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "models",
        "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf",
    ),
)
requires_mtp_model = pytest.mark.skipif(
    not os.path.exists(MTP_MODEL_PATH), reason="MTP-capable test model not found"
)


def _make_mtp_llm(**overrides):
    cfg = LlamaConfig(
        model_path=MTP_MODEL_PATH,
        n_ctx=512,
        n_gpu_layers=-1,
        verbose=False,
        **overrides,
    )
    return Llama(MTP_MODEL_PATH, config=cfg)


@requires_mtp_model
def test_speculative_smoke():
    """speculative=True must produce non-empty output and not raise."""
    llm = _make_mtp_llm()
    try:
        out = llm.generate(
            "Explain MoE in one sentence.",
            max_tokens=32,
            sampling=SamplingParams(seed=0, temperature=0.0, n_draft_max=2),
            speculative=True,
        )
        assert isinstance(out, str) and out.strip()
    finally:
        llm.close()


@requires_mtp_model
def test_speculative_matches_baseline_greedy():
    """Greedy (temp=0) speculative decode must produce the **same** token
    sequence as the per-token baseline."""
    prompt = "Count from one to five:"
    sampling = SamplingParams(seed=0, temperature=0.0, n_draft_max=2)

    llm = _make_mtp_llm()
    try:
        baseline = llm.generate(
            prompt, max_tokens=24, sampling=sampling, speculative=False
        )
    finally:
        llm.close()

    llm = _make_mtp_llm()
    try:
        spec = llm.generate(prompt, max_tokens=24, sampling=sampling, speculative=True)
    finally:
        llm.close()

    assert spec == baseline


@requires_mtp_model
def test_speculative_with_grammar():
    """Grammar + speculative must produce grammar-valid output."""
    from llama_cpp import LlamaGrammar

    json_grammar = LlamaGrammar.from_string(
        'root ::= "{" ws "\\"ok\\"" ws ":" ws "true" ws "}" ws\nws ::= [ \\t\\n]*\n'
    )
    llm = _make_mtp_llm()
    try:
        out = llm.create_chat_completion(
            messages=[{"role": "user", "content": "Reply only valid JSON."}],
            max_tokens=32,
            grammar=json_grammar,
            speculative=True,
            n_draft_max=2,
            temperature=0.0,
            seed=0,
        )
        text = out["choices"][0]["message"]["content"]
        assert "{" in text and "}" in text
    finally:
        llm.close()


@requires_mtp_model
def test_speculative_streaming():
    """generate_stream with speculative=True must yield the same concatenated
    text as the non-streaming path under greedy sampling.
    """
    prompt = "List three colors:"
    sp = SamplingParams(seed=0, temperature=0.0, n_draft_max=2)

    llm = _make_mtp_llm()
    try:
        non_stream = llm.generate(prompt, max_tokens=24, sampling=sp, speculative=True)
    finally:
        llm.close()

    llm = _make_mtp_llm()
    try:
        chunks = list(
            llm.generate_stream(prompt, max_tokens=24, sampling=sp, speculative=True)
        )
    finally:
        llm.close()

    assert "".join(chunks) == non_stream


@requires_mtp_model
def test_speculative_with_cache_prompt():
    """Two-turn chat with cache_prompt=True + speculative=True must extend
    the prompt-cache mirror correctly.
    """
    llm = _make_mtp_llm()
    try:
        sp = SamplingParams(seed=0, temperature=0.0, n_draft_max=2)
        llm.generate(
            "Hello",
            max_tokens=8,
            sampling=sp,
            speculative=True,
            reset_kv_cache=True,
            cache_prompt=True,
        )
        # Mirror should be non-empty and aligned to KV.
        assert len(llm._cached_prompt_tokens) > 0
        kv_max = llm.ctx.kv_cache_seq_pos_max(0)
        assert len(llm._cached_prompt_tokens) == kv_max + 1
        # Continuation reuses the prefix.
        out2 = llm.generate(
            "Hello world",
            max_tokens=8,
            sampling=sp,
            speculative=True,
            reset_kv_cache=False,
            cache_prompt=True,
        )
        assert isinstance(out2, str) and out2.strip()
    finally:
        llm.close()


@requires_mtp_model
def test_speculative_n_draft_max_bounds():
    """Both n_draft_max=1 and n_draft_max=8 must work end-to-end.

    n_rs_seq must be >= n_draft_max so the recurrent draft context can
    roll back rejected drafts on hybrid attention models.
    """
    for n in (1, 8):
        llm = _make_mtp_llm(n_rs_seq=max(2, n))
        try:
            out = llm.generate(
                "Hi",
                max_tokens=12,
                sampling=SamplingParams(seed=0, temperature=0.0, n_draft_max=n),
                speculative=True,
            )
            assert isinstance(out, str)
        finally:
            llm.close()


@requires_mtp_model
def test_speculative_stop_sequence():
    """A multi-token stop must be honored under speculative; the stop string
    must NOT appear in the returned text.
    """
    llm = _make_mtp_llm()
    try:
        out = llm.generate(
            "Counting: 1 2 3 STOPHERE 4 5",
            max_tokens=64,
            sampling=SamplingParams(seed=0, temperature=0.0, n_draft_max=2),
            stop=["STOPHERE"],
            speculative=True,
        )
        assert "STOPHERE" not in out
    finally:
        llm.close()
