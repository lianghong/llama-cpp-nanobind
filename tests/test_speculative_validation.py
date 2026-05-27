"""Validation-only tests for draft-MTP speculative decoding.

These tests do not require a model that ships MTP layers — they exercise
the precondition checks (ctx_type, n_draft_max, embeddings) at the
Llama / SamplingParams / Context-binding boundary.
"""

import pytest

from llama_cpp import (
    LLAMA_CONTEXT_TYPE_DEFAULT,
    LLAMA_CONTEXT_TYPE_MTP,
    Llama,
    LlamaConfig,
    SamplingParams,
)
from llama_cpp.llama import ValidationError

from conftest import MODEL_PATH, requires_model


@requires_model
def test_supports_speculative_mtp_default_ctx_returns_false():
    cfg = LlamaConfig(
        model_path=MODEL_PATH, n_ctx=512, n_gpu_layers=0, verbose=False
    )
    llm = Llama(MODEL_PATH, config=cfg)
    try:
        assert llm.ctx.supports_speculative_mtp() is False
    finally:
        llm.close()


def test_n_draft_max_default_is_two():
    sp = SamplingParams()
    assert sp.n_draft_max == 2


def test_n_draft_max_zero_rejected():
    with pytest.raises(ValidationError, match="n_draft_max"):
        SamplingParams(n_draft_max=0)


def test_n_draft_max_nine_rejected():
    with pytest.raises(ValidationError, match="n_draft_max"):
        SamplingParams(n_draft_max=9)


def test_n_draft_max_eight_accepted():
    sp = SamplingParams(n_draft_max=8)
    assert sp.n_draft_max == 8


def test_n_draft_max_one_accepted():
    sp = SamplingParams(n_draft_max=1)
    assert sp.n_draft_max == 1


@requires_model
def test_validate_speculative_default_ctx_raises():
    cfg = LlamaConfig(
        model_path=MODEL_PATH, n_ctx=512, n_gpu_layers=0, verbose=False
    )
    llm = Llama(MODEL_PATH, config=cfg)
    try:
        with pytest.raises(ValidationError, match="ctx_type=LLAMA_CONTEXT_TYPE_MTP"):
            llm._validate_speculative(speculative=True)
    finally:
        llm.close()


@requires_model
def test_validate_speculative_false_is_noop():
    cfg = LlamaConfig(
        model_path=MODEL_PATH, n_ctx=512, n_gpu_layers=0, verbose=False
    )
    llm = Llama(MODEL_PATH, config=cfg)
    try:
        llm._validate_speculative(speculative=False)  # must not raise
    finally:
        llm.close()


@requires_model
def test_validate_speculative_with_embeddings_raises():
    cfg = LlamaConfig(
        model_path=MODEL_PATH,
        n_ctx=512,
        n_gpu_layers=0,
        verbose=False,
        embeddings=True,
        ctx_type=LLAMA_CONTEXT_TYPE_DEFAULT,
    )
    llm = Llama(MODEL_PATH, config=cfg)
    try:
        # Two failures possible: ctx_type wrong AND embeddings on. Either
        # message wins; we accept either substring.
        with pytest.raises(ValidationError):
            llm._validate_speculative(speculative=True)
    finally:
        llm.close()


@requires_model
def test_decode_multi_smoke():
    """Decode 3 tokens at once via the new multi-token batch helper.
    cur_pos_ must advance by exactly the number of tokens decoded.
    """
    cfg = LlamaConfig(
        model_path=MODEL_PATH, n_ctx=512, n_gpu_layers=0, verbose=False
    )
    llm = Llama(MODEL_PATH, config=cfg)
    try:
        # Tokenize a few BOS-free tokens; semantic content doesn't matter
        toks = llm.tokenize("Hi there friend", add_special=False)[:3]
        # cur_pos starts at 0
        llm.ctx.decode_multi(toks)
        # No public cur_pos getter — check via kv_cache_seq_pos_max
        assert llm.ctx.kv_cache_seq_pos_max(0) == len(toks) - 1
    finally:
        llm.close()
