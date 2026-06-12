"""Validation tests for draft-MTP speculative decoding.

Most tests exercise the precondition checks (ctx_type, n_draft_max,
embeddings) on the small non-MTP test model. One positive test
(`test_supports_speculative_mtp_real_mtp_model_returns_true`) requires a
genuine MTP checkpoint and skips when it is absent.
"""

import pytest

from llama_cpp import (
    LLAMA_CONTEXT_TYPE_DEFAULT,
    Llama,
    LlamaConfig,
    SamplingParams,
)
from llama_cpp.llama import ValidationError

from conftest import MODEL_PATH, MTP_MODEL_PATH, requires_model, requires_mtp_model


@requires_model
def test_supports_speculative_mtp_default_ctx_returns_false():
    # Regression: a plain Qwen3.5 checkpoint (no nextn_predict_layers metadata)
    # must report False from the metadata gate alone — MTP-context allocation
    # success is not a capability signal (pre-b9180 llama.cpp builds allocate
    # a degenerate MTP context for qwen35-arch models with zero MTP layers,
    # and the draft-MTP decode path then SIGABRTs at runtime).
    cfg = LlamaConfig(model_path=MODEL_PATH, n_ctx=512, n_gpu_layers=0, verbose=False)
    llm = Llama(MODEL_PATH, config=cfg)
    try:
        assert llm.mtp_predict_layers() == 0
        assert llm.supports_speculative_mtp() is False
    finally:
        llm.close()


@requires_mtp_model
def test_supports_speculative_mtp_real_mtp_model_returns_true():
    # A genuine MTP checkpoint declares nextn_predict_layers > 0 and the gate
    # must let it through (the metadata fix must not regress real MTP support).
    # n_gpu_layers=-1: the 35B checkpoint is impractical on CPU, and the
    # probe's success path allocates a second (draft) context.
    cfg = LlamaConfig(
        model_path=MTP_MODEL_PATH, n_ctx=512, n_gpu_layers=-1, verbose=False
    )
    llm = Llama(MTP_MODEL_PATH, config=cfg)
    try:
        assert llm.mtp_predict_layers() > 0
        assert llm.supports_speculative_mtp() is True
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
def test_validate_speculative_non_mtp_model_raises():
    """A non-MTP checkpoint must be rejected by speculative=True even when
    the user-facing ctx_type is the (correct) default."""
    cfg = LlamaConfig(model_path=MODEL_PATH, n_ctx=512, n_gpu_layers=0, verbose=False)
    llm = Llama(MODEL_PATH, config=cfg)
    try:
        with pytest.raises(ValidationError, match="MTP graph"):
            llm._validate_speculative(speculative=True)
    finally:
        llm.close()


@requires_model
def test_validate_speculative_false_is_noop():
    cfg = LlamaConfig(model_path=MODEL_PATH, n_ctx=512, n_gpu_layers=0, verbose=False)
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
    cfg = LlamaConfig(model_path=MODEL_PATH, n_ctx=512, n_gpu_layers=0, verbose=False)
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


def test_generate_tokens_speculative_mtp_symbol_exists():
    """The new C++ entry point must be importable as a module-level binding."""
    from llama_cpp import _llama

    assert hasattr(_llama, "generate_tokens_speculative_mtp")
