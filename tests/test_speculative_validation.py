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
