"""Tests for MTP (Multi-Token Prediction) context type.

MTP is a context-construction-time configuration: setting `ctx_type` on
`LlamaConfig` selects the `LLAMA_CONTEXT_TYPE_MTP` graph variant in
llama.cpp. It is only valid for models that ship MTP layers (currently
Qwen3.5 and Qwen3.5-MoE / Qwen3.6 MTP checkpoints). On a model without
MTP layers, context construction fails with `ModelLoadError`.
"""

import pytest

from llama_cpp import (
    LLAMA_CONTEXT_TYPE_DEFAULT,
    LLAMA_CONTEXT_TYPE_MTP,
    Llama,
    LlamaConfig,
    ModelLoadError,
    SamplingParams,
)
from llama_cpp.llama import ValidationError

from conftest import MODEL_PATH, MTP_MODEL_PATH, requires_model, requires_mtp_model


# --- Pure validation (no model required) ---


def test_constants_exposed():
    """The two enum values must be importable from llama_cpp root."""
    assert LLAMA_CONTEXT_TYPE_DEFAULT == 0
    assert LLAMA_CONTEXT_TYPE_MTP == 1


def test_default_ctx_type_is_default():
    cfg = LlamaConfig(model_path="/dev/null")
    assert cfg.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT


def test_mtp_ctx_type_accepted_in_config():
    """Validation only checks the enum range; the model load is what fails."""
    cfg = LlamaConfig(model_path="/dev/null", ctx_type=LLAMA_CONTEXT_TYPE_MTP)
    assert cfg.ctx_type == LLAMA_CONTEXT_TYPE_MTP


def test_invalid_ctx_type_rejected():
    with pytest.raises(ValidationError, match="ctx_type"):
        LlamaConfig(model_path="/dev/null", ctx_type=2)


def test_negative_ctx_type_rejected():
    with pytest.raises(ValidationError, match="ctx_type"):
        LlamaConfig(model_path="/dev/null", ctx_type=-1)


# --- End-to-end with a model ---


@requires_model
def test_default_ctx_type_loads():
    """Default ctx_type must work on the standard test model (regression
    guard: adding the new field must not break existing flows).
    """
    cfg = LlamaConfig(
        model_path=MODEL_PATH,
        n_ctx=512,
        n_gpu_layers=0,
        verbose=False,
    )
    llm = Llama(MODEL_PATH, config=cfg)
    try:
        assert llm.config.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT
    finally:
        llm.close()


@requires_model
def test_mtp_on_non_mtp_model_raises():
    """The standard Qwen3.5-4B test model does not ship MTP layers, so
    requesting MTP must fail at context construction with ModelLoadError.
    """
    cfg = LlamaConfig(
        model_path=MODEL_PATH,
        n_ctx=512,
        n_gpu_layers=0,
        ctx_type=LLAMA_CONTEXT_TYPE_MTP,
        verbose=False,
    )
    with pytest.raises(ModelLoadError):
        Llama(MODEL_PATH, config=cfg)


# --- End-to-end on an MTP-capable model ---


@requires_mtp_model
def test_mtp_loads_on_mtp_model_and_generates():
    """Positive path: an MTP-capable Qwen3.6-MoE checkpoint must load with
    `ctx_type=MTP` and produce non-empty output. The model ships
    `nextn_predict_layers=1` and `blk.*.nextn.*` tensors, which libllama
    requires for the MTP graph.
    """
    cfg = LlamaConfig(
        model_path=MTP_MODEL_PATH,
        n_ctx=512,
        n_gpu_layers=-1,
        ctx_type=LLAMA_CONTEXT_TYPE_MTP,
        verbose=False,
    )
    llm = Llama(MTP_MODEL_PATH, config=cfg)
    try:
        assert llm.config.ctx_type == LLAMA_CONTEXT_TYPE_MTP
        out = llm.generate(
            "Hello,",
            max_tokens=8,
            sampling=SamplingParams(seed=0, temperature=0.0),
        )
        assert isinstance(out, str)
        assert len(out) > 0
    finally:
        llm.close()


@requires_mtp_model
def test_mtp_default_ctx_also_loads_on_mtp_model():
    """The MTP-capable checkpoint must also load under the default
    context type (MTP layers are optional at runtime — they are simply
    unused unless `ctx_type=MTP`).
    """
    cfg = LlamaConfig(
        model_path=MTP_MODEL_PATH,
        n_ctx=512,
        n_gpu_layers=-1,
        ctx_type=LLAMA_CONTEXT_TYPE_DEFAULT,
        verbose=False,
    )
    llm = Llama(MTP_MODEL_PATH, config=cfg)
    try:
        assert llm.config.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT
    finally:
        llm.close()
