"""UnifiedLLM speculative-decoding tests that need an MTP-capable model.

These live in their own file so no other module-scoped fixture (e.g. the
``unified_llm`` fixture in ``test_unified.py``) keeps a separate model
resident in VRAM while the large 35B MTP checkpoint is loaded.

Default model: ``./models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`` (override via
``LLAMA_MTP_TEST_MODEL``). Tests skip cleanly when the file is absent.
"""

from llama_cpp.unified import UnifiedLLM

from conftest import MTP_MODEL_PATH, requires_mtp_model


# MTP test model (35B-A3B) is large; keep n_ctx tiny to fit common VRAM budgets.
_MTP_TEST_NCTX = 1024


@requires_mtp_model
def test_unified_llm_speculative_auto_enables_on_mtp():
    """On an MTP checkpoint, speculative='auto' (default) must resolve to True."""
    with UnifiedLLM(MTP_MODEL_PATH, verbose=False, n_ctx=_MTP_TEST_NCTX) as llm:
        assert llm.speculative_enabled is True
        kwargs = llm.backend._sampling_kwargs(None)
        assert kwargs.get("speculative") is True


@requires_mtp_model
def test_unified_llm_speculative_explicit_false_on_mtp():
    """speculative=False must disable even when MTP is available."""
    with UnifiedLLM(
        MTP_MODEL_PATH, verbose=False, n_ctx=_MTP_TEST_NCTX, speculative=False
    ) as llm:
        assert llm.speculative_enabled is False
        kwargs = llm.backend._sampling_kwargs(None)
        assert "speculative" not in kwargs


@requires_mtp_model
def test_unified_llm_speculative_smoke_generation():
    """Smoke test: speculative-enabled UnifiedLLM produces output without raising.

    Qwen3.6 is a thinking model — the response may be empty after the
    ``<think>...</think>`` block is stripped if the budget runs out inside
    the thinking block.  We just assert the call returns a string.
    """
    with UnifiedLLM(
        MTP_MODEL_PATH, verbose=False, n_ctx=_MTP_TEST_NCTX, n_draft_max=2
    ) as llm:
        assert llm.speculative_enabled is True
        response = llm.generate("Say hi.", max_tokens=256)
        assert isinstance(response, str)
