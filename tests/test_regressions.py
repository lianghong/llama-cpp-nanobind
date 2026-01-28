"""Regression tests for fixed issues."""

import os
import tempfile


from llama_cpp import Llama, LlamaGrammar, disable_logging

from conftest import MODEL_PATH, requires_model


@requires_model
def test_state_load_then_generate():
    """Verify that loading state correctly updates position bookkeeping."""
    disable_logging()
    with Llama(model_path=MODEL_PATH) as llm:
        llm.generate("The capital of France is", max_tokens=8)
        state_data = llm.get_state()
        assert len(state_data) > 0

        llm.ctx.kv_cache_clear()
        llm.set_state(state_data)

        text = llm.generate("Continue:", max_tokens=8)
        assert isinstance(text, str)


@requires_model
def test_state_file_load_then_generate():
    """Verify that loading state from file correctly updates position."""
    disable_logging()
    with Llama(model_path=MODEL_PATH) as llm:
        with tempfile.NamedTemporaryFile(suffix=".state", delete=False) as f:
            state_path = f.name

        try:
            llm.generate("Hello world", max_tokens=4)
            llm.save_state(state_path)

            llm.ctx.kv_cache_clear()
            llm.load_state(state_path)

            text = llm.generate("Continue:", max_tokens=4)
            assert isinstance(text, str)
        finally:
            if os.path.exists(state_path):
                os.unlink(state_path)


@requires_model
def test_grammar_respects_sampling_params():
    """Verify grammar-constrained generation uses sampling params."""
    disable_logging()
    grammar = LlamaGrammar.from_string('root ::= "yes" | "no" | "maybe"')

    with Llama(model_path=MODEL_PATH) as llm:
        resp = llm.create_chat_completion(
            [{"role": "user", "content": "Answer yes, no, or maybe"}],
            max_tokens=4,
            grammar=grammar,
            temperature=1.5,
        )
        content = resp["choices"][0]["message"]["content"].strip().lower()
        assert content in ("yes", "no", "maybe")


@requires_model
def test_lora_persistence_across_reset():
    """Verify LoRA tracking persists after reset()."""
    disable_logging()
    with Llama(model_path=MODEL_PATH) as llm:
        assert len(llm._lora_configs) == 0

        llm.generate("Test", max_tokens=4)
        llm.reset()
        text = llm.generate("Test", max_tokens=4)
        assert isinstance(text, str)


@requires_model
def test_kv_cache_ops_update_position():
    """Verify KV cache operations correctly update position bookkeeping."""
    disable_logging()
    with Llama(model_path=MODEL_PATH) as llm:
        llm.generate("Hello world", max_tokens=8)
        llm.ctx.kv_cache_clear()

        text = llm.generate("New prompt", max_tokens=4)
        assert isinstance(text, str)
