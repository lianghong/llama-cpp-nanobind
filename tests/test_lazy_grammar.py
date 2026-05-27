"""Tests for lazy grammar (llama_sampler_init_grammar_lazy_patterns)."""

import pytest

from llama_cpp import LlamaGrammar
from llama_cpp.llama import ValidationError

from conftest import requires_model


# --- Pure construction & validation (no model required) ---


def test_eager_grammar_is_not_lazy():
    g = LlamaGrammar.from_string('root ::= "yes" | "no"')
    assert g.is_lazy is False


def test_lazy_with_pattern_marks_lazy():
    g = LlamaGrammar.lazy(
        'root ::= "{" "}"',
        trigger_patterns=[r"<tool>"],
    )
    assert g.is_lazy is True
    assert g._trigger_patterns == ["<tool>"]
    assert g._trigger_tokens == []


def test_lazy_with_tokens_marks_lazy():
    g = LlamaGrammar.lazy(
        'root ::= "{" "}"',
        trigger_tokens=[123, 456],
    )
    assert g.is_lazy is True
    assert g._trigger_tokens == [123, 456]


def test_lazy_with_both_pattern_and_tokens():
    g = LlamaGrammar.lazy(
        'root ::= "x"',
        trigger_patterns=[r"<json>", r"<call>"],
        trigger_tokens=[7],
    )
    assert g._trigger_patterns == ["<json>", "<call>"]
    assert g._trigger_tokens == [7]


def test_lazy_without_triggers_rejected():
    """A lazy grammar with no trigger would never activate — must raise."""
    with pytest.raises(ValidationError, match="trigger"):
        LlamaGrammar.lazy('root ::= "x"')


def test_lazy_with_empty_lists_rejected():
    with pytest.raises(ValidationError, match="trigger"):
        LlamaGrammar.lazy('root ::= "x"', trigger_patterns=[], trigger_tokens=[])


def test_constructor_kwargs_path():
    """The constructor also accepts trigger_patterns/tokens directly."""
    g = LlamaGrammar('root ::= "x"', trigger_patterns=["<x>"])
    assert g.is_lazy is True


# --- End-to-end with a model ---


@requires_model
def test_lazy_grammar_constructs_native_sampler(llm):
    """_ensure_sampler must succeed for a lazy grammar (separate code path
    from eager grammar — exercises the lazy_patterns C entry point).
    """
    g = LlamaGrammar.lazy(
        'root ::= "{" "}"',
        trigger_patterns=[r"<tool_call>"],
    )
    g._ensure_sampler(llm.model)
    assert g._sampler is not None


@requires_model
def test_lazy_grammar_does_not_constrain_until_trigger(llm):
    """With a trigger that won't appear in the output, the grammar should
    stay inactive and generation should produce normal text (not be forced
    into the grammar's tiny vocabulary).
    """
    # Grammar that would force "yes"/"no" if active.
    g = LlamaGrammar.lazy(
        'root ::= "yes" | "no"',
        trigger_patterns=[r"__NEVER_MATCHES_THIS_SENTINEL__"],
    )
    out = llm.create_chat_completion(
        [{"role": "user", "content": "Say hello in one short sentence."}],
        max_tokens=12,
        grammar=g,
        seed=7,
    )
    text = out["choices"][0]["message"]["content"]
    assert isinstance(text, str)
    assert len(text) > 0
    # If the grammar were active from the start the model would be locked
    # into "yes" or "no"; with no trigger match, it should produce free-form
    # text. We assert the negation rather than positive content (more
    # robust across model checkpoints).
    assert text.strip() not in {"yes", "no"}
