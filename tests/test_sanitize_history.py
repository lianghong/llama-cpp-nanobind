"""Tests for UnifiedLLM.sanitize_history and backend.strip_thinking.

Covers the Unsloth multi-turn hygiene rule ("do not feed prior thought blocks
back into the next turn") without requiring a real model load. We exercise
ChatTemplateBackend._parse_thinking / strip_thinking directly against Gemma 4
channel blocks and Qwen-style <think> tags.
"""

from llama_cpp.unified import ChatTemplateBackend


def _bare_backend() -> ChatTemplateBackend:
    """Instantiate ChatTemplateBackend without running __init__.

    strip_thinking / _parse_thinking only touch class-level regex patterns,
    so we don't need a real Llama / ModelConfig.
    """
    return ChatTemplateBackend.__new__(ChatTemplateBackend)


# ---------------------------------------------------------------------------
# strip_thinking — pattern-level coverage
# ---------------------------------------------------------------------------


def test_strip_thinking_gemma4_channel_block():
    """Gemma 4 reasoning lives inside <|channel>...<channel|>.

    Unsloth: strip before sending history back, keep only the visible answer.
    """
    raw = "<|channel>reasoning step 1\nreasoning step 2<channel|>Final answer."
    out = _bare_backend().strip_thinking(raw)
    assert "reasoning step" not in out
    assert "Final answer." in out


def test_strip_thinking_qwen_think_tags():
    raw = "<think>let me consider...\n</think>\n\nThe answer is 42."
    out = _bare_backend().strip_thinking(raw)
    assert "let me consider" not in out
    assert "The answer is 42." in out


def test_strip_thinking_no_thinking_markup_passthrough():
    raw = "just a plain response"
    out = _bare_backend().strip_thinking(raw)
    assert out.strip() == "just a plain response"


def test_strip_thinking_bracket_style():
    raw = "[THINK]inner monologue[/THINK]visible part"
    out = _bare_backend().strip_thinking(raw)
    assert "inner monologue" not in out
    assert "visible part" in out


# ---------------------------------------------------------------------------
# sanitize_history — shape / role preservation
# ---------------------------------------------------------------------------


class _StubBackend:
    """Minimal backend that records strip_thinking calls."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def strip_thinking(self, text: str) -> str:
        self.calls.append(text)
        # Mimic the real strip: chop anything before "###ANSWER:" if present.
        marker = "###ANSWER:"
        if marker in text:
            return text.split(marker, 1)[1].strip()
        return text


class _StubUnified:
    """Minimal UnifiedLLM-like object exposing sanitize_history via the real method."""

    def __init__(self) -> None:
        self.backend = _StubBackend()

    # Bind the real method off the class
    from llama_cpp.unified import UnifiedLLM  # noqa: PLC0415

    sanitize_history = UnifiedLLM.sanitize_history


def test_sanitize_history_strips_assistant_only():
    llm = _StubUnified()
    history = [
        {"role": "system", "content": "<|think|>You are helpful."},
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "###THINK### noise ###ANSWER: real reply"},
        {"role": "user", "content": "Q2"},
    ]
    cleaned = llm.sanitize_history(history)

    # Only assistant messages go through strip_thinking.
    assert llm.backend.calls == ["###THINK### noise ###ANSWER: real reply"]

    # Role ordering preserved.
    assert [m["role"] for m in cleaned] == ["system", "user", "assistant", "user"]

    # System / user passed through unchanged.
    assert cleaned[0]["content"] == "<|think|>You are helpful."
    assert cleaned[1]["content"] == "Q1"
    assert cleaned[3]["content"] == "Q2"

    # Assistant content has thinking stripped.
    assert cleaned[2]["content"] == "real reply"


def test_sanitize_history_does_not_mutate_input():
    llm = _StubUnified()
    original = [
        {"role": "assistant", "content": "###ANSWER: hi"},
    ]
    snapshot = [dict(m) for m in original]
    _ = llm.sanitize_history(original)
    assert original == snapshot


def test_sanitize_history_preserves_extra_keys():
    llm = _StubUnified()
    history = [
        {
            "role": "assistant",
            "content": "###ANSWER: yes",
            "name": "bot",
            "tool_calls": [],
        },
    ]
    cleaned = llm.sanitize_history(history)
    assert cleaned[0]["name"] == "bot"
    assert cleaned[0]["tool_calls"] == []
    assert cleaned[0]["content"] == "yes"


def test_sanitize_history_non_string_content_passes_through():
    """OpenAI-style multimodal content (list of parts) shouldn't be strip-called."""
    llm = _StubUnified()
    history = [
        {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
    ]
    cleaned = llm.sanitize_history(history)
    # Non-string content left intact; strip_thinking not invoked.
    assert llm.backend.calls == []
    assert cleaned[0]["content"] == [{"type": "text", "text": "hi"}]
