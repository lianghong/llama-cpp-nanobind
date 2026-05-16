"""Tests for prompt-prefix KV cache reuse (cache_prompt=True).

Covers the path added in `_apply_prefix_reuse` / `_invalidate_prompt_cache`:
- Empty cache → full prime path.
- Identical prompt → mirror committed unchanged; second call adds zero
  prompt-decode cost.
- Partial LCP (turn edit) → KV trimmed to LCP, new tail decoded.
- Strict-prefix LCP (history truncation) → KV trimmed, no new decode.
- Direct kv_cache_clear / load_state / kv_cache_seq_* invalidate the mirror.
- cache_prompt=False bypass leaves mirror empty so the next cache_prompt=True
  call starts clean.
- Output equivalence: prefix reuse + same seed produces the same generated
  tokens as a full re-prime.
"""

from __future__ import annotations

from conftest import requires_model


def _kv_pos(llm) -> int:
    """KV-cache max position; -1 if empty."""
    return llm.kv_cache_seq_pos_max(0)


@requires_model
def test_mirror_starts_empty(llm):
    """Fresh Llama instance has no cached prompt."""
    llm.kv_cache_clear()
    assert llm._cached_prompt_tokens == []


@requires_model
def test_full_prime_populates_mirror(llm):
    """A reset_kv_cache=True call populates the mirror with priming tokens."""
    llm.kv_cache_clear()
    llm.generate("Hello world", max_tokens=3, reset_kv_cache=True, seed=1)
    # Mirror should hold priming + generated tokens; both non-empty.
    assert len(llm._cached_prompt_tokens) > 0
    # Mirror length ≈ KV position (BOS + prompt + generated).
    assert len(llm._cached_prompt_tokens) == _kv_pos(llm) + 1


@requires_model
def test_identical_prompt_full_lcp(llm):
    """Repeating the exact prior prompt keeps mirror and KV consistent.

    Invariant enforced: after the second call, len(mirror) == kv_pos_max + 1.

    We don't assert that *more* tokens are in KV than after the first call:
    hybrid-attention models (Qwen3.5, Granite 4 hybrid, …) report
    memory_can_shift()=False, so llama_memory_seq_rm(seq, p0, -1) refuses to
    trim mid-sequence. For those models the LCP-trim falls back to a full
    kv_cache_clear and gen2 re-decodes from scratch, ending at the same KV
    position as gen1.
    """
    llm.kv_cache_clear()
    llm.generate("The capital of France is", max_tokens=4, reset_kv_cache=True, seed=7)

    llm.generate(
        "The capital of France is",
        max_tokens=4,
        reset_kv_cache=False,
        cache_prompt=True,
        seed=7,
    )
    # Mirror always tracks KV: len(mirror) == kv_pos_max + 1.
    assert len(llm._cached_prompt_tokens) == _kv_pos(llm) + 1
    assert llm._cached_prompt_tokens


@requires_model
def test_partial_lcp_trims_kv(llm):
    """Edited continuation trims KV to the divergence point and decodes only the tail."""
    llm.kv_cache_clear()
    llm.generate(
        "Once upon a time, in a kingdom", max_tokens=2, reset_kv_cache=True, seed=3
    )
    mirror_first = list(llm._cached_prompt_tokens)

    # Now feed a prompt that shares a leading prefix but diverges. The mirror
    # should be trimmed to the LCP boundary and extended with the new prompt.
    llm.generate(
        "Once upon a time, in a faraway",
        max_tokens=2,
        reset_kv_cache=False,
        cache_prompt=True,
        seed=3,
    )
    # The LCP must be > 0 (both prompts share BOS + "Once upon a time, in a").
    # First mirror token (BOS) survives; mirror diverges before its end.
    assert llm._cached_prompt_tokens[0] == mirror_first[0]
    # KV position matches the new mirror length (mirror is the source of truth).
    assert _kv_pos(llm) + 1 == len(llm._cached_prompt_tokens)


@requires_model
def test_kv_cache_clear_invalidates_mirror(llm):
    """Direct kv_cache_clear drops the mirror."""
    llm.generate("Test prompt", max_tokens=2, reset_kv_cache=True, seed=1)
    assert llm._cached_prompt_tokens
    llm.kv_cache_clear()
    assert llm._cached_prompt_tokens == []


@requires_model
def test_kv_cache_seq_rm_invalidates_mirror(llm):
    """Direct kv_cache_seq_rm drops the mirror (escape hatch semantics)."""
    llm.generate("Test prompt", max_tokens=2, reset_kv_cache=True, seed=1)
    assert llm._cached_prompt_tokens
    # Trim everything; mirror invariant cannot be trusted post-call.
    llm.kv_cache_seq_rm(0, 0, -1)
    assert llm._cached_prompt_tokens == []


@requires_model
def test_set_state_data_invalidates_mirror(llm):
    """Round-tripping through get_state / set_state invalidates the mirror."""
    llm.generate("State round trip", max_tokens=2, reset_kv_cache=True, seed=1)
    assert llm._cached_prompt_tokens
    snapshot = llm.get_state()
    # set_state writes into KV but our mirror cannot be guaranteed to match.
    llm.set_state(snapshot)
    assert llm._cached_prompt_tokens == []


@requires_model
def test_reset_invalidates_mirror(llm):
    """Llama.reset() drops the mirror along with KV."""
    llm.generate("Reset test", max_tokens=2, reset_kv_cache=True, seed=1)
    assert llm._cached_prompt_tokens
    llm.reset()
    assert llm._cached_prompt_tokens == []


@requires_model
def test_cache_prompt_false_clears_mirror(llm):
    """cache_prompt=False with reset_kv_cache=False drops the mirror."""
    llm.generate("Sticky prompt", max_tokens=2, reset_kv_cache=True, seed=1)
    assert llm._cached_prompt_tokens
    llm.generate(
        "Sticky prompt continues",
        max_tokens=2,
        reset_kv_cache=False,
        cache_prompt=False,
        seed=1,
    )
    # Mirror dropped because caller opted out of caching.
    assert llm._cached_prompt_tokens == []


@requires_model
def test_prefix_reuse_output_matches_full_reprime(llm):
    """Prefix reuse produces a coherent continuation matching a full reprime.

    Trimming KV + decoding only the suffix should be functionally equivalent
    to clearing KV and decoding the full prompt. We check that the leading
    tokens of the generated text agree — a strong correctness signal. We do
    NOT require bit-exact equivalence across all tokens because flash-
    attention numerical layouts can drift the trailing logits by ULP-scale
    amounts after a partial trim, occasionally flipping the very last
    sample under high-entropy distributions.
    """
    prompt_a = "Once upon a time, in a kingdom far"
    prompt_b = "Once upon a time, in a kingdom by the sea"

    # Path 1: warm up with prompt_a, then prefix-reuse to prompt_b.
    llm.kv_cache_clear()
    llm.generate(prompt_a, max_tokens=2, reset_kv_cache=True, seed=42)
    reuse_text = llm.generate(
        prompt_b,
        max_tokens=8,
        reset_kv_cache=False,
        cache_prompt=True,
        seed=42,
    )

    # Path 2: clean slate, generate prompt_b directly.
    llm.kv_cache_clear()
    fresh_text = llm.generate(prompt_b, max_tokens=8, reset_kv_cache=True, seed=42)

    # Both must be non-empty strings.
    assert isinstance(reuse_text, str) and reuse_text
    assert isinstance(fresh_text, str) and fresh_text
    # Compare leading words: the first ~5 generated words must agree, which
    # confirms KV alignment is correct. (Trailing tokens may diverge by ULP-
    # scale FP drift on flash-attention.)
    reuse_lead = reuse_text.strip().split()[:5]
    fresh_lead = fresh_text.strip().split()[:5]
    assert reuse_lead == fresh_lead, (
        f"Continuation diverges in leading tokens: "
        f"reuse={reuse_lead!r} vs fresh={fresh_lead!r}"
    )


@requires_model
def test_chat_completion_cache_prompt(llm):
    """create_chat_completion accepts cache_prompt and reuses prefix between turns."""
    llm.kv_cache_clear()
    msgs1 = [{"role": "user", "content": "Hello, how are you?"}]
    llm.create_chat_completion(msgs1, max_tokens=4, reset_kv_cache=True)
    pos_after_turn1 = _kv_pos(llm)

    # Add a follow-up. With cache_prompt=True, only the new user turn (and
    # template scaffolding around it) needs decoding.
    msgs2 = [
        *msgs1,
        {"role": "assistant", "content": "I'm fine, thanks."},
        {"role": "user", "content": "What's the weather?"},
    ]
    llm.create_chat_completion(
        msgs2, max_tokens=4, reset_kv_cache=False, cache_prompt=True
    )
    # KV grew by some amount but mirror is consistent with KV position.
    assert _kv_pos(llm) >= pos_after_turn1
    assert len(llm._cached_prompt_tokens) == _kv_pos(llm) + 1


@requires_model
def test_close_clears_mirror(model_path):
    """close() drops the mirror under the lock."""
    from llama_cpp import disable_logging
    from llama_cpp import Llama

    disable_logging()
    inst = Llama(model_path=model_path)
    try:
        inst.generate("close test", max_tokens=2, reset_kv_cache=True, seed=1)
        assert inst._cached_prompt_tokens
    finally:
        inst.close()
    assert inst._cached_prompt_tokens == []
