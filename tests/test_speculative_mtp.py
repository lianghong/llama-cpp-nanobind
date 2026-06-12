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

from conftest import MTP_MODEL_PATH, requires_mtp_model


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


# --- Continuation / mode-switch regression tests ---------------------------
# These guard the load-bearing speculative off-by-1 (user KV ends 1 behind the
# prompt-cache mirror; the next speculative turn self-heals) and the mode-switch
# guard that forces a KV reset across a speculative↔non-speculative boundary.
# A length invariant alone never caught these — only a differential
# "continuation output == fresh full re-prime" check does.
#
# All tests below share ONE module-scoped 35B instance. Each test resets KV
# (reset_kv_cache=True) at its start, so sharing is safe — and it keeps the
# 35B model's VRAM footprint to a single load, which matters because the full
# suite already loads it several times and a 20GB card has no headroom for the
# extra per-test loads (transient OOM → ModelLoadError elsewhere in the suite).

_CONT_PROMPT = "The capital of France is"


@pytest.fixture(scope="module")
def mtp_llm():
    if not os.path.exists(MTP_MODEL_PATH):
        pytest.skip("MTP-capable test model not found")
    instance = _make_mtp_llm(n_rs_seq=8)
    yield instance
    instance.close()


def _spec_continue(llm, first_spec, second_spec):
    """Run turn1 (first_spec), continue with turn2 (second_spec) reusing KV,
    then re-run turn2's exact text from a clean KV on the SAME instance.
    Returns (cont, fresh). Greedy + deterministic, so a correct continuation
    must byte-match the fresh re-prime.
    """
    sp = SamplingParams(seed=0, temperature=0.0, n_draft_max=4)
    g1 = llm.generate(
        _CONT_PROMPT,
        max_tokens=10,
        sampling=sp,
        speculative=first_spec,
        reset_kv_cache=True,
        cache_prompt=True,
    )
    cont = llm.generate(
        _CONT_PROMPT + g1,
        max_tokens=10,
        sampling=sp,
        speculative=second_spec,
        reset_kv_cache=False,
        cache_prompt=True,
    )
    # "fresh" = same text primed from a cleared KV on this same instance.
    fresh = llm.generate(
        _CONT_PROMPT + g1,
        max_tokens=10,
        sampling=sp,
        speculative=second_spec,
        reset_kv_cache=True,
        cache_prompt=True,
    )
    return cont, fresh


@requires_mtp_model
def test_speculative_continuation_modes(mtp_llm):
    """Differential check across all four mode transitions on the same shared
    instance:

    * spec→spec / nonspec→nonspec: same-mode continuation must byte-match a
      fresh re-prime (spec→spec exercises the load-bearing off-by-1 self-heal).
    * spec→nonspec: the guard heals the off-by-1 in place (decodes the undecoded
      tail token) and continues WITHOUT a reset — output correct, prefix reused.
    * nonspec→spec: the draft recurrent state can't be resumed, so the guard
      forces a reset; output must still be correct.
    """
    for first, second in [(True, True), (False, False), (True, False), (False, True)]:
        cont, fresh = _spec_continue(mtp_llm, first, second)
        assert cont == fresh, f"mode transition spec={first}->{second} diverged"


@requires_mtp_model
def test_speculative_to_nonspeculative_preserves_prefix(mtp_llm):
    """spec→nonspec continuation must HEAL in place (reuse the cached prefix),
    not full-reset. Guards against a regression back to the reset-everything
    approach: the mirror must retain the turn-1 prefix and extend it."""
    sp = SamplingParams(seed=0, temperature=0.0, n_draft_max=4)
    g1 = mtp_llm.generate(
        _CONT_PROMPT,
        max_tokens=10,
        sampling=sp,
        speculative=True,
        reset_kv_cache=True,
        cache_prompt=True,
    )
    mirror_before = list(mtp_llm._cached_prompt_tokens)
    mtp_llm.generate(
        _CONT_PROMPT + g1,
        max_tokens=8,
        sampling=sp,
        speculative=False,
        reset_kv_cache=False,
        cache_prompt=True,
    )
    mirror_after = list(mtp_llm._cached_prompt_tokens)
    # A full reset would have rebuilt the mirror from scratch; an in-place heal
    # keeps the turn-1 prefix as a strict prefix and extends it.
    assert len(mirror_after) > len(mirror_before)
    assert mirror_after[: len(mirror_before)] == mirror_before


@requires_mtp_model
def test_speculative_stop_then_continue_does_not_crash(mtp_llm):
    """Regression: speculative + multi-token stop followed by a speculative
    continuation previously aborted the process at
    speculative.cpp GGML_ASSERT(impl) because stop tokens were left in the C++
    mirror. Must now complete and stay deterministic across repeats."""
    ps = "Counting: 1 2 3 STOPHERE 4 5 6 7 8"
    sp = SamplingParams(seed=0, temperature=0.0, n_draft_max=4)
    conts = []
    for _ in range(2):
        gs = mtp_llm.generate(
            ps,
            max_tokens=64,
            sampling=sp,
            stop=["STOPHERE"],
            speculative=True,
            reset_kv_cache=True,
            cache_prompt=True,
        )
        assert "STOPHERE" not in gs
        cont = mtp_llm.generate(
            ps + gs,
            max_tokens=10,
            sampling=sp,
            speculative=True,
            reset_kv_cache=False,
            cache_prompt=True,
        )
        conts.append(cont)
    assert conts[0] == conts[1]  # deterministic, and no crash reaching here


@requires_mtp_model
def test_speculative_max_tokens_one(mtp_llm):
    """Extreme early termination: max_tokens=1 exits the loop after a single
    emitted token (the corrected_id tail). Must return non-empty text and keep
    the mirror aligned for a clean continuation."""
    sp = SamplingParams(seed=0, temperature=0.0, n_draft_max=4)
    one = mtp_llm.generate(
        _CONT_PROMPT,
        max_tokens=1,
        sampling=sp,
        speculative=True,
        reset_kv_cache=True,
        cache_prompt=True,
    )
    assert isinstance(one, str) and one != ""
    # Continuation after a 1-token speculative turn must stay coherent.
    cont = mtp_llm.generate(
        _CONT_PROMPT + one,
        max_tokens=8,
        sampling=sp,
        speculative=True,
        reset_kv_cache=False,
        cache_prompt=True,
    )
    assert isinstance(cont, str) and cont.strip()


@requires_mtp_model
def test_speculative_streaming_callback_cancel(mtp_llm):
    """Cancelling a speculative stream early (consumer stops iterating) must
    not crash and must leave the instance reusable."""
    sp = SamplingParams(seed=0, temperature=0.0, n_draft_max=4)
    collected = []
    for chunk in mtp_llm.generate_stream(
        "Tell me a long story about a robot.",
        max_tokens=64,
        sampling=sp,
        speculative=True,
        reset_kv_cache=True,
    ):
        collected.append(chunk)
        if len(collected) >= 3:
            break  # early cancellation
    assert collected
    # Instance must remain usable after an early-cancelled stream.
    out = mtp_llm.generate(
        "Hello",
        max_tokens=4,
        sampling=sp,
        speculative=True,
        reset_kv_cache=True,
    )
    assert isinstance(out, str)
