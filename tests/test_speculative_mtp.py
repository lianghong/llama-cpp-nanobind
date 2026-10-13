"""End-to-end tests for draft-MTP speculative decoding.

These tests require an MTP-capable Qwen3.6-MoE checkpoint
(`nextn_predict_layers > 0` + `blk.*.nextn.*` tensors). The default
location is `models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`, overridable via
`LLAMA_MTP_TEST_MODEL`. Tests skip cleanly when absent.
"""

import os

import pytest
from conftest import MTP_MODEL_PATH, requires_mtp_model

from llama_cpp import (
    Llama,
    LlamaConfig,
    SamplingParams,
)


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
@pytest.mark.parametrize("prompt_count", [12, 24, 70])
@pytest.mark.parametrize("budget", [1, 8])
def test_first_callback_precedes_final_draft_prefill(prompt_count, budget):
    """Earlier chunks catch up immediately; the last waits for the callback."""
    from llama_cpp import _llama

    with _make_mtp_llm(n_batch=32, n_ubatch=16) as llm:
        sp = SamplingParams(temperature=0.0, seed=0, n_draft_max=2)
        prompt = llm.tokenize(
            "A red ball and a blue box. " * 10 + "The ball is",
            add_special=False,
        )[:prompt_count]
        assert len(prompt) == prompt_count

        def ordinary(tokens, count):
            llm.kv_cache_clear()
            return _llama.generate_tokens(
                llm.ctx,
                llm._build_sampler(sp),
                tokens,
                count,
                False,
                llm.model.eos(),
                [],
            )

        expected = ordinary(prompt, budget)
        assert expected
        llm.kv_cache_clear()
        llm.perf_reset()
        seen = []
        prefilled_at_first = []

        def callback(token):
            if not seen:
                prefilled_at_first.append(llm.perf()["n_mtp_prefill"])
            seen.append(token)
            return True

        actual = _llama.generate_tokens_speculative_mtp(
            llm.ctx,
            llm._build_sampler(sp),
            None,
            prompt,
            budget,
            False,
            llm.model.eos(),
            2,
            [],
            callback,
            0,
        )
        assert seen == actual == expected
        final_chunk = (prompt_count - 1) % 16 + 1
        assert prefilled_at_first == [prompt_count - final_chunk]
        assert llm.perf()["n_mtp_prefill"] == prompt_count
        # A one-token exit must still process that last chunk and its final
        # sampled token, leaving the pending hidden row valid for continuation.
        prefix = prompt + actual
        cached_ordinary = _llama.generate_tokens(
            llm.ctx,
            llm._build_sampler(sp),
            prefix,
            4,
            False,
            llm.model.eos(),
            [],
            len(prefix),
        )
        # Replay the same MTP prefill to reproduce KV AND current logits.
        # Serialized state only restores KV; a fresh whole-prefix prefill can
        # change near-tied greedy choices because recurrent batching differs.
        llm.kv_cache_clear()
        assert (
            _llama.generate_tokens_speculative_mtp(
                llm.ctx,
                llm._build_sampler(sp),
                None,
                prompt,
                budget,
                False,
                llm.model.eos(),
                2,
                [],
                None,
                0,
            )
            == actual
        )
        continuation = _llama.generate_tokens_speculative_mtp(
            llm.ctx,
            llm._build_sampler(sp),
            None,
            prefix,
            4,
            False,
            llm.model.eos(),
            2,
            [],
            None,
            len(prefix),
        )
        assert continuation == cached_ordinary


@requires_mtp_model
def test_speculative_reuses_graph_mode_after_kv_clear():
    """Repeated requests retain graph shape while the invalid driver is rebuilt."""
    sp = SamplingParams(temperature=0.0, seed=0, n_draft_max=2)
    with _make_mtp_llm() as llm:
        prompt = "Count from one to twenty:"
        expected = llm.generate(prompt, max_tokens=16, sampling=sp)
        llm.perf_reset()
        for _ in range(3):
            actual = llm.generate(prompt, max_tokens=16, sampling=sp, speculative=True)
            assert actual == expected
            assert llm.perf()["n_mtp_graph_switches"] == 1

        # Ordinary generation restores its graph mode once.
        assert llm.generate(prompt, max_tokens=16, sampling=sp) == expected
        assert llm.perf()["n_mtp_graph_switches"] == 2
        assert llm.generate(prompt, max_tokens=16, sampling=sp) == expected
        assert llm.perf()["n_mtp_graph_switches"] == 2
        assert (
            llm.generate(prompt, max_tokens=16, sampling=sp, speculative=True)
            == expected
        )
        assert llm.perf()["n_mtp_graph_switches"] == 3


@requires_mtp_model
@pytest.mark.parametrize("raise_error", [False, True])
@pytest.mark.parametrize("draft_max", [1, 2, 3, 4])
def test_speculative_abnormal_callback_restores_graph_mode(
    mtp_llm, raise_error, draft_max
):
    from llama_cpp import _llama

    llm = mtp_llm
    sp = SamplingParams(temperature=0.0, seed=0, n_draft_max=draft_max)
    expected = llm.generate(_CONT_PROMPT, max_tokens=4, sampling=sp)
    llm.kv_cache_clear()
    llm.perf_reset()

    def callback(token):
        if raise_error:
            raise RuntimeError("callback failed")
        return False

    def run():
        return _llama.generate_tokens_speculative_mtp(
            llm.ctx,
            llm._build_sampler(sp),
            None,
            llm.tokenize(_CONT_PROMPT, add_special=False),
            8,
            llm._effective_add_bos,
            llm.model.eos(),
            draft_max,
            [],
            callback,
            0,
        )

    if raise_error:
        with pytest.raises(RuntimeError, match="callback failed"):
            run()
    else:
        assert run() == []
    assert llm.ctx.kv_cache_seq_pos_max(0) == -1
    assert llm.perf()["n_mtp_graph_switches"] == 2
    assert llm.perf()["n_mtp_prefill"] == 0
    assert llm.generate(_CONT_PROMPT, max_tokens=4, sampling=sp) == expected
    assert llm.perf()["n_mtp_graph_switches"] == 2
    assert (
        llm.generate(_CONT_PROMPT, max_tokens=4, sampling=sp, speculative=True)
        == expected
    )
    assert llm.perf()["n_mtp_graph_switches"] == 3


@requires_mtp_model
@pytest.mark.parametrize("draft_max", [1, 2, 3, 4])
def test_first_token_stop_skips_final_draft_prefill(mtp_llm, draft_max):
    from llama_cpp import _llama

    llm = mtp_llm
    sp = SamplingParams(temperature=0.0, seed=0, n_draft_max=draft_max)
    prompt = llm.tokenize(_CONT_PROMPT, add_special=False)
    llm.kv_cache_clear()
    expected = _llama.generate_tokens(
        llm.ctx, llm._build_sampler(sp), prompt, 4, False, llm.model.eos(), []
    )
    assert expected
    llm.kv_cache_clear()
    llm.perf_reset()
    seen = []
    result = _llama.generate_tokens_speculative_mtp(
        llm.ctx,
        llm._build_sampler(sp),
        None,
        prompt,
        8,
        False,
        llm.model.eos(),
        draft_max,
        [[expected[0]]],
        lambda token: seen.append(token) is None,
        0,
    )
    assert result == seen == []
    assert llm.perf()["n_mtp_prefill"] == 0
    assert llm.ctx.kv_cache_seq_pos_max(0) == -1
    assert (
        _llama.generate_tokens_speculative_mtp(
            llm.ctx,
            llm._build_sampler(sp),
            None,
            prompt,
            4,
            False,
            llm.model.eos(),
            draft_max,
            [],
            None,
            0,
        )
        == expected
    )


@requires_mtp_model
@pytest.mark.parametrize("method", ["decode", "decode_one", "decode_multi"])
def test_ordinary_decode_restores_graph_mode(mtp_llm, method):
    llm = mtp_llm
    sp = SamplingParams(temperature=0.0, seed=0)
    expected = llm.generate(_CONT_PROMPT, max_tokens=4, sampling=sp)
    llm.perf_reset()
    llm.generate(_CONT_PROMPT, max_tokens=4, sampling=sp, speculative=True)
    llm.kv_cache_clear()
    assert llm.perf()["n_mtp_graph_switches"] == 1
    token = llm.tokenize("Hello", add_special=False)[0]
    getattr(llm.ctx, method)(token if method == "decode_one" else [token])
    assert llm.perf()["n_mtp_graph_switches"] == 2
    assert llm.generate(_CONT_PROMPT, max_tokens=4, sampling=sp) == expected
    assert llm.perf()["n_mtp_graph_switches"] == 2


@requires_mtp_model
def test_speculative_graph_mode_after_context_reset(mtp_llm):
    llm = mtp_llm
    sp = SamplingParams(temperature=0.0, seed=0)
    expected = llm.generate(_CONT_PROMPT, max_tokens=4, sampling=sp)
    llm.perf_reset()
    llm.generate(_CONT_PROMPT, max_tokens=4, sampling=sp, speculative=True)
    assert llm.perf()["n_mtp_graph_switches"] == 1
    llm.reset()
    assert (
        llm.generate(_CONT_PROMPT, max_tokens=4, sampling=sp, speculative=True)
        == expected
    )
    assert llm.perf()["n_mtp_graph_switches"] == 2
    assert llm.generate(_CONT_PROMPT, max_tokens=4, sampling=sp) == expected
    assert llm.perf()["n_mtp_graph_switches"] == 3


@requires_mtp_model
def test_speculative_performance_counters():
    """Accepted/proposed counts survive KV clearing and reset with perf_reset."""
    with _make_mtp_llm() as llm:
        fields = ("n_drafted", "n_accepted", "n_verify", "n_mtp_prefill")
        llm.perf_reset()
        assert all(llm.perf()[key] == 0 for key in fields)
        llm.generate(
            "Count from one to twenty:",
            max_tokens=24,
            sampling=SamplingParams(temperature=0.0, seed=0, n_draft_max=2),
            speculative=True,
        )
        metrics = llm.perf()
        assert metrics["n_mtp_prefill"] > 0
        assert metrics["n_verify"] > 0
        assert 0 < metrics["n_drafted"] <= 2 * metrics["n_verify"]
        assert 0 <= metrics["n_accepted"] <= metrics["n_drafted"]
        llm.kv_cache_clear()
        assert all(llm.perf()[key] == metrics[key] for key in fields)
        llm.perf_reset()
        assert all(llm.perf()[key] == 0 for key in fields)
        llm.generate("Hello", max_tokens=2)
        assert all(llm.perf()[key] == 0 for key in fields)


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

    n_rs_seq must be >= n_draft_max so the recurrent target context can
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
# Differential continuation tests compare each mode transition with a fresh
# prime. The shared model keeps GPU residency to one checkpoint at a time.

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
    """Every mode transition must match a fresh prime of the same prefix."""
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
    """Extreme early termination: max_tokens=1 returns one target token. Must return non-empty text and keep
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
@pytest.mark.parametrize("draft_max", [1, 2, 3, 4])
def test_speculative_streaming_callback_cancel(mtp_llm, draft_max):
    """Cancelling a speculative stream early (consumer stops iterating) must
    not crash and must leave the instance reusable."""
    sp = SamplingParams(seed=0, temperature=0.0, n_draft_max=draft_max)
    expected = mtp_llm.generate("Hello", max_tokens=4, sampling=sp)
    collected = []
    stream = mtp_llm.generate_stream(
        "Tell me a long story about a robot.",
        max_tokens=64,
        sampling=sp,
        speculative=True,
        reset_kv_cache=True,
    )
    try:
        for chunk in stream:
            collected.append(chunk)
            if len(collected) >= 3:
                break  # early cancellation
    finally:
        stream.close()
    assert len(collected) == 3
    # Instance must remain usable after an early-cancelled stream.
    out = mtp_llm.generate(
        "Hello",
        max_tokens=4,
        sampling=sp,
        speculative=True,
        reset_kv_cache=True,
    )
    assert out == expected
    assert mtp_llm.generate("Hello", max_tokens=4, sampling=sp) == expected


@requires_mtp_model
def test_speculative_prefill_exceeds_batch():
    """Every prefill chunk must feed its hidden rows to MTP before the next."""
    llm = _make_mtp_llm(n_batch=32, n_ubatch=16)
    try:
        prompt = "A red ball and a blue box. " * 12 + "The ball is"
        sp = SamplingParams(temperature=0.0, seed=0)
        assert len(llm.tokenize(prompt)) > llm.config.n_batch
        baseline = llm.generate(prompt, max_tokens=16, sampling=sp)
        result = llm.generate(prompt, max_tokens=16, sampling=sp, speculative=True)
        assert result == baseline
    finally:
        llm.close()


@requires_mtp_model
@pytest.mark.parametrize("draft_max", [1, 2, 3, 4])
def test_speculative_stream_holds_stop_prefix(mtp_llm, draft_max):
    """Use actual generated token IDs so the multi-token stop definitely fires."""
    from llama_cpp import _llama

    llm = mtp_llm
    sp = SamplingParams(temperature=0.0, seed=0, n_draft_max=draft_max)
    prompt = llm.tokenize("Count from one to twenty:", add_special=False)
    llm.kv_cache_clear()
    baseline = _llama.generate_tokens_speculative_mtp(
        llm.ctx,
        llm._build_sampler(sp),
        None,
        prompt,
        24,
        llm._effective_add_bos,
        llm.model.eos(),
        draft_max,
        [],
        None,
        0,
    )
    assert len(baseline) >= 8
    stop = baseline[3:6]
    # Use the first occurrence, in case the model generated a repetition.
    stop_at = next(
        i
        for i in range(len(baseline) - len(stop) + 1)
        if baseline[i : i + len(stop)] == stop
    )
    seen = []
    llm.kv_cache_clear()
    result = _llama.generate_tokens_speculative_mtp(
        llm.ctx,
        llm._build_sampler(sp),
        None,
        prompt,
        24,
        llm._effective_add_bos,
        llm.model.eos(),
        draft_max,
        [stop],
        lambda token: seen.append(token) is None,
        0,
    )
    assert seen == result == baseline[:stop_at]


@requires_mtp_model
@pytest.mark.parametrize("prompt_count", [512, 2048, 4096])
@pytest.mark.parametrize("draft_max", [1, 2, 3, 4])
def test_speculative_long_prompt_stop_and_cancel(prompt_count, draft_max):
    """Stops and callback cancellation must recover at each sweep setting."""
    from llama_cpp import _llama

    cfg = LlamaConfig(
        model_path=MTP_MODEL_PATH,
        n_ctx=8192,
        n_batch=512,
        n_ubatch=512,
        n_rs_seq=4,
        n_gpu_layers=-1,
        verbose=False,
    )
    with Llama(MTP_MODEL_PATH, config=cfg) as llm:
        sp = SamplingParams(temperature=0.0, seed=0, n_draft_max=draft_max)
        prompt = llm.tokenize(
            "A red ball and a blue box. " * prompt_count,
            add_special=False,
        )[:prompt_count]
        assert len(prompt) == prompt_count

        def run(stops, callback):
            return _llama.generate_tokens_speculative_mtp(
                llm.ctx,
                llm._build_sampler(sp),
                None,
                prompt,
                16,
                False,
                llm.model.eos(),
                draft_max,
                stops,
                callback,
                0,
            )

        expected = run([], None)
        assert len(expected) >= 8
        stop = expected[4:7]
        stop_at = next(
            i
            for i in range(len(expected) - len(stop) + 1)
            if expected[i : i + len(stop)] == stop
        )
        seen = []
        llm.kv_cache_clear()
        stopped = run([stop], lambda token: seen.append(token) is None)
        assert stopped == seen == expected[:stop_at]
        assert llm.ctx.kv_cache_seq_pos_max(0) == -1

        seen = []

        def cancel(token):
            if len(seen) == 4:
                return False
            seen.append(token)
            return True

        assert run([], cancel) == seen == expected[:4]
        assert llm.ctx.kv_cache_seq_pos_max(0) == -1
        assert run([], None) == expected


@requires_mtp_model
@pytest.mark.parametrize("budget", [1, 2, 5, 9])
def test_speculative_finishes_with_current_logits(mtp_llm, budget):
    llm = mtp_llm
    sp = SamplingParams(temperature=0.0, seed=0, n_draft_max=4)
    out = llm.generate(_CONT_PROMPT, max_tokens=budget, sampling=sp, speculative=True)
    assert llm.ctx.kv_cache_seq_pos_max(0) + 1 == len(llm._cached_prompt_tokens)
    # An unchanged full-prefix continuation must sample current target logits.
    continuation = llm.generate(
        _CONT_PROMPT + out,
        max_tokens=4,
        sampling=sp,
        reset_kv_cache=False,
        speculative=True,
    )
    fresh = llm.generate(_CONT_PROMPT + out, max_tokens=4, sampling=sp)
    assert continuation == fresh


@requires_mtp_model
def test_speculative_near_context_limit(mtp_llm):
    """The draft driver must not decode its fixed draft width past n_ctx."""
    from llama_cpp import _llama

    llm = mtp_llm
    sp = SamplingParams(temperature=0.0, seed=0, n_draft_max=8)
    token = llm.tokenize(" a", add_special=False)[0]
    prompt = [token] * (llm.n_ctx() - 3)
    llm.kv_cache_clear()
    result = _llama.generate_tokens_speculative_mtp(
        llm.ctx,
        llm._build_sampler(sp),
        None,
        prompt,
        8,
        False,
        llm.model.eos(),
        8,
        [],
        None,
        0,
    )
    assert len(result) <= 3
    assert llm.ctx.kv_cache_seq_pos_max(0) < llm.n_ctx()


@requires_mtp_model
def test_speculative_continuation_invalidates_device_snapshot(mtp_llm):
    from llama_cpp import LlamaError

    llm = mtp_llm
    sp = SamplingParams(temperature=0.0, seed=0)
    text = llm.generate(_CONT_PROMPT, max_tokens=4, sampling=sp, speculative=True)
    handle = llm.save_seq_state_on_device()
    llm.generate(
        _CONT_PROMPT + text,
        max_tokens=4,
        sampling=sp,
        speculative=True,
        reset_kv_cache=False,
    )
    with pytest.raises(LlamaError, match="stale"):
        llm.load_seq_state_on_device(handle)
