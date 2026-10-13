"""MTP cache retention across natural EOG and supported prefix edits."""

from pathlib import Path

import pytest
from conftest import MTP_MODEL_PATH, requires_mtp_model

from llama_cpp import Llama, LlamaConfig, SamplingParams, _llama

requires_cache_fixture = pytest.mark.skipif(
    Path(MTP_MODEL_PATH).name != "Qwen3.5-4B-Q4_K_M-MTP.gguf",
    reason="requires the calibrated Qwen3.5-4B MTP cache fixture",
)


@pytest.fixture
def cache_llm():
    with Llama(
        MTP_MODEL_PATH,
        config=LlamaConfig(
            model_path=MTP_MODEL_PATH,
            n_ctx=512,
            n_batch=32,
            n_ubatch=16,
            n_rs_seq=4,
            n_threads=4,
            n_threads_batch=4,
            n_gpu_layers=-1,
            verbose=False,
        ),
    ) as llm:
        yield llm


@requires_mtp_model
@pytest.mark.parametrize("prompt_size", [5, 70])
def test_first_natural_eog_retains_prefill(cache_llm, prompt_size):
    llm = cache_llm
    prompt = llm.tokenize("The capital of France is " * 20, add_special=False)[
        :prompt_size
    ]
    sampler = llm._build_sampler(
        SamplingParams(temperature=0.0, seed=0, logit_bias={llm.model.eos(): 1000.0})
    )
    llm.perf_reset()
    result = _llama.generate_tokens_speculative_mtp(
        llm.ctx,
        sampler,
        None,
        prompt,
        8,
        False,
        llm.model.eos(),
        2,
        [],
        None,
        0,
    )
    assert result == []
    assert llm.ctx.kv_cache_seq_pos_max(0) + 1 == len(prompt)
    assert llm.perf()["n_mtp_prefill"] == len(prompt)


def run_native(llm, prompt, budget, width=2, skip=0, grammar=None, eos=None):
    return _llama.generate_tokens_speculative_mtp(
        llm.ctx,
        llm._build_sampler(SamplingParams(temperature=0.0, seed=0)),
        grammar,
        prompt,
        budget,
        False,
        llm.model.eos() if eos is None else eos,
        width,
        [],
        None,
        skip,
    )


@requires_mtp_model
@pytest.mark.parametrize("width", [1, 2, 4])
@pytest.mark.parametrize("text", [" Paris", " one two three four five six seven"])
def test_verified_eog_retains_returned_prefix(cache_llm, width, text):
    llm = cache_llm
    prompt = llm.tokenize("The capital of France is", add_special=False)
    grammar = _llama.GrammarSampler(llm.model, f'root ::= "{text}"', "root")
    llm.perf_reset()
    result = run_native(llm, prompt, 32, width, grammar=grammar)
    assert llm.detokenize(result) == text
    assert llm.perf()["n_verify"] > 0
    prefix = prompt + result
    assert llm.ctx.kv_cache_seq_pos_max(0) + 1 == len(prefix)
    # Full-prefix continuation must sample from the retained token's logits.
    llm.perf_reset()
    continued = run_native(llm, prefix, 8, width, len(prefix))
    assert llm.perf()["n_mtp_prefill"] == 0
    # Replay the same constrained prefix, finishing at the token budget rather
    # than EOG. This leaves logits for the final returned token by construction.
    llm.kv_cache_clear()
    grammar = _llama.GrammarSampler(llm.model, f'root ::= "{text}"', "root")
    replay = run_native(llm, prompt, len(result), width, grammar=grammar)
    assert replay == result
    expected = run_native(llm, prefix, 8, width, len(prefix))
    assert continued == expected


@requires_mtp_model
def test_noop_tail_trim_retains_driver(cache_llm):
    llm = cache_llm
    prompt = llm.tokenize("Count from one to ten:", add_special=False)
    prefix = prompt + run_native(llm, prompt, 8)
    assert llm.ctx.kv_cache_seq_rm(0, len(prefix), -1)
    llm.perf_reset()
    result = run_native(llm, prefix, 1, skip=len(prefix))
    assert result
    assert llm.perf()["n_mtp_prefill"] == 0


@requires_mtp_model
@requires_cache_fixture
@pytest.mark.parametrize("removed", [1, 3, 5])
def test_prefix_edit_reuses_only_supported_history(cache_llm, removed):
    llm = cache_llm
    # Keep the original literal-marker fixture: its final verify batch makes
    # one/three-token rollback supported and a five-token rollback unsupported.
    # The public chat tests below exercise correctly parsed template markers.
    prompt = llm.tokenize(
        llm._format_chat_messages(
            [{"role": "user", "content": "What is 2 + 2? Reply with one word."}]
        ),
        add_special=True,
    )

    def generate(tokens, budget, reset):
        return llm._generate_from_tokens(
            tokens,
            max_tokens=budget,
            sampler=llm._build_sampler(SamplingParams(temperature=0.0, seed=0)),
            add_bos=False,
            reset_kv_cache=reset,
            cache_prompt=True,
            speculative=True,
            n_draft_max=4,
        )

    result = generate(prompt, 96, True)
    assert len(result) == 5  # This fixture reaches genuine EOG.
    prefix = prompt + result
    assert llm._cached_prompt_tokens == prefix
    llm.perf_reset()
    continued = generate(prefix[:-removed], 8, False)
    assert continued == prefix[-removed:]
    expected_prefill = len(prefix) - removed if removed == 5 else 0
    assert llm.perf()["n_mtp_prefill"] == expected_prefill
    assert llm.perf()["n_verify"] > 0
    # The restored hidden row must seed useful drafts, beyond correct logits
    # for the first regenerated token. EOG itself is excluded from acceptance.
    assert llm.perf()["n_accepted"] == removed - 1
    assert llm._cached_prompt_tokens == prefix[:-removed] + continued
    assert llm.ctx.kv_cache_seq_pos_max(0) + 1 == len(llm._cached_prompt_tokens)


@requires_mtp_model
@requires_cache_fixture
def test_public_logits_describe_retained_eog_prefix(cache_llm):
    llm = cache_llm
    prompt = llm.tokenize(
        llm._format_chat_messages(
            [{"role": "user", "content": "What is 2 + 2? Reply with one word."}]
        ),
        add_special=True,
    )
    result = run_native(llm, prompt, 96, width=4)
    assert len(result) == 5
    logits = llm.ctx.logits()
    sampler = llm._build_sampler(SamplingParams(temperature=0.0, repeat_penalty=1.0))
    # A fresh sampler without penalties is an independent last-row oracle.
    assert max(range(len(logits)), key=logits.__getitem__) == sampler.sample(llm.ctx)


@requires_mtp_model
def test_chat_preparation_parses_template_markers(cache_llm):
    llm = cache_llm
    messages = [{"role": "user", "content": "What is 2 + 2?"}]
    text, tokens, count = llm._prepare_chat(messages, add_bos=False)
    expected = llm.tokenize(text, add_special=False, parse_special=True)
    assert tokens == expected
    assert count == len(expected)


@requires_mtp_model
@pytest.mark.parametrize("stream", [False, True])
def test_chat_generation_uses_template_tokens(cache_llm, monkeypatch, stream):
    llm = cache_llm
    messages = [{"role": "user", "content": "What is 2 + 2?"}]
    expected = llm.tokenize(
        llm._format_chat_messages(messages),
        add_special=False,
        parse_special=True,
    )
    seen = []
    token = llm.tokenize("4", add_special=False)[0]

    def fake_generate(
        ctx, sampler, grammar, prompt, budget, bos, eos, width, stops, callback, skip
    ):
        seen.extend(prompt)
        if callback is not None:
            callback(token)
        return [token]

    monkeypatch.setattr(_llama, "generate_tokens_speculative_mtp", fake_generate)
    response = llm.create_chat_completion(
        messages, stream=stream, max_tokens=1, temperature=0.0, speculative=True
    )
    if stream:
        list(response)
    assert seen == expected


@requires_mtp_model
@requires_cache_fixture
@pytest.mark.parametrize("stream", [False, True])
def test_completed_chat_reuses_only_new_turn_tokens(cache_llm, stream):
    llm = cache_llm
    messages = []
    for index, number in enumerate([2, 3, 4]):
        messages.append(
            {
                "role": "user",
                "content": f"What is {number} + {number}? Reply only with the digit.",
            }
        )
        _, prompt, _ = llm._prepare_chat(messages)
        cached = list(llm._cached_prompt_tokens)
        assert prompt[: len(cached)] == cached
        llm.perf_reset()
        completion = llm.create_chat_completion(
            messages,
            max_tokens=32,
            stream=stream,
            reset_kv_cache=index == 0,
            temperature=0.0,
            seed=0,
            speculative=True,
            n_draft_max=2,
        )
        if stream:
            response = "".join(
                chunk["choices"][0]["delta"].get("content", "") for chunk in completion
            )
        else:
            response = completion["choices"][0]["message"]["content"]
        assert response.rstrip().endswith(str(number * 2))
        assert 0 < len(llm._cached_prompt_tokens) - len(prompt) < 32
        assert llm._cached_prompt_tokens[: len(prompt)] == prompt
        assert llm.perf()["n_mtp_prefill"] == len(prompt) - len(cached)
        assert llm.ctx.kv_cache_seq_pos_max(0) + 1 == len(llm._cached_prompt_tokens)
        messages.append({"role": "assistant", "content": response})
