"""Model-based checks for terminal cache, interruption and process restarts."""

import os
import re
import subprocess
import sys

import pytest
from conftest import MTP_MODEL_PATH, requires_mtp_model

from llama_cpp import Llama, LlamaConfig, SamplingParams, cli
from llama_cpp._chat_store import ChatStore


@pytest.fixture
def chat_model():
    with Llama(
        MTP_MODEL_PATH,
        config=LlamaConfig(
            model_path=MTP_MODEL_PATH,
            n_ctx=512,
            n_batch=32,
            n_ubatch=16,
            n_rs_seq=2,
            n_threads=4,
            n_threads_batch=4,
            n_gpu_layers=-1,
            verbose=False,
        ),
        sampling=SamplingParams(temperature=0.0, seed=0),
    ) as llm:
        assert llm.supports_speculative_mtp()
        yield llm


def chat_session(llm):
    return cli._ChatSession(
        llm,
        system_prompt=None,
        max_tokens=256,
        speculative=True,
        draft_width=2,
    )


@requires_mtp_model
def test_cli_completed_chat_retains_cache(chat_model):
    llm = chat_model
    session = chat_session(llm)
    previous_cache = []
    for number in [2, 3, 4]:
        response = "".join(
            session.reply(f"What is {number} + {number}? Reply only with the digit.")
        )
        assert response.strip()
        cache = llm._cached_prompt_tokens
        assert cache
        assert llm.ctx.kv_cache_seq_pos_max(0) + 1 == len(cache)
        assert llm.perf()["n_mtp_prefill"] > 0
        if previous_cache:
            assert cache[: len(previous_cache)] == previous_cache
            assert llm.perf()["n_mtp_prefill"] < len(previous_cache)
        previous_cache = list(cache)


@requires_mtp_model
def test_cli_clear_restarts_native_conversation(chat_model):
    llm = chat_model
    session = chat_session(llm)
    prompt = "What is 2 + 2? Reply only with the digit."
    original = "".join(session.reply(prompt))
    original_prefill = llm.perf()["n_mtp_prefill"]
    session.clear()
    assert llm._cached_prompt_tokens == []
    assert llm.ctx.kv_cache_seq_pos_max(0) == -1
    assert "".join(session.reply(prompt)) == original
    assert llm.perf()["n_mtp_prefill"] == original_prefill


@requires_mtp_model
def test_cli_partial_stream_closes_worker_before_reuse(chat_model):
    llm = chat_model
    session = chat_session(llm)
    prompt = "What is 2 + 2? Reply only with the digit."
    original = "".join(session.reply(prompt))
    reply = session.reply("List the integers from one to one thousand.")
    assert next(reply)
    reply.close()
    assert not llm.is_stuck
    assert llm._cached_prompt_tokens == []
    assert llm.ctx.kv_cache_seq_pos_max(0) == -1
    continued = "".join(session.reply("What is 3 + 3? Reply only with the digit."))
    assert continued.strip()
    # Replaying completed history in a fresh context is the independent oracle.
    with Llama(
        MTP_MODEL_PATH,
        config=llm.config,
        sampling=llm.sampling,
    ) as fresh:
        replay = chat_session(fresh)
        assert "".join(replay.reply(prompt)) == original
        expected = "".join(replay.reply("What is 3 + 3? Reply only with the digit."))
    assert continued == expected


@requires_mtp_model
@pytest.mark.parametrize("mtp", ["on", "off"])
def test_cli_resumed_process_matches_uninterrupted_chat(tmp_path, mtp):
    """Compare fresh-process replay against a model with its live cache intact."""
    prompts = [
        f"What is {number} + {number}? Reply only with the digit."
        for number in (2, 3, 4)
    ]
    settings = [
        "--model",
        MTP_MODEL_PATH,
        "--ctx",
        "512",
        "--max-tokens",
        "256",
        "--temperature",
        "0",
        "--seed",
        "0",
        "--mtp",
        mtp,
    ]
    command = [sys.executable, "-m", "llama_cpp"]

    def run(directory, arguments, transcript=None):
        environment = dict(os.environ, XDG_STATE_HOME=str(directory))
        environment.pop("LLAMA_MODEL", None)
        return subprocess.run(
            [*command, *arguments],
            input=transcript,
            text=True,
            capture_output=True,
            env=environment,
            check=True,
            timeout=120,
        )

    baseline = tmp_path / "baseline"
    run(baseline, [*settings, "-i"], "\n".join([*prompts, "/exit", ""]))
    resumed = tmp_path / "resumed"
    run(resumed, [*settings, "-p", prompts[0]])
    result = run(
        resumed,
        ["--resume", "last", "-i", "--stats"],
        "\n".join([*prompts[1:], "/exit", ""]),
    )
    uninterrupted = ChatStore(baseline / "llama-chat").load()
    restored = ChatStore(resumed / "llama-chat").load()
    assert restored == uninterrupted
    assert len(restored.messages) == 6
    if mtp == "on":
        prefills = [int(value) for value in re.findall(r"prefill=(\d+)", result.stderr)]
        assert len(prefills) == 2
        # The resumed turn rebuilds from history; its following turn reuses KV.
        assert prefills[0] > prefills[1] > 0
