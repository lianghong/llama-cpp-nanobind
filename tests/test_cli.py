"""Terminal app contracts: history, streaming cleanup, input modes and errors."""

import io
import subprocess
import sys
from copy import deepcopy

import pytest

from llama_cpp import cli
from llama_cpp import LlamaError
from llama_cpp._chat_store import ChatStore


class ChatModel:
    def __init__(self):
        self.is_stuck = False
        self.supports_mtp = True
        self.probes = 0
        self.calls = []
        self.events = []
        self.error = None
        self.closed = False
        self.config = None
        self.sampling = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.closed = True

    def supports_speculative_mtp(self):
        self.probes += 1
        return self.supports_mtp

    def perf_reset(self):
        pass

    def perf(self):
        return {"n_drafted": 4, "n_accepted": 3, "n_mtp_prefill": 24}

    def kv_cache_clear(self):
        self.events.append("clear")

    def create_chat_completion(self, messages, **kwargs):
        self.calls.append((deepcopy(messages), kwargs))

        def chunks():
            try:
                yield {"choices": [{"delta": {"content": "answer"}}]}
                if self.error is not None:
                    error = self.error
                    self.error = None
                    raise error
                yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}
            finally:
                self.events.append("stream closed")

        return chunks()


@pytest.fixture
def model(monkeypatch, tmp_path):
    path = tmp_path / "local.gguf"
    path.touch()
    instance = ChatModel()

    def factory(model_path, *, config, sampling):
        assert model_path == str(path)
        instance.config = config
        instance.sampling = sampling
        return instance

    monkeypatch.setattr(cli, "Llama", factory)
    monkeypatch.delenv("LLAMA_MODEL", raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    return path, instance


def session(model):
    return cli._ChatSession(
        model,
        system_prompt="Be concise.",
        max_tokens=32,
        speculative=True,
        draft_width=2,
    )


def test_completed_turns_use_full_history_and_cache(model):
    _, instance = model
    chat = session(instance)
    assert "".join(chat.reply("first")) == "answer"
    assert "".join(chat.reply("second")) == "answer"
    messages, options = instance.calls[1]
    assert messages == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "second"},
    ]
    assert options["reset_kv_cache"] is False
    assert options["cache_prompt"] is True
    assert options["speculative"] is True


def test_closing_partial_reply_discards_turn_and_joins_stream(model):
    _, instance = model
    chat = session(instance)
    "".join(chat.reply("first"))
    reply = chat.reply("discard me")
    assert next(reply) == "answer"
    reply.close()
    assert instance.events[-2:] == ["stream closed", "clear"]
    "".join(chat.reply("next"))
    assert [m["content"] for m in instance.calls[-1][0]] == [
        "Be concise.",
        "first",
        "answer",
        "next",
    ]


def test_clear_keeps_system_instruction(model):
    _, instance = model
    chat = session(instance)
    "".join(chat.reply("old"))
    chat.clear()
    "".join(chat.reply("new"))
    assert instance.calls[-1][0] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "new"},
    ]


@pytest.mark.parametrize("error", [KeyboardInterrupt(), ValueError("failed")])
def test_interactive_recovers_without_committing_failed_reply(
    model, monkeypatch, capsys, error
):
    path, instance = model
    instance.error = error
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO("discard\nnext\n/exit\n"))
    assert cli.main(["-m", str(path), "-i"]) == 0
    assert instance.calls[1][0] == [{"role": "user", "content": "next"}]
    assert instance.events[:2] == ["stream closed", "clear"]
    assert instance.closed
    output = capsys.readouterr()
    assert (
        "interrupted" in output.err
        if isinstance(error, KeyboardInterrupt)
        else ("failed" in output.err)
    )


def test_stuck_worker_never_clears_or_reuses_context(model, monkeypatch, capsys):
    path, instance = model
    instance.is_stuck = True
    instance.error = LlamaError("worker did not stop")
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO("hello\nnever run\n"))
    assert cli.main(["-m", str(path), "-i"]) == 1
    assert len(instance.calls) == 1
    assert "clear" not in instance.events
    assert instance.closed
    assert "worker did not stop" in capsys.readouterr().err


def test_interactive_commands_and_exit(model, monkeypatch, capsys):
    path, instance = model
    monkeypatch.setattr(
        cli.sys, "stdin", io.StringIO("/help\nfirst\n/clear\nsecond\n/quit\n")
    )
    assert cli.main(["-m", str(path), "-i", "--system", "Be concise."]) == 0
    assert instance.calls[1][0] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "second"},
    ]
    assert instance.closed
    output = capsys.readouterr()
    assert "Conversation cleared." in output.err
    assert "Commands:" in output.err


@pytest.mark.parametrize("mtp", ["auto", "on", "off"])
def test_single_prompt_stdout_is_only_reply(model, capsys, mtp):
    path, instance = model
    assert cli.main(["-m", str(path), "-p", "hello", "--mtp", mtp]) == 0
    assert capsys.readouterr().out == "answer\n"
    assert instance.calls[0][1]["speculative"] is (mtp != "off")
    assert instance.config.load_mtp is (mtp != "off")
    assert instance.probes == (0 if mtp == "off" else 1)
    assert instance.closed


def test_auto_uses_ordinary_model_and_forced_mtp_errors(model, capsys):
    path, instance = model
    instance.supports_mtp = False
    assert cli.main(["-m", str(path), "-p", "hello"]) == 0
    assert instance.calls[0][1]["speculative"] is False
    assert cli.main(["-m", str(path), "-p", "hello", "--mtp", "on"]) == 1
    assert len(instance.calls) == 1
    assert "cannot use MTP" in capsys.readouterr().err


def test_piped_multiline_prompt_and_environment_model(model, monkeypatch, capsys):
    path, instance = model
    monkeypatch.setenv("LLAMA_MODEL", str(path))
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO("first line\nsecond line\n"))
    assert cli.main([]) == 0
    assert instance.calls[0][0] == [
        {"role": "user", "content": "first line\nsecond line\n"}
    ]
    assert capsys.readouterr().out == "answer\n"


def test_stats_use_stderr_and_draft_width_configures_rollback(model, capsys):
    path, instance = model
    assert (
        cli.main(
            [
                "-m",
                str(path),
                "-p",
                "hello",
                "--stats",
                "--draft-width",
                "4",
                "--ctx",
                "512",
                "--threads",
                "2",
                "--temperature",
                "0",
                "--seed",
                "0",
            ]
        )
        == 0
    )
    assert instance.config.n_rs_seq == 4
    assert instance.config.n_ctx == 512
    assert instance.config.n_threads == instance.config.n_threads_batch == 2
    assert instance.sampling.temperature == 0
    assert instance.sampling.seed == 0
    output = capsys.readouterr()
    assert output.out == "answer\n"
    assert "prefill=24" in output.err
    assert "accepted=3/4" in output.err


@pytest.mark.parametrize(
    "options",
    [
        ["--ctx", "0"],
        ["--threads", "0"],
        ["--max-tokens", "0"],
        ["--temperature", "nan"],
        ["--temperature", "inf"],
        ["--draft-width", "9"],
        ["--gpu-layers", "-2"],
        ["--prompt", ""],
    ],
)
def test_invalid_options_fail_before_model_load(model, capsys, options):
    path, instance = model
    with pytest.raises(SystemExit) as error:
        cli.main(["-m", str(path), *options])
    assert error.value.code == 2
    assert instance.config is None
    assert capsys.readouterr().err


def test_missing_model_has_clear_error(model, capsys):
    path, instance = model
    with pytest.raises(SystemExit) as error:
        cli.main(["-m", str(path.with_name("missing.gguf")), "-p", "hello"])
    assert error.value.code == 2
    assert "model file not found" in capsys.readouterr().err
    assert instance.config is None


def test_one_shot_inference_error_closes_model(model, capsys):
    path, instance = model
    instance.error = RuntimeError("generation failed")
    assert cli.main(["-m", str(path), "-p", "hello"]) == 1
    assert instance.closed
    assert instance.events == ["stream closed", "clear"]
    assert "generation failed" in capsys.readouterr().err


def test_module_entry_point_help_without_model():
    result = subprocess.run(
        [sys.executable, "-m", "llama_cpp", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Chat with a local GGUF model" in result.stdout
    assert "--model" in result.stdout
    assert "--mtp" in result.stdout


def test_autosave_and_resume_restore_model_settings_and_history(
    model, monkeypatch, capsys
):
    path, instance = model
    assert (
        cli.main(
            [
                "-m",
                str(path),
                "-p",
                "first",
                "--system",
                "Be concise.",
                "--ctx",
                "512",
                "--max-tokens",
                "32",
                "--temperature",
                "0",
                "--seed",
                "0",
                "--threads",
                "2",
                "--gpu-layers",
                "0",
                "--mtp",
                "on",
                "--draft-width",
                "4",
            ]
        )
        == 0
    )
    store = ChatStore.default()
    saved = store.load()
    assert saved.settings.model == str(path.resolve())
    assert len(saved.messages) == 3
    # A saved model wins over the fallback environment; no model flag is needed.
    monkeypatch.setenv("LLAMA_MODEL", str(path.with_name("missing.gguf")))
    assert cli.main(["--resume", "last", "-p", "second"]) == 0
    assert instance.config.n_ctx == 512
    assert instance.config.n_threads == instance.config.n_threads_batch == 2
    assert instance.config.n_gpu_layers == 0
    assert instance.config.n_rs_seq == 4
    assert instance.sampling.temperature == instance.sampling.seed == 0
    assert instance.calls[-1][1]["max_tokens"] == 32
    assert [message["content"] for message in instance.calls[-1][0]] == [
        "Be concise.",
        "first",
        "answer",
        "second",
    ]
    assert len(store.load().messages) == 5
    assert capsys.readouterr().out == "answer\nanswer\n"


def test_resume_explicit_settings_override_saved_values(model, capsys):
    path, instance = model
    assert (
        cli.main(
            ["-m", str(path), "-p", "first", "--system", "Old.", "--temperature", "1"]
        )
        == 0
    )
    assert (
        cli.main(
            [
                "--resume",
                "last",
                "-p",
                "second",
                "--system",
                "",
                "--ctx",
                "1024",
                "--max-tokens",
                "12",
                "--threads",
                "2",
                "--temperature",
                "0",
                "--seed",
                "0",
                "--gpu-layers",
                "0",
                "--mtp",
                "off",
                "--draft-width",
                "1",
            ]
        )
        == 0
    )
    assert instance.config.n_ctx == 1024
    assert instance.config.n_threads == 2
    assert instance.config.n_gpu_layers == 0
    assert instance.config.load_mtp is False
    assert instance.sampling.temperature == instance.sampling.seed == 0
    messages, options = instance.calls[-1]
    assert messages[0] == {"role": "user", "content": "first"}
    assert options["max_tokens"] == 12
    assert options["n_draft_max"] == 1
    assert options["speculative"] is False
    assert ChatStore.default().load().settings.system == ""
    capsys.readouterr()


def test_resume_allows_model_path_override_after_model_moves(
    model, monkeypatch, capsys
):
    path, instance = model
    assert cli.main(["-m", str(path), "-p", "first"]) == 0
    moved = path.with_name("moved.gguf")
    path.rename(moved)

    def factory(model_path, *, config, sampling):
        assert model_path == str(moved)
        instance.config = config
        instance.sampling = sampling
        return instance

    monkeypatch.setattr(cli, "Llama", factory)
    assert cli.main(["--resume", "last", "-m", str(moved), "-p", "second"]) == 0
    assert instance.calls[-1][0][0]["content"] == "first"
    assert ChatStore.default().load().settings.model == str(moved)
    capsys.readouterr()


def test_resume_missing_saved_model_fails_without_changing_snapshot(model, capsys):
    path, instance = model
    assert cli.main(["-m", str(path), "-p", "first"]) == 0
    previous = ChatStore.default().path.read_bytes()
    path.unlink()
    with pytest.raises(SystemExit) as error:
        cli.main(["--resume", "last", "-p", "second"])
    assert error.value.code == 2
    assert len(instance.calls) == 1
    assert ChatStore.default().path.read_bytes() == previous
    assert "model file not found" in capsys.readouterr().err


@pytest.mark.parametrize("new_options", [[], ["--new"]])
def test_fresh_chat_replaces_previous_history_only_after_completed_reply(
    model, monkeypatch, capsys, new_options
):
    path, instance = model
    assert cli.main(["-m", str(path), "-p", "previous"]) == 0
    store = ChatStore.default()
    previous = store.path.read_bytes()
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO("/exit\n"))
    assert cli.main(["-m", str(path), "-i", *new_options]) == 0
    assert store.path.read_bytes() == previous
    assert cli.main(["-m", str(path), "-p", "fresh", *new_options]) == 0
    assert instance.calls[-1][0] == [{"role": "user", "content": "fresh"}]
    assert [m["content"] for m in store.load().messages] == ["fresh", "answer"]
    capsys.readouterr()


@pytest.mark.parametrize("contents", [None, "{broken", '{"version": 99}'])
def test_resume_missing_or_invalid_save_fails_before_model_load(
    model, capsys, contents
):
    _, instance = model
    store = ChatStore.default()
    if contents is not None:
        store.path.parent.mkdir(parents=True)
        store.path.write_text(contents)
    with pytest.raises(SystemExit) as error:
        cli.main(["--resume", "last", "-p", "hello"])
    assert error.value.code == 2
    assert instance.config is None
    assert "cannot resume" in capsys.readouterr().err
    if contents is not None:
        assert store.path.read_text() == contents


def test_new_chat_can_replace_corrupt_snapshot(model, capsys):
    path, _ = model
    store = ChatStore.default()
    store.path.parent.mkdir(parents=True)
    store.path.write_text("{broken")
    assert cli.main(["--new", "-m", str(path), "-p", "fresh"]) == 0
    assert store.load().messages[0]["content"] == "fresh"
    capsys.readouterr()


def test_new_and_resume_are_mutually_exclusive(model, capsys):
    _, instance = model
    with pytest.raises(SystemExit) as error:
        cli.main(["--new", "--resume", "last"])
    assert error.value.code == 2
    assert instance.config is None
    assert "not allowed" in capsys.readouterr().err


@pytest.mark.parametrize("error", [RuntimeError("failed"), KeyboardInterrupt()])
def test_failed_resumed_reply_preserves_saved_conversation(model, capsys, error):
    path, instance = model
    assert cli.main(["-m", str(path), "-p", "completed"]) == 0
    store = ChatStore.default()
    before = store.path.read_bytes()
    instance.error = error
    assert cli.main(["--resume", "last", "-p", "discard"]) == (
        130 if isinstance(error, KeyboardInterrupt) else 1
    )
    assert store.path.read_bytes() == before
    assert instance.events[-2:] == ["stream closed", "clear"]
    capsys.readouterr()


def test_clear_saves_empty_history_with_system_instruction(model, monkeypatch, capsys):
    path, instance = model
    assert cli.main(["-m", str(path), "-p", "old", "--system", "Be concise."]) == 0
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO("/clear\n/exit\n"))
    assert cli.main(["--resume", "last", "-i"]) == 0
    assert ChatStore.default().load().messages == [
        {"role": "system", "content": "Be concise."}
    ]
    assert cli.main(["--resume", "last", "-p", "new"]) == 0
    assert instance.calls[-1][0] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "new"},
    ]
    capsys.readouterr()


def test_save_error_reports_failure_and_keeps_completed_in_memory_history(
    model, monkeypatch, capsys
):
    path, instance = model
    assert cli.main(["-m", str(path), "-p", "previous"]) == 0
    store = ChatStore.default()
    previous = store.path.read_bytes()

    def fail_save(*_args):
        raise OSError("disk full")

    monkeypatch.setattr(ChatStore, "save", fail_save)
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO("first\nsecond\n/exit\n"))
    assert cli.main(["--resume", "last", "-i"]) == 0
    assert [m["content"] for m in instance.calls[-1][0]] == [
        "previous",
        "answer",
        "first",
        "answer",
        "second",
    ]
    assert store.path.read_bytes() == previous
    assert "Could not save conversation" in capsys.readouterr().err
