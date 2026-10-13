"""Snapshot validation and filesystem failure checks through the store interface."""

import json
import os
import stat
from dataclasses import asdict, replace

import pytest

from llama_cpp import _chat_store
from llama_cpp._chat_store import ChatSettings, ChatStore, SavedChat


@pytest.fixture
def snapshot(tmp_path):
    return SavedChat(
        ChatSettings(model=str(tmp_path / "local.gguf"), system="Be concise."),
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "你好\nExplain 🦙."},
            {"role": "assistant", "content": "<think>\n\n</think>\nHello."},
        ],
    )


def test_round_trip_and_private_snapshot_permissions(snapshot, tmp_path):
    store = ChatStore(tmp_path / "state")
    store.save(snapshot)
    assert store.load() == snapshot
    if os.name == "posix":
        assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
        assert stat.S_IMODE(store.path.parent.stat().st_mode) == 0o700


@pytest.mark.parametrize("failure", ["fsync", "replace"])
def test_failed_atomic_write_preserves_last_snapshot_and_removes_temporary_file(
    snapshot, tmp_path, monkeypatch, failure
):
    store = ChatStore(tmp_path / "state")
    store.save(snapshot)
    previous = store.path.read_bytes()

    def fail(*_args):
        raise OSError("disk full")

    monkeypatch.setattr(_chat_store.os, failure, fail)
    with pytest.raises(OSError, match="disk full"):
        store.save(replace(snapshot, messages=[]))
    assert store.path.read_bytes() == previous
    assert list(store.path.parent.iterdir()) == [store.path]
    assert store.load() == snapshot


@pytest.mark.parametrize(
    "field, value",
    [
        ("model", None),
        ("system", 1),
        ("ctx", 0),
        ("ctx", True),
        ("max_tokens", -1),
        ("threads", "4"),
        ("seed", 0.5),
        ("gpu_layers", -2),
        ("draft_width", 9),
        ("mtp", "unknown"),
        ("temperature", "0.7"),
        ("temperature", True),
        ("temperature", float("nan")),
        ("temperature", float("inf")),
        ("temperature", -0.1),
        ("temperature", 10**400),
    ],
)
def test_invalid_saved_settings_are_rejected_before_use(
    snapshot, tmp_path, field, value
):
    store = ChatStore(tmp_path)
    settings = asdict(snapshot.settings)
    settings[field] = value
    store.path.write_text(
        json.dumps({"version": 1, "settings": settings, "messages": snapshot.messages})
    )
    with pytest.raises(ValueError):
        store.load()


@pytest.mark.parametrize(
    "messages",
    [
        None,
        [{"role": "system", "content": "Wrong."}],
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "partial"},
        ],
        [
            {"role": "system", "content": "Be concise."},
            {"role": "assistant", "content": "out of order"},
            {"role": "user", "content": "question"},
        ],
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": 1},
            {"role": "assistant", "content": "answer"},
        ],
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": " "},
            {"role": "assistant", "content": "answer"},
        ],
    ],
)
def test_invalid_or_incomplete_history_is_rejected(snapshot, tmp_path, messages):
    store = ChatStore(tmp_path)
    store.path.write_text(
        json.dumps(
            {"version": 1, "settings": asdict(snapshot.settings), "messages": messages}
        )
    )
    with pytest.raises(ValueError):
        store.load()


def test_default_store_uses_xdg_state_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    assert ChatStore.default().path == tmp_path / "llama-chat" / "last.json"


def test_relative_xdg_state_home_uses_home_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", "relative/path")
    monkeypatch.setattr(_chat_store.Path, "home", lambda: tmp_path)
    assert ChatStore.default().path == (
        tmp_path / ".local" / "state" / "llama-chat" / "last.json"
    )
