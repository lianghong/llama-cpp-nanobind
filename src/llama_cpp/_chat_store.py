"""Versioned chat snapshots; native state is rebuilt from completed history."""

import contextlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ChatSettings:
    model: str
    system: str | None = None
    ctx: int = 4096
    max_tokens: int = 256
    temperature: float = 0.7
    seed: int = -1
    threads: int = 4
    gpu_layers: int = -1
    mtp: str = "auto"
    draft_width: int = 2


@dataclass(frozen=True)
class SavedChat:
    settings: ChatSettings
    messages: list[dict[str, str]]


def _read_settings(data: Any) -> ChatSettings:
    if not isinstance(data, dict) or set(data) != {
        field.name for field in fields(ChatSettings)
    }:
        raise ValueError("saved model settings are incomplete or unrecognized")
    if not isinstance(data["model"], str) or not data["model"]:
        raise ValueError("saved model path must be a non-empty string")
    if data["system"] is not None and not isinstance(data["system"], str):
        raise ValueError("saved system instruction must be a string")
    for name in ("ctx", "max_tokens", "seed", "threads", "gpu_layers", "draft_width"):
        if type(data[name]) is not int:
            raise ValueError(f"saved {name} must be an integer")
    if any(data[name] < 1 for name in ("ctx", "max_tokens", "threads")):
        raise ValueError("saved context, reply limit and threads must be positive")
    if data["gpu_layers"] < -1 or not 1 <= data["draft_width"] <= 8:
        raise ValueError("saved GPU layers or draft width are out of range")
    temperature = data["temperature"]
    if type(temperature) not in (int, float):
        raise ValueError("saved temperature must be finite and non-negative")
    try:
        valid_temperature = math.isfinite(temperature) and temperature >= 0
    except OverflowError:
        valid_temperature = False
    if not valid_temperature:
        raise ValueError("saved temperature must be finite and non-negative")
    if data["mtp"] not in ("auto", "on", "off"):
        raise ValueError("saved MTP mode is unrecognized")
    return ChatSettings(**data)


def _read_messages(data: Any, settings: ChatSettings) -> list[dict[str, str]]:
    if not isinstance(data, list):
        raise ValueError("saved history must be a list")
    messages: list[dict[str, str]] = []
    for item in data:
        if (
            not isinstance(item, dict)
            or set(item) != {"role", "content"}
            or not isinstance(item["role"], str)
            or not isinstance(item["content"], str)
        ):
            raise ValueError("saved messages must contain string role and content")
        messages.append({"role": item["role"], "content": item["content"]})
    initial = (
        [{"role": "system", "content": settings.system}] if settings.system else []
    )
    if messages[: len(initial)] != initial:
        raise ValueError("saved system instruction does not match the settings")
    turns = messages[len(initial) :]
    if len(turns) % 2:
        raise ValueError("saved history contains an incomplete turn")
    for index, message in enumerate(turns):
        role = "user" if index % 2 == 0 else "assistant"
        if message["role"] != role:
            raise ValueError("saved history must alternate user and assistant turns")
        if role == "user" and not message["content"].strip():
            raise ValueError("saved user messages must not be empty")
    return messages


class ChatStore:
    """Load the last snapshot and replace it only after a complete file write."""

    def __init__(self, directory: Path) -> None:
        self.path = directory / "last.json"

    @classmethod
    def default(cls) -> ChatStore:
        directory = Path(os.environ.get("XDG_STATE_HOME", ""))
        if not directory.is_absolute():
            directory = Path.home() / ".local" / "state"
        return cls(directory / "llama-chat")

    def load(self) -> SavedChat:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or set(data) != {
            "version",
            "settings",
            "messages",
        }:
            raise ValueError("saved conversation format is unrecognized")
        if type(data["version"]) is not int or data["version"] != 1:
            raise ValueError("saved conversation version is unsupported")
        settings = _read_settings(data["settings"])
        return SavedChat(settings, _read_messages(data["messages"], settings))

    def save(self, chat: SavedChat) -> None:
        payload = json.dumps(
            {
                "version": 1,
                "settings": asdict(chat.settings),
                "messages": chat.messages,
            },
            indent=2,
            allow_nan=False,
        )
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            # NamedTemporaryFile is private (0600) and on the destination
            # filesystem, so replacement cannot expose a partial JSON file.
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=".last-",
                suffix=".tmp",
                delete=False,
            ) as output:
                temporary = Path(output.name)
                output.write(payload + "\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                with contextlib.suppress(OSError):
                    temporary.unlink()
