"""A small terminal chat app using a single reusable Llama context."""

import argparse
import contextlib
import math
import os
import sys
import time
from collections.abc import Callable, Generator, Iterator, Sequence
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, TextIO

from ._about import __version__
from ._chat_store import ChatSettings, ChatStore, SavedChat
from .llama import Llama, LlamaConfig, LlamaError, SamplingParams, ValidationError


_HELP = """Commands:
  /clear       Start a new conversation.
  /help        Show these commands.
  /exit        Exit (also /quit or Ctrl+D).
Ctrl+C interrupts a reply; at the input prompt it exits.
"""


def _close_iterator(iterator: Iterator[Any]) -> None:
    close = getattr(iterator, "close", None)
    if callable(close):
        close()


class _ChatSession:
    """Commit completed turns and reset uncertain native state on interruption."""

    def __init__(
        self,
        llm: Llama,
        *,
        system_prompt: str | None,
        max_tokens: int,
        speculative: bool,
        draft_width: int,
        history: Sequence[dict[str, str]] = (),
        on_commit: Callable[[list[dict[str, str]]], None] | None = None,
    ) -> None:
        self.llm = llm
        self.max_tokens = max_tokens
        self.speculative = speculative
        self.draft_width = draft_width
        self._initial_messages = (
            [{"role": "system", "content": system_prompt}] if system_prompt else []
        )
        self._messages = list(self._initial_messages) + [
            dict(message) for message in history if message["role"] != "system"
        ]
        self._on_commit = on_commit

    def _save(self) -> None:
        if self._on_commit is not None:
            self._on_commit([dict(message) for message in self._messages])

    def clear(self) -> None:
        self.llm.kv_cache_clear()
        self._messages = list(self._initial_messages)
        self._save()

    def reply(self, prompt: str) -> Generator[str]:
        messages = [dict(message) for message in self._messages]
        messages.append({"role": "user", "content": prompt})
        pieces: list[str] = []
        completed = False
        chunks: Iterator[dict[str, Any]] | None = None
        try:
            self.llm.perf_reset()
            chunks = self.llm.create_chat_completion(
                messages,
                max_tokens=self.max_tokens,
                stream=True,
                reset_kv_cache=False,
                cache_prompt=True,
                speculative=self.speculative,
                n_draft_max=self.draft_width,
            )
            for chunk in chunks:
                text = chunk["choices"][0]["delta"].get("content", "")
                if text:
                    pieces.append(text)
                    yield text
            self._messages = messages + [
                {"role": "assistant", "content": "".join(pieces)}
            ]
            completed = True
            self._save()
        finally:
            try:
                if chunks is not None:
                    _close_iterator(chunks)
            finally:
                # A partial visible reply is never added to the conversation.
                # Closing the owned stream first joins its generation worker.
                if not completed and not self.llm.is_stuck:
                    self.llm.kv_cache_clear()


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _temperature(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return number


def _model_path(value: str) -> Path:
    try:
        path = Path(value).expanduser()
        exists = path.is_file()
    except (OSError, RuntimeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    if not exists:
        raise argparse.ArgumentTypeError(f"model file not found: {path}")
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llama-chat",
        description="Chat with a local GGUF model in your terminal.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "-m",
        "--model",
        type=_model_path,
        help="Local GGUF path (or set LLAMA_MODEL).",
    )
    conversation = parser.add_mutually_exclusive_group()
    conversation.add_argument(
        "--resume",
        choices=["last"],
        help="Continue the last saved conversation and its model settings.",
    )
    conversation.add_argument(
        "--new",
        action="store_true",
        help="Start a fresh conversation (the default).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("-p", "--prompt", help="Answer one prompt and exit.")
    mode.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Interactive chat, including when stdin is a pipe.",
    )
    parser.add_argument(
        "--system",
        help="System instruction retained throughout the conversation.",
    )
    parser.add_argument(
        "--ctx",
        type=_positive_int,
        help="Context size (default: 4096).",
    )
    parser.add_argument(
        "-n",
        "--max-tokens",
        type=_positive_int,
        help="Maximum reply tokens (default: 256).",
    )
    parser.add_argument(
        "--temperature",
        type=_temperature,
        help="Sampling temperature; 0 selects greedy decoding (default: 0.7).",
    )
    parser.add_argument("--seed", type=int, help="Sampling seed (default: -1).")
    parser.add_argument(
        "--threads",
        type=_positive_int,
        help="CPU threads (default: 4).",
    )
    parser.add_argument(
        "--gpu-layers",
        type=int,
        help="GPU layers; -1 offloads all layers, 0 uses CPU (default: -1).",
    )
    parser.add_argument(
        "--mtp",
        choices=["auto", "on", "off"],
        help="Embedded multi-token prediction; auto uses supported models (default).",
    )
    parser.add_argument(
        "--draft-width",
        type=int,
        choices=range(1, 9),
        help="MTP draft width (default: 2).",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print reply timings and MTP counters to stderr.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable native model diagnostics.",
    )
    return parser


def _print_reply(
    session: _ChatSession,
    prompt: str,
    *,
    stats: bool,
    output: TextIO,
    diagnostics: TextIO,
) -> None:
    started = time.perf_counter()
    first_text: float | None = None
    with contextlib.closing(session.reply(prompt)) as reply:
        for piece in reply:
            if first_text is None:
                first_text = time.perf_counter() - started
            print(piece, end="", flush=True, file=output)
    print(file=output, flush=True)
    if stats:
        elapsed = time.perf_counter() - started
        details = [f"elapsed={elapsed:.3f}s"]
        if first_text is not None:
            details.append(f"first_text={first_text:.3f}s")
        if session.speculative:
            perf = session.llm.perf()
            drafted = int(perf["n_drafted"])
            accepted = int(perf["n_accepted"])
            details.append(f"prefill={int(perf['n_mtp_prefill'])}")
            details.append(f"accepted={accepted}/{drafted}")
        print("[" + ", ".join(details) + "]", file=diagnostics)


def _interactive(session: _ChatSession, *, stats: bool) -> int:
    print(
        f"Ready. MTP {'on' if session.speculative else 'off'}. "
        "Type /help for commands.",
        file=sys.stderr,
    )
    while True:
        try:
            prompt = input("You> ").strip()
        except EOFError:
            return 0
        if not prompt:
            continue
        if prompt in {"/exit", "/quit"}:
            return 0
        if prompt == "/help":
            print(_HELP, end="", file=sys.stderr)
            continue
        if prompt == "/clear":
            session.clear()
            print("Conversation cleared.", file=sys.stderr)
            continue
        print("Assistant> ", end="", flush=True)
        try:
            _print_reply(
                session,
                prompt,
                stats=stats,
                output=sys.stdout,
                diagnostics=sys.stderr,
            )
        except KeyboardInterrupt:
            print("\nReply interrupted.", file=sys.stderr)
            if session.llm.is_stuck:
                return 1
        except (LlamaError, RuntimeError, ValueError) as exc:
            print(f"\nError: {exc}", file=sys.stderr)
            if session.llm.is_stuck:
                return 1
            print(
                "Use /clear to start over if the conversation fills the context.",
                file=sys.stderr,
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    store = ChatStore.default()
    saved: SavedChat | None = None
    if args.resume:
        try:
            saved = store.load()
        except (OSError, UnicodeError, ValueError) as exc:
            parser.error(
                f"cannot resume {store.path}: {exc}; use --new with --model "
                "to start a fresh conversation"
            )
    model = args.model
    if model is None:
        model = saved.settings.model if saved else os.environ.get("LLAMA_MODEL")
    if not model:
        parser.error("provide --model or set LLAMA_MODEL")
    try:
        model_path = _model_path(str(model)).resolve()
    except (argparse.ArgumentTypeError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    overrides = {
        field.name: getattr(args, field.name)
        for field in fields(ChatSettings)
        if field.name != "model" and getattr(args, field.name) is not None
    }
    settings = replace(
        saved.settings if saved else ChatSettings(model=str(model_path)),
        model=str(model_path),
        **overrides,
    )
    if settings.gpu_layers < -1:
        parser.error("--gpu-layers must be -1 or non-negative")

    def save_history(messages: list[dict[str, str]]) -> None:
        try:
            store.save(SavedChat(settings, messages))
        except OSError as exc:
            print(
                f"Could not save conversation to {store.path}: {exc}", file=sys.stderr
            )

    interactive = args.interactive or (args.prompt is None and sys.stdin.isatty())
    try:
        prompt = (
            None
            if interactive
            else (args.prompt if args.prompt is not None else sys.stdin.read())
        )
        if prompt is not None and not prompt.strip():
            parser.error("prompt must not be empty")
        config = LlamaConfig(
            model_path=str(model_path),
            n_ctx=settings.ctx,
            n_threads=settings.threads,
            n_threads_batch=settings.threads,
            n_gpu_layers=settings.gpu_layers,
            n_rs_seq=max(2, settings.draft_width) if settings.mtp != "off" else 2,
            load_mtp=settings.mtp != "off",
            verbose=args.verbose,
        )
        sampling = SamplingParams(temperature=settings.temperature, seed=settings.seed)
        with Llama(str(model_path), config=config, sampling=sampling) as llm:
            speculative = settings.mtp != "off" and llm.supports_speculative_mtp()
            if settings.mtp == "on" and not speculative:
                raise ValidationError(
                    "this model cannot use MTP; use --mtp auto or off"
                )
            session = _ChatSession(
                llm,
                system_prompt=settings.system,
                max_tokens=settings.max_tokens,
                speculative=speculative,
                draft_width=settings.draft_width,
                history=saved.messages if saved else (),
                on_commit=save_history,
            )
            if interactive:
                return _interactive(session, stats=args.stats)
            assert prompt is not None
            _print_reply(
                session,
                prompt,
                stats=args.stats,
                output=sys.stdout,
                diagnostics=sys.stderr,
            )
            return 0
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except BrokenPipeError:
        # A pipeline consumer such as head may close stdout during a reply.
        # Avoid a second BrokenPipeError from Python's final stdout flush.
        with open(os.devnull, "w") as sink:
            os.dup2(sink.fileno(), sys.stdout.fileno())
        return 0
    except (LlamaError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
