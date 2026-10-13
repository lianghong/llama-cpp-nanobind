# Terminal chat app

`llama-chat` runs a local GGUF model in a terminal. It keeps one model loaded,
streams replies, and reuses the conversation cache across completed turns.
Models with supported embedded MTP layers use draft width 2 automatically.

## Start chatting

Install the project into `.venv` after setting up matching
[llama.cpp headers and libraries](SYSTEM_LIBRARIES.md):

```bash
rtk proxy uv pip install --python .venv/bin/python -e .
rtk proxy .venv/bin/llama-chat --model /path/to/model.gguf
```

The module entry point works too:

```bash
rtk proxy .venv/bin/python -m llama_cpp --model /path/to/model.gguf
```

Enter one message per line. `/help` shows the commands:

| Input | Effect |
| --- | --- |
| `/clear` | Clear the conversation and its cache; keep the system instruction |
| `/exit` or `/quit` | Exit |
| Ctrl+D | Exit at the input prompt |
| Ctrl+C | Interrupt a reply; exit if pressed at the input prompt |

The Ctrl+C behavior above applies when you launch the installed
`.venv/bin/llama-chat` entry point directly in your terminal. The `rtk proxy`
wrapper exits on Ctrl+C, ending the session.

Interrupted replies are excluded from history. The next message continues
from completed turns. If the conversation fills the context, use `/clear`
or restart with a larger `--ctx`.

## Save and resume

Completed turns are saved automatically in interactive, single-prompt, and
piped-input modes. Continue the last conversation without supplying the model
or its settings again:

```bash
rtk proxy .venv/bin/llama-chat --resume last
```

The snapshot includes the model path, system instruction, conversation history,
context and reply limits, sampling settings, thread and GPU settings, and MTP
mode and draft width. Explicit command-line settings override saved values:

```bash
rtk proxy .venv/bin/llama-chat --resume last --ctx 8192
```

`--model` can override a saved path if the model file has moved. On resume, the
saved model path takes precedence over `LLAMA_MODEL`; the environment supplies
the model for a fresh conversation.

Start fresh with `--new` and a model path:

```bash
rtk proxy .venv/bin/llama-chat --new --model /path/to/model.gguf
```

Starting fresh is also the default when `--resume` is omitted. `--new` and
`--resume` cannot be combined. The previous snapshot is replaced after the first
completed reply; exiting a fresh chat without sending a message preserves it.
`/clear` immediately saves an empty conversation with its system instruction.
Interrupted or failed replies never enter saved history.

The snapshot is a local JSON file at `~/.local/state/llama-chat/last.json`, or
`$XDG_STATE_HOME/llama-chat/last.json` when `XDG_STATE_HOME` is an absolute path.
There is one snapshot: the latest successful save replaces the previous one,
including when multiple chat processes are running. Writes replace the file
atomically. A save failure reports an error to stderr; the previous file stays
intact and the conversation continues in memory.

Native model state is rebuilt from saved history on the first resumed reply.
Subsequent turns reuse the cache as usual. A missing or invalid snapshot produces
a clear error before loading the model; use `--new` with a model to start again.

## Single prompts and pipelines

`--prompt` answers once and exits. Standard output contains the streamed
reply, making it suitable for redirection:

```bash
rtk proxy .venv/bin/llama-chat -m /path/to/model.gguf \
  -p "Explain memory usage during inference." > answer.txt
```

Without `--prompt`, non-terminal stdin becomes one complete prompt, including
newlines:

```bash
rtk proxy .venv/bin/llama-chat -m /path/to/model.gguf < question.txt
```

Use `--interactive` to treat piped lines as successive chat messages and
commands.

`LLAMA_MODEL` supplies the model path when `--model` is omitted:

```bash
rtk proxy env LLAMA_MODEL=/path/to/model.gguf .venv/bin/llama-chat
```

## Model and generation settings

Use `--system "Be concise."` for a system instruction. Context defaults to
4096 tokens, replies to at most 256 tokens, temperature to 0.7, and CPU threads
to four. GPU layer offload defaults to `-1` (all); `--gpu-layers 0` uses CPU.

```bash
rtk proxy .venv/bin/llama-chat -m /path/to/model.gguf \
  --system "Be concise." --ctx 8192 --temperature 0 --seed 0
```

`--mtp auto` is the default: models without supported MTP use ordinary
decoding. `--mtp on` requires support and reports an error otherwise.
`--mtp off` uses ordinary decoding and skips loading MTP weights.
`--draft-width` accepts 1–8; start with 2. MTP performance depends on the model
and draft acceptance.

`--stats` prints elapsed time and first-text latency to stderr. In MTP mode,
it also shows prompt tokens prefilled and accepted/proposed draft counts.
`--verbose` enables native loading diagnostics.

```bash
rtk proxy .venv/bin/llama-chat --help
```

No model download or additional web dependency is needed.
