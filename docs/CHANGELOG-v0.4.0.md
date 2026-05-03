# Changelog — v0.4.0 (2026-05-03)

**Focus:** multi-cycle review fixes, Unsloth-aligned sampling presets, streaming concurrency hardening, and Qwen 3.5 / Gemma 4 operational updates.

No breaking API changes. Version bumped from `0.3.6` → `0.4.0` in `pyproject.toml`, `CMakeLists.txt`, and `src/llama_cpp/_about.py`.

---

## Correctness

### C++ bindings (`src/bindings/llama_cpp.cpp`)

- **`set_state_data` — safe no-copy use of the nb::bytes buffer protocol.** Lifetime invariant is now documented: the calling frame holds a strong reference to the `nb::bytes` argument, so the buffer survives GIL release during the heavy deserialize.
- **`prime_generation` — single-pass BOS prepend.** Replaced an O(n) `insert(begin(), …)` with a single-pass construction; avoids repeated shift-by-one on large primings.
- **`LoraAdapter` now tracks its parent `Model`.** `set_adapters_lora` validates each adapter belongs to the current context's model and raises `ValueError` on mismatch instead of segfaulting.
- **Designated initializers** for `llama_token_data` / `llama_token_data_array` across the sampling paths.
- **Signed/unsigned comparisons** switched to `std::cmp_*` to silence sign-compare warnings and remove surprising wraparound.
- **Logits pointers made `const`** where they are read-only; stray redundant null-writes removed.
- **Extracted `Model::read_c_string` template** for the four two-call snprintf-style llama.cpp APIs (`desc`, `meta_val_str`, `meta_key_by_index`, `meta_val_by_index`) — removes copy-paste bugs across the four call sites.

### Python — `Llama` (`src/llama_cpp/llama.py`)

- **`generate_stream` — lock acquired in the main thread before spawning the worker.** Previously the worker grabbed `self._lock` after dispatch, leaving a window where a concurrent caller could race with it. On join timeout the generator now raises `LlamaError` **and keeps the lock held**, preventing a zombie worker from corrupting state if the instance is reused.
- **Worker detokenizes inline.** The worker pushes `bytes` into the queue; the consumer never calls `Model` methods concurrently with the worker.
- **BOS suppression on session continuations.** `generate()` and `create_chat_completion()` now suppress the BOS when `reset_kv_cache=False`, so continuation turns don't insert a stray BOS mid-sequence.
- **`_validate_prompt_token_count` now covers `create_chat_completion`** (previously only `generate` / `generate_stream`).
- **Unknown `**sampling_overrides` raise `ValidationError` at the API boundary** instead of silently being ignored.
- **`Llama.__call__` uses accurate completion-token counting.** Replaced the lossy `detokenize → retokenize` roundtrip with `_generate_with_token_count`.
- **`generate_async(stream=True)`** is now actually incremental: bridged to `generate_stream` via an `asyncio.Queue` so chunks arrive as they are produced rather than being buffered.

### Python — `LlamaPool` (`src/llama_cpp/pool.py`)

- **Cross-loop reuse detection.** The pool records the event loop it was bound to at first use; subsequent use from a different loop raises a clear `RuntimeError` instead of deadlocking.
- **`close()` / `close_graceful()` serialized via `_close_lock`** — concurrent close calls are now idempotent.
- **`close_graceful` no longer busy-loops** when it sees a lone sentinel in the queue.

### Python — `UnifiedLLM` (`src/llama_cpp/unified.py`)

- **Partial-init safety.** `__init__` sets `_closed`, `llm`, and `backend` before any operation that can raise, so `close()` is safe even if construction fails halfway.
- **`close()` nulls `backend.llm`** before dropping the backend — closes a window where `backend` could still reach a freed context.

---

## Unsloth alignment (per upstream docs)

- **Qwen 3.5 defaults updated** for thinking mode:
  `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repeat_penalty=1.0`.
  (The Qwen 3-era `think_*` override block is no longer needed and was dropped.)
- **New `qwen3.5-coding` preset** — `temperature=0.6`, `presence_penalty=0.0` — tuned for precise coding / WebDev workloads.
- **`UnifiedLLM.sanitize_history()`** — new helper that strips prior-turn thought blocks from `assistant` messages before feeding conversation history back to the model. Handles Gemma 4 `<|channel>...<channel|>`, Qwen `<think>...</think>`, and bracket-style `[THINK]...[/THINK]`. Required by Gemma 4 per upstream; beneficial for Qwen 3 / 3.5 thinking models.
- **CLAUDE.md operational notes** added for:
  - Qwen 3.5: `UD-Q2_K_XL` minimum recommended quant, Ollama incompatibility (separate mmproj), `--cache-type-k bf16 --cache-type-v bf16` hint.
  - Gemma 4: **do NOT use the CUDA 13.2 runtime** (upstream flags it as producing poor outputs); recommended quants `Q8_0` (E2B/E4B) and `UD-Q4_K_XL` (26B-A4B / 31B).

---

## Code quality

- **`_classify_qwen35_variant` helper extracted** — removed duplicated size-list logic between `detect_from_metadata` and `detect_model_family`.
- **`LlamaConfig.add_bos` is no longer mutated at load time.** The effective value lives on `Llama._effective_add_bos`, so the same `LlamaConfig` can now be shared across multiple model loads without surprise mutation.
- **`Backend.strip_thinking` is now a proper protected interface method.** `ChatTemplateBackend` overrides it instead of `UnifiedLLM` reaching through to a private `_parse_thinking`.
- **`_parse_tool_calls`** now enforces a 1 MB cap before `json.loads` as defense in depth against hostile tool-call payloads.
- **Dead `logsumexp` helper removed.**
- **`build-tidy/` added to `.gitignore`.**

---

## Tests

New suites added:

- **`test_sanitize_history`** (8 cases): Gemma 4 channel block, Qwen think tags, bracket-style thinking, passthrough, role preservation, extra keys, non-string content, no-mutation guarantee.
- **`test_pool_close`** (7 cases): sentinel wake-up for `close` / `close_graceful`, checkout-after-close, concurrent-close idempotency, reinjection survival, no busy-loop on lone sentinel, cross-loop detection.
- **`test_validation`** (5 cases): unknown / misspelled / multi-key sampling overrides.
- **`test_partial_init`** (3 cases): `close()` safe after failed `__init__` via unknown-family, unknown-type, and manual `__new__` paths.
- **`test_auto_detect_bos`** updated: asserts the `LlamaConfig` is NOT mutated; checks `_effective_add_bos` instead.

---

## Lint / format

Clean on all tooling:

- `clang-format` / `clang-tidy` (0 user warnings)
- `ruff check` / `ruff format`
- `mypy`

---

## Upgrade

```bash
./scripts/build_wheel.sh --clean --install
uv run pytest -q
```

Produces `dist/llama_cpp_nanobind-0.4.0-cp314-cp314-linux_x86_64.whl`.
