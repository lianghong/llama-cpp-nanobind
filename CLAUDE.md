# CLAUDE.md

Guidance for Claude Code working in this repo.

## Project Overview

High-performance nanobind bindings for llama.cpp, packaged as a wheel-ready Python library with CUDA on by default. API mirrors `llama-cpp-python` where it doesn't conflict with upstream llama.cpp.

## Commands

### Setup
```bash
uv venv --python 3.14 .venv && source .venv/bin/activate
uv pip install -e .          # editable install
uv pip install -e .[dev]     # + dev tools
uv pip install -e .[test]    # + test deps
```

### Wheel build
```bash
./scripts/build_wheel.sh                 # native
./scripts/build_wheel.sh --portable      # no -march=native
./scripts/build_wheel.sh --clean --install
./scripts/build_wheel.sh --fast-math     # opt in (alters numerics)
```
Env: `PYTHON`, `LLAMA_PREFIX` (default `/usr/local`), `CMAKE_BUILD_TYPE`, `JOBS`. The script does **not** set `CMAKE_PREFIX_PATH` — scikit-build-core injects its own so `find_package(nanobind)` resolves to the build-isolation env. For non-`/usr/local` llama.cpp installs, set `LLAMA_PREFIX=/custom/path`.

### Tests
```bash
uv run pytest -q
uv run pytest tests/test_inference.py::test_basic_generation -v
```

### Code quality
```bash
ruff format src/ tests/ examples/ tools/
isort src/ tests/ examples/ tools/
ruff check src/ tests/ examples/ tools/
mypy src/llama_cpp/
clang-format -i src/bindings/llama_cpp.cpp
clang-tidy -p build-tidy src/bindings/llama_cpp.cpp     # needs compile_commands.json
python3.14 -We -m py_compile <file>                     # PEP 758/765
```

### Build config
```bash
CMAKE_BUILD_TYPE=RelWithDebInfo uv pip install -e .
CMAKE_ARGS="-DCMAKE_PREFIX_PATH=/opt/llama" uv pip install -e .
CMAKE_ARGS="-DLLAMA_PORTABLE=ON" uv pip install -e .
```

llama.cpp search order: `CMAKE_PREFIX_PATH` → `/usr/local` (Linux) → Homebrew (macOS) → standard CMake paths. Full setup walkthrough: `docs/SYSTEM_LIBRARIES.md`.

## Python 3.14 Compliance

Strict PEP 758 + PEP 765. All code compiled with `-We`. Audit: `docs/PEP758_PEP765_COMPLIANCE.md`.

- **PEP 758**: Parentheses are required around exception lists when using `as`. `except (ValueError, TypeError) as e:` ✅; `except ValueError, TypeError as e:` ❌.
- **PEP 765**: `return` / `break` / `continue` must not exit a `finally` block directly. Use a nested function/loop scope if needed.

## Architecture

```
src/bindings/llama_cpp.cpp     single-file nanobind module (C++ → Python)
src/llama_cpp/llama.py         Llama class — low-level inference API
src/llama_cpp/unified.py       UnifiedLLM — multi-model auto-detect
models/                        GGUF files (not committed)
```

**Linking**: Extension links against system-installed llama.cpp (no bundling, no RPATH). Linux defaults to `/usr/local/{include,lib}`; macOS auto-detects Homebrew via `brew --prefix llama.cpp`. Build with GPU support upstream (`-DGGML_CUDA=ON` Linux, `-DGGML_METAL=ON` macOS).

### Key design patterns

**State management** — `cur_pos_` tracks KV position; updated by `load_state`, `set_state_data`, and `kv_cache_seq_*` ops. `set_state_data` rolls back `cur_pos_` on load failure. State save/load uses `nb::bytes` buffer protocol (single memcpy, no per-element conversion).

**Sampler chain** — Grammar applies *before* sampler chain. Canonical order: DRY → penalties → top_n_sigma → top_k → top_p → min_p → XTC → temp_ext/temp → dist. Always read `cur_p.selected` from sampler, never argmax. In `generate_tokens_with_details`, `llama_sampler_apply` is called explicitly for logprobs — do **not** then call `generate_next`/`llama_sampler_sample`, which would re-apply the chain and advance the dist sampler's RNG.

**Streaming** — `generate_stream()` is true incremental (background thread + queue, GIL released during decode/sample, re-acquired only for the Python callback). `generate(..., stream=True)` is buffered (simpler, higher latency). All paths share `_token_to_text_incremental()` for UTF-8 decode across token boundaries (emoji, CJK).

**Stop sequences** — Multi-token via `generate_tokens_multi_stop()`. Tokens buffered up to `max_stop_len` before yielding so partial stop-prefix tokens never reach the consumer; remaining buffered tokens always flushed at end.

**Memory safety** — Destructors check `if (ptr_)` then null. `Model::close()`, `Context::close()`, `Context::reset()` hold `g_resource_mutex` to serialize frees. Logging holds `g_log_mutex` (concurrent `llama_log_set` calls). `g_model_count` atomic ref-counts the backend. Nanobind `keep_alive<1, 2>()` ties Context/SamplerChain/LoraAdapter to Model. No `__del__` (atexit + RAII instead). Context freed before Model.

**LlamaPool** — `pool_size` independent `Llama` instances (Llama is not thread-safe), `asyncio.Queue` checkout. `close()` is immediate; `close_graceful(timeout=30)` waits for in-flight returns. Records the binding event loop on first use; cross-loop reuse raises. VRAM ≈ `model_size × pool_size`. Optional `warmup=True` runs 3-token dummy inference per instance (compiles CUDA kernels, removes cold-start variance).

**Prompt-prefix KV reuse (`cache_prompt=True`, default)** — When paired with `reset_kv_cache=False`, `_apply_prefix_reuse` computes the LCP between cached and new prompts, calls `kv_cache_seq_rm(0, n_match, -1)`, and the C++ generator decodes only `priming[n_match:]` via `skip_decode_prefix`. Mirror invariant: `len(_cached_prompt_tokens) == kv_pos_max + 1`. Hybrid models (Qwen 3.5, Granite 4 hybrid, some flash-attn configs) report `memory_can_shift()=False`; for them `seq_rm` returns `False` and we fall back to `kv_cache_clear` + full re-prime — correctness preserved, speedup lost.

## Implementation Rules

### C++ bindings (`src/bindings/llama_cpp.cpp`)

1. Run `clang-format -i` and `clang-tidy` after changes.
2. Always release GIL for long ops: `nb::call_guard<nb::gil_scoped_release>()` or manual.
3. When mutating KV / loading state, save old `cur_pos_` and roll back on failure.
4. Reuse buffers (`single_batch_`) instead of per-call allocation.
5. Use RAII for temp resources (e.g. `BatchGuard` around `llama_batch`).
6. Apply grammar before the sampler chain.
7. Hold `g_resource_mutex` for resource frees; `g_log_mutex` for logging config.
8. Validate token range before indexing logits (sampler can return `LLAMA_TOKEN_NULL` = -1).
9. Validate `cur_p.selected ∈ [0, cur_p.size)` after `llama_sampler_apply` (grammar can empty the candidate set).
10. After `size_t → int32_t` casts, verify `static_cast<size_t>(result) == original`.
11. For two-call snprintf-style llama.cpp string APIs, verify size matches and explicitly null-terminate. Use `Model::read_c_string` template.
12. State save/load uses `nb::bytes` directly — manage GIL manually around heavy llama.cpp calls.
13. Logprobs path reads `cur_p.data[cur_p.selected].id` after explicit apply — do **not** call `generate_next` afterwards.

### Python wrappers (`src/llama_cpp/llama.py`, `unified.py`)

1. After `reset()`, call `_reapply_lora_adapters()`.
2. Validate `config.embeddings=True` before `embed()` / `create_embedding()`.
3. Use `add_special=True` when computing `max_tokens` for chat.
4. After tokenization, call `_validate_prompt_token_count(len(prompt_tokens))` (rejects > 2×n_ctx, DoS guard).
5. Validate stop sequences via `_validate_stop_sequences(stop)` on every entry point (max 20 sequences, 500 chars each). Used by `generate`, `generate_stream`, `create_chat_completion`.
6. Use `_tokenize_stop_sequences(stop)` for `str|int → list[list[int]]` conversion (single source of truth).
7. Use `generate_tokens_multi_stop()` when stops present and details not needed (fast path).
8. Use `_token_to_text_incremental()` for all streaming UTF-8 decode (handles split multi-byte chars; guaranteed final flush).
9. **Never** cache or reuse grammar samplers — `llama_sampler_accept` mutates internal state.
10. Use `if x is not None else fallback`, not `x or fallback`, when `0` / `0.0` are valid.
11. `LlamaConfig.add_bos` defaults to `None` (auto-detected from model metadata after load); effective value lives on `Llama._effective_add_bos` so the same `LlamaConfig` is reusable across loads. C++ has a `prompt[0] != bos` guard against double-BOS.
12. All public methods that touch `self.model`/`self.ctx` must call `_check_closed()` first.
13. **`cache_prompt` mirror invalidation**: any code path that mutates KV outside the generation loop must call `_invalidate_prompt_cache()` — `kv_cache_clear`, direct `kv_cache_seq_*`, `set_state_data`, `load_state`, `reset`, `embed`/`create_embedding`, `close`.
14. **BOS rule with `cache_prompt`**: `reset_kv_cache=False` + `cache_prompt=True` uses `_effective_add_bos` (model preference). The legacy "suppress BOS on continuation" rule applies only when `cache_prompt=False`.
15. `UnifiedLLM` warns when requested `n_ctx > model.n_ctx_train()` (quality may degrade).

### Test model

Default: `./models/Qwen3.5-4B-Q4_K_M.gguf` (override via `LLAMA_TEST_MODEL`; update `conftest.py` for different paths).

## Performance

**Build (Release)**: `-O3 -march=native -mtune=native -flto=auto -funroll-loops -fno-plt`. `-ffast-math` is **off by default** (alters softmax/sampling numerics, sets FTZ/DAZ process-wide); opt in with `-DLLAMA_FAST_MATH=ON` or `--fast-math`. LTO if supported. Compiler: `gcc-15`/`g++-15` then `gcc`/`g++` on Linux, `clang`/`clang++` on macOS.

**Runtime**: GIL released during all C++ ops including streaming. Per-token batch allocation eliminated. Fast stop-sequence path. Session continuation via `reset_kv_cache=False`. Prompt-prefix reuse via `cache_prompt=True` (default). O(n_vocab) candidate vector allocated once per call, not per token. `nb::bytes` for state save/load (single memcpy). Validation overhead < 0.1% (O(1) on hot paths).

## Testing

| File | Concern |
|---|---|
| `test_inference.py` | Core gen, chat, embeddings, state, logprobs |
| `test_async.py` | Async API correctness |
| `test_optimizations.py` | Embedding ctx reuse, KV cache, multi-token stops |
| `test_regressions.py` | State load pos, grammar sampling, LoRA persistence |
| `test_unified.py` | UnifiedLLM multi-model |
| `test_pool.py` | LlamaPool + warmup |
| `test_streaming.py` | Incremental streaming (needs model) |
| `test_streaming_logic.py` | Streaming threading (no model) |
| `test_prefix_reuse.py` | `cache_prompt` LCP, mirror invalidation |
| `test_sanitize_history.py` | Multi-turn thinking-block stripping |
| `test_pool_close.py` | Pool shutdown semantics |
| `test_partial_init.py` | `close()` after failed `__init__` |
| `test_validation.py` | Sampling override validation |

`conftest.py` provides `model_path` and `test_model` fixtures.

**Memory safety**: `examples/verify_double_free.py` runs 20 cleanup scenarios. Run with allocator checking:
```bash
MALLOC_CHECK_=3 python examples/verify_double_free.py
```

## Common Pitfalls

1. **Empty embeddings** — set `LlamaConfig(embeddings=True)`.
2. **Lost LoRA adapters** — auto-reapplied after `reset()`; don't reapply manually.
3. **Single-token stop only** — use `generate_tokens_multi_stop()` for multi-token stops like `<|end_of_turn|>`.
4. **Context overflow** — `UnifiedLLM` validates `max_tokens > 0` and raises.
5. **Thread safety** — no concurrent calls on a single `Llama`; use multiple instances or `LlamaPool`.
6. **Global logging** — `verbose=False` affects all instances (llama.cpp limitation).
7. **Grammar sampler reuse** — never; create fresh each generation.
8. **Falsy traps** — `is not None`, not `or`, when `0`/`0.0` are valid.
9. **High-compression prompts** — `"a" * 10MB` tokenizes to massive counts; rejected by 2×n_ctx guard.
10. **Translation hallucination** — LLMs editorialize/sentiment-drift; mitigate with low temp (0.1–0.3) + structured FAITHFULNESS/FLUENCY/STYLE prompt sections.
11. **14B+ on 16GB Apple Silicon** — ~10 tok/s (mem-bw bound); use `--ctx 4096` (default 10240 may crash Metal); prefer 4–8B models.
12. **Streaming thread leak** — early termination during stuck C++ gen logs a warning; don't reuse the instance until the worker returns.
13. **Quantized KV cache** — only F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL allowed. K-quants (Q4_K, …) rejected by `LlamaConfig`. Quantized V (anything besides F32/F16/BF16) requires `flash_attn=1` — `LlamaConfig` raises `ValidationError` otherwise.
14. **`reset_kv_cache=False` semantics** — with default `cache_prompt=True`, divergent prompts trim KV to LCP, not append. For "stitch arbitrary prompts", pass `cache_prompt=False`.

## Model Support (UnifiedLLM)

`UnifiedLLM` only supports a curated set; other GGUFs raise `UnsupportedModelError` at construction. For unsupported models, drop to `Llama`.

| Family | Variants | Context | Sampling defaults | Source |
|---|---|---|---|---|
| **Qwen 3.5** | 0.8/2/4/9B (small, no thinking); 27B / 35B-A3B / 122B-A10B / 397B-A17B (thinking) | 262K (1M YaRN) | small T=0.7/top_p=0.8; thinking T=1.0/top_p=0.95; top_k=20, presence=1.5, repeat=1.0 | unsloth.ai/docs/models/qwen3.5 |
| **Qwen 3.6** | 27B dense, 35B-A3B MoE; thinking + instruct (general/reasoning) | 262K (1M YaRN) | thinking T=1.0/top_p=0.95; instruct general T=0.7/top_p=0.8; reasoning T=1.0/top_p=0.95 | unsloth.ai/docs/models/qwen3.6 |
| **Gemma 4** | E2B/E4B (128K), 26B-A4B/31B (256K). Thinking via `<\|think\|>` system prefix | 128K / 256K | T=1.0, top_p=0.95, top_k=64, repeat=1.0 | unsloth.ai/docs/models/gemma-4 |
| **IBM Granite 4.1** | 3B / 8B / 30B dense | 16K min, 131K max | Deterministic: T=0, top_p=1, top_k=0 | unsloth.ai/docs/models/ibm-granite-4.1 |

**Opt-in presets** (auto-detect picks the family default; pass `family="..."` to override):
- `qwen3.5-coding`, `qwen3.6-coding` — T=0.6, presence=0.0 for coding/WebDev/Arena.
- `qwen3.6-instruct`, `qwen3.6-instruct-reasoning` — non-thinking variants.

`UnifiedLLM.chat(messages, *, thinking=False, sanitize_history=True, reset_kv_cache=True, cache_prompt=True)` auto-strips prior thinking blocks. Single-turn `generate(prompt, ...)` and `generate_with_thinking(...)` remain. Detection state machine: filename match (`detect_model_family`) → metadata refinement (`detect_from_metadata`) — see `src/llama_cpp/unified.py`.

**Multi-turn hygiene**: `sanitize_history(messages)` strips `<|channel>...<channel|>` (Gemma 4), `<think>...</think>` (Qwen), and `[THINK]...[/THINK]` from historical `assistant` messages. Required by Gemma 4 upstream; beneficial for all thinking models.

### Operational notes (per upstream)

- **Qwen 3.5**: prefer `UD-Q2_K_XL`+. Gibberish often = ctx too low; try `--cache-type-k bf16 --cache-type-v bf16`. Ollama doesn't work (separate mmproj vision files); use llama.cpp-compatible backends.
- **Qwen 3.6**: MTP variants (`*-MTP.gguf`) accept the same presets. Disable thinking via `--chat-template-kwargs '{"enable_thinking":false}'` or `family="qwen3.6-instruct"`.
- **Gemma 4 / Granite 4.1**: **Do NOT use the CUDA 13.2 runtime** — upstream flags poor outputs. Recommended Gemma quants: `Q8_0` (E2B/E4B), `UD-Q4_K_XL` (26B-A4B / 31B). Granite is deterministic by default; override sampling kwargs for creativity.

## Updating llama.cpp

```bash
cd /path/to/llama.cpp && mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DGGML_CUDA=ON
make -j$(nproc) && sudo make install
cd /path/to/llama-cpp-nanobind && uv pip install -e .
```
On macOS: `brew install llama.cpp`. Update C++ bindings if the upstream API changed.

## Reference

- API surface: `docs/API.md`
- Per-version changelogs: `docs/CHANGELOG-v0.3.6.md`, `docs/CHANGELOG-v0.4.0.md`, `docs/CHANGELOG-2026-05-02.md`, `docs/CHANGELOG-2026-05-27.md`
- System library setup: `docs/SYSTEM_LIBRARIES.md`
- Python 3.14 compliance audit: `docs/PEP758_PEP765_COMPLIANCE.md`
- Code review history: `docs/CODE_REVIEW_FIXES_2026-03-31_v2.md`, `docs/IMPROVEMENTS_2026-03-31_v2.md`
