# Changelog — 2026-05-02

**Focus:** optimizations, build hygiene, and expanded model support.
No breaking API changes.

---

## C++ bindings (`src/bindings/llama_cpp.cpp`)

### Correctness
- **`Context::reset()` now holds `g_resource_mutex`.** Previously only `close()` held the lock while freeing `ctx_`, so a concurrent `reset()` could double-free the same context if GC and an explicit reset raced. `reset()` now takes the same lock, closing the race.
- **Dead code removed.** The unused `compute_top_logprobs()` helper was deleted.

### Performance
- **`get_state_data()` is zero-copy into Python bytes.** Replaced the `std::vector<uint8_t>` → `nb::bytes` path (which allocates + memcpys twice) with direct `PyBytes_FromStringAndSize(nullptr, size)` allocation and `_PyBytes_Resize()` if the serializer wrote fewer bytes. For a 250 MB KV state this saves one full-buffer allocation + one memcpy.

## Build system (`CMakeLists.txt`, `scripts/build_wheel.sh`)

- **Compiler detection uses `find_program`.** Dropped the hardcoded `/usr/local/bin/gcc-15` / `/usr/local/bin/g++-15` paths that broke on any system where `gcc-15` lives in `/usr/bin` (the default on Ubuntu 24.10+). CMake now resolves `find_program(NAMES gcc-15 gcc)` and `g++-15 g++` on Linux, and `clang` / `clang++` on macOS, falling back to distro defaults.
- **`-ffast-math` is now opt-in.** Enabling `-ffast-math` unconditionally was a silent correctness risk for an inference library (it alters softmax/sampling numerics and can set FTZ/DAZ process-wide). It's now gated behind `-DLLAMA_FAST_MATH=ON` (default OFF).
- **`scripts/build_wheel.sh`.** New helper that wraps `python -m build --wheel` with sane defaults:
  - `--portable` → `-DLLAMA_PORTABLE=ON`
  - `--fast-math` → `-DLLAMA_FAST_MATH=ON`
  - `--clean` → `rm -rf build dist`
  - `--install` → `uv pip install --force-reinstall dist/*.whl`
  - Honors `LLAMA_PREFIX` (default `/usr/local`), `PYTHON` (default `python3.14`), `CMAKE_BUILD_TYPE`, `JOBS`
  - Does **not** set `CMAKE_PREFIX_PATH` — scikit-build-core needs to inject that so `find_package(nanobind)` resolves inside the isolated build env. For non-`/usr/local` llama.cpp installs, set `LLAMA_PREFIX=/path` and the script will prepend it.

## Python — `src/llama_cpp/llama.py`

- **`_tokenize_stop_sequences(stop)` extracted.** The 9-line `str | int` → `list[list[int]]` loop was duplicated across `generate()`, `generate_stream()`, and `create_chat_completion()`. Consolidated into a single helper.
- **`_validate_prompt_token_count(n)` extracted.** The `2 × n_ctx` DoS guard was duplicated in `generate()` and `generate_stream()`. Consolidated.

Net: ~50 lines of duplication removed with no behavior change.

## UnifiedLLM — `src/llama_cpp/unified.py`

- **Granite 4.x support.** Added architecture-based detection in `detect_from_metadata()` — matches `granite`, `granitehybrid`, and `granitemoe` arch values. Updated the default `"granite"` config:
  - `supports_thinking = True`
  - `max_ctx = 131072`
  - `temperature = 0.7`, `top_p = 0.9`, `top_k = 40`, `repeat_penalty = 1.05`
  - `stop_sequences = ["<|end_of_text|>", "<|endoftext|>"]`
  - Thinking overrides: `think_temperature = 0.6`, `think_top_p = 0.95`

## Repo hygiene

- Removed 16 stale planning / review / changelog markdown files under `docs/` that were superseded by `_v2` counterparts or by `git log`.
- Removed 6 unrelated translator-demo outputs under `tools/outputs/`.
- Removed the `.cleanup-summary.md` artifact from a prior 2026-03-31 cleanup round.

## Verification

- `ruff check`: clean on all modified files
- `python3.14 -We -m py_compile`: clean across `src/`
- `tools/validate_pep758_765.py`: all 41 files compliant
- `clang-format --dry-run -Werror` on `llama_cpp.cpp`: clean

Pytest requires a rebuilt extension — rerun after:

```bash
./scripts/build_wheel.sh --clean --install
uv run pytest -q
```
