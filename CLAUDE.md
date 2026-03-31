# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

High-performance nanobind bindings for llama.cpp, packaged as a wheel-ready Python library with CUDA enabled by default. The interface mirrors llama-cpp-python where it does not conflict with upstream llama.cpp.

## Build System & Commands

### Development Setup

```bash
# Create virtual environment with uv
uv venv --python 3.14 .venv
source .venv/bin/activate

# Install package in editable mode
uv pip install -e .

# Install with dev dependencies
uv pip install -e .[dev]

# Install with test dependencies
uv pip install -e .[test]
```

### Testing

```bash
# Run all tests
uv run pytest -q

# Run specific test file
uv run pytest tests/test_inference.py -v

# Run specific test
uv run pytest tests/test_inference.py::test_basic_generation -v
```

### Code Quality

```bash
# Python: format
ruff format src/ tests/ examples/ tools/

# Python: sort imports
isort src/ tests/ examples/ tools/

# Python: lint
ruff check src/ tests/ examples/ tools/

# Python: type checking
mypy src/llama_cpp/

# C++: format (Google-based, 100-col, 2-space indent)
clang-format -i src/bindings/llama_cpp.cpp

# C++: static analysis (requires compile_commands.json)
clang-tidy -p build-tidy src/bindings/llama_cpp.cpp

# Python 3.14: Verify PEP 758/765 compliance
python3.14 -We -m py_compile <file>
```

### Python 3.14 Exception Handling Requirements

**CRITICAL:** This project requires strict compliance with Python 3.14's PEP 758 and PEP 765.

#### PEP 758: Unparenthesized Exception Lists

**Allowed Forms:**
```python
# ✅ Form 1: Parenthesized (existing, always valid)
except (ValueError, TypeError):
    ...

# ✅ Form 2: Unparenthesized (NEW - allowed without 'as')
except ValueError, TypeError:
    ...

# ✅ Form 3: With capture (REQUIRES parentheses)
except (ValueError, TypeError) as e:
    ...

# ❌ INVALID: Unparenthesized with 'as'
except ValueError, TypeError as e:  # SyntaxError
    ...
```

**Rule:** Parentheses are REQUIRED when using the `as` keyword to capture exception instances.

#### PEP 765: Control Flow in Finally Blocks

**Disallowed Patterns:**
```python
# ❌ INVALID: return from finally
def f():
    try:
        ...
    finally:
        return 42  # SyntaxWarning (future SyntaxError)

# ❌ INVALID: break from finally
for x in items:
    try:
        ...
    finally:
        break  # SyntaxWarning

# ❌ INVALID: continue from finally
for x in items:
    try:
        ...
    finally:
        continue  # SyntaxWarning
```

**Allowed Patterns:**
```python
# ✅ VALID: Control flow in nested scope
try:
    ...
finally:
    def inner():
        return 42  # OK - exits inner function, not finally
    
    for x in items:
        break  # OK - exits inner loop, not finally
```

**Rule:** `return`, `break`, and `continue` statements MUST NOT exit a `finally` block directly. Use nested scopes if control flow is needed.

**Verification:** All code is compiled with `-We` flag. See `docs/PEP758_PEP765_COMPLIANCE.md` for audit report.

### Build Configuration

The project uses scikit-build-core with CMake. Key environment variables:

```bash
# Custom build type
CMAKE_BUILD_TYPE=RelWithDebInfo uv pip install -e .

# Custom llama.cpp install prefix (overrides defaults)
CMAKE_ARGS="-DCMAKE_PREFIX_PATH=/opt/llama" uv pip install -e .

# Portable build (no -march=native)
CMAKE_ARGS="-DLLAMA_PORTABLE=ON" uv pip install -e .
```

### System Library Search Order

The build system searches for llama.cpp in this priority order:

1. **User override**: `CMAKE_PREFIX_PATH` environment variable
2. **Linux**: `/usr/local` (explicitly checked for system installs)
3. **macOS**: Homebrew llama.cpp location (auto-detected)
4. **Fallback**: Standard CMake search paths (`/usr`, etc.)

**Example:** Install llama.cpp to `/usr/local`:

```bash
# Build and install llama.cpp
cd /path/to/llama.cpp
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DGGML_CUDA=ON
make -j$(nproc)
sudo make install

# Verify installation
ls /usr/local/include/llama.h       # Headers
ls /usr/local/lib/libllama.so       # Shared library

# Build Python bindings (automatically finds /usr/local)
cd /path/to/llama-cpp-nanobind
uv pip install -e .
```

**To verify which libraries are being used:**

```bash
# During build, CMake prints:
# -- Using system llama.cpp from /usr/local
# -- Found llama.h: /usr/local/include
# -- Found libllama: /usr/local/lib/libllama.so
# -- Found libggml: /usr/local/lib/libggml.so
# ...

# After install, check Python extension links:
ldd .venv/lib/python3.14/site-packages/llama_cpp/_llama*.so
```

## Architecture

### Component Structure

1. **External Dependencies** (`models/`)
   - llama.cpp **must be installed on the system** (headers + shared libraries)
   - **Linux**: Defaults to `/usr/local/include` and `/usr/local/lib`
   - **macOS**: Auto-detects Homebrew install via `brew --prefix llama.cpp`
   - CMake discovers paths via `find_path()` and `find_library()`
   - **No bundled dependencies**: Project links against system-installed llama.cpp
   - `models/`: GGUF model files (not included)

2. **C++ Bindings** (`src/bindings/llama_cpp.cpp`)
   - Single-file nanobind extension module
   - Exposes llama.cpp C API to Python
   - **Critical**: GIL released during heavy C++ operations (decode, generate, tokenize)
   - Maintains internal state: `cur_pos_` for KV cache position tracking
   - Reuses `single_batch_` to eliminate per-token allocations
   - Style enforced by `.clang-format` (Google-based) and `.clang-tidy`
   - All RAII classes explicitly delete copy and move operations (rule of 5)

3. **Python Wrappers** (`src/llama_cpp/`)
   - `llama.py`: Core `Llama` class with high-level inference API
   - `unified.py`: `UnifiedLLM` class for multi-model support (auto-detects Qwen3, Gemma, Mistral, etc.)
   - `__init__.py`: Package initializer with public API exports

4. **Library Linking**
   - Extension links against system-installed llama.cpp shared libraries
   - No RPATH or library bundling — relies on system linker paths
   - On macOS, Homebrew llama.cpp is auto-detected via `brew --prefix`

### Key Design Patterns

#### State Management
- **KV Cache Position**: `cur_pos_` tracks context position, updated by:
  - `load_state()`, `set_state_data()`: Sync from KV cache after load
  - `kv_cache_seq_rm()`, `kv_cache_seq_keep()`, `kv_cache_seq_add()`: Update when modifying sequence 0
- **State Save/Load**: Uses `nb::bytes` buffer protocol for zero-copy transfer between Python and C++ (no per-element conversion); GIL managed manually — released during heavy llama.cpp calls, held for Python object construction
- **LoRA Adapters**: Tracked in `_lora_configs` list, reapplied via `_reapply_lora_adapters()` after `reset()`

#### Sampling Pipeline
- Grammar constraints apply **before** sampler chain
- Sampler chain uses canonical ordering: DRY → penalties → top_n_sigma → top_k → top_p → min_p → XTC → temp_ext/temp → dist
- Sampler chain respects temperature/top_p/top_k even with grammar
- Use `cur_p.selected` from sampler, not argmax
- In `generate_tokens_with_details` (logprobs path), use `cur_p.selected` directly after explicit `llama_sampler_apply` — do NOT call `generate_next`/`llama_sampler_sample` which would apply the chain a second time (advancing the dist sampler's RNG)
- New samplers: DRY (anti-repetition on raw logits), XTC (cross-token consistency on filtered candidates), dynamic temperature (`temp_ext`), top-n-sigma (truncate by standard deviations)

#### Stop Sequences
- Multi-token stop sequences supported via `generate_tokens_multi_stop()`
- Fast path when logprobs/echo not needed (avoids O(n_vocab) overhead)

#### Async Support
- Thread-pool based (not truly async within C++)
- Single lock per `Llama` instance (concurrent calls serialize)
- For true parallelism, use multiple `Llama` instances

#### Streaming Generation
- **`generate_stream()`**: True incremental streaming using background thread + queue
  - Tokens yielded immediately as generated (low latency)
  - Background worker thread runs C++ generation with GIL released
  - Main thread yields tokens from queue as they arrive
  - Exceptions propagated from worker thread to caller
  - Early termination via `threading.Event` cancellation flag (checked in callback)
  - 5s join timeout allows current decode step to complete
  - **Thread leak detection**: If worker thread doesn't stop within timeout, logs warning about potential data race if instance reused
  - **Worker liveness check**: `token_queue.get(timeout=0.5)` with `thread.is_alive()` check — prevents permanent block if worker thread dies unexpectedly
  - **Multi-token stop sequence buffering**: tokens are buffered up to `max_stop_len` before yielding to prevent partial stop sequence tokens from reaching the consumer
  - **Flush invariant**: remaining buffered tokens at end-of-generation are always yielded (partial stop sequence prefixes are NOT treated as stops)
  - **UTF-8 decoding**: Uses `_token_to_text_incremental()` helper for consistent handling of multi-byte characters split across token boundaries
- **`generate(..., stream=True)`**: Buffered streaming
  - All tokens generated first, then yielded (higher latency)
  - Simpler implementation, no threading overhead
  - Suitable when latency not critical
  - **UTF-8 decoding**: Uses shared `_token_to_text_incremental()` helper with guaranteed final flush
- **All streaming paths**: Unified UTF-8 handling via `_token_to_text_incremental()` ensures consistent behavior across `generate_stream()`, `generate(stream=True)`, and `create_chat_completion(stream=True)`. Incremental decoder properly handles emoji, CJK, and other multi-byte UTF-8 characters that may span token boundaries.

#### Memory Safety (Double-Free Prevention)
- **C++ classes**: All destructors check `if (ptr_)` before free, then set `ptr_ = nullptr`
- **Thread-safe close**: `Model::close()` and `Context::close()` hold `g_init_mutex` to prevent races between GC/`__del__` and explicit `close()`
- **Thread-safe logging**: Logging configuration functions (`set_log_level`, `disable_logging`, `reset_logging`) hold `g_log_mutex` to prevent concurrent `llama_log_set()` calls during multi-threaded initialization
- **Backend ref-counting**: `g_model_count` atomic prevents backend double-free
- **Nanobind `keep_alive`**: Context, SamplerChain, LoraAdapter keep Model alive via `nb::keep_alive<1, 2>()`
- **LoRA adapters**: Freed automatically by llama.cpp with the model (destructor is `= default`)
- **No `__del__`**: Neither `Llama` nor `UnifiedLLM` uses `__del__` (avoids GIL issues during shutdown); cleanup via atexit + RAII
- **`close()` idempotency**: `Llama._closed` flag and `UnifiedLLM`'s `self.llm is None` check
- **Destruction order**: Context freed before Model (C++ dependency)
- **Instance tracking**: `weakref.ref` sets prevent circular references; atexit handler calls `close()` on all live instances
- **State load rollback**: `set_state_data()` saves old `cur_pos_` and restores it if load fails, maintaining context position integrity

#### Parallel Inference (LlamaPool)
- **Purpose**: True concurrent processing with multiple model instances
- **Architecture**:
  - Creates `pool_size` independent `Llama` instances
  - `asyncio.Queue`-based checkout/return ensures exclusive instance access (Llama is not thread-safe)
  - Each instance is checked out by at most one request at a time
  - Instances returned to pool after use via try/finally
- **Shutdown**: Two modes:
  - `close()`: Immediate force-close; logs warning if instances are checked out (in-flight)
  - `close_graceful(timeout=30.0)`: Waits for in-flight requests to return, then force-closes after timeout
  - `__aexit__` calls `close_graceful()` — async context manager is the recommended pattern
- **Timeout Handling**: `_checkout_instance()` distinguishes between "pool busy" timeout and "pool closed" error — prevents infinite retry loops
- **GPU Memory**: `VRAM ≈ model_size × pool_size`
- **Model Warmup** (optional, `warmup=True`):
  - Runs dummy inference (3 tokens) on each instance during init
  - Pre-loads GPU caches, compiles CUDA kernels
  - Eliminates cold-start latency variability
  - Adds 1-3s to initialization time
  - Recommended for production APIs with strict SLAs
  - Warmup failures are non-fatal (logged as warnings)

## Critical Implementation Details

### When Modifying C++ Bindings (`src/bindings/llama_cpp.cpp`)

1. **Run `clang-format -i` and `clang-tidy`** after changes to maintain code style and catch issues
2. **Always release GIL** for long operations: `nb::call_guard<nb::gil_scoped_release>()` or manual `nb::gil_scoped_release`
3. **Update `cur_pos_` with rollback support**: When modifying KV cache or loading state, save old position for rollback on failure. State load operations must validate success before updating `cur_pos_`
4. **Reuse buffers** (like `single_batch_`) instead of per-call allocation
5. **Use RAII for temporary resources**: `Context::decode` uses `BatchGuard` struct for `llama_batch` cleanup instead of try/catch
6. **Respect sampler chain** after grammar constraints
7. **Hold `g_init_mutex` for resource freeing, `g_log_mutex` for logging**: `close()` methods hold `g_init_mutex` to prevent races with GC; logging functions hold `g_log_mutex` to prevent concurrent `llama_log_set()` calls
8. **Validate token range before indexing logits**: sampler can return `LLAMA_TOKEN_NULL` (-1); always check `token >= 0 && token < n_vocab` before `logits[token]`
9. **Validate sampler selection**: After `llama_sampler_apply()`, always check `cur_p.selected >= 0 && cur_p.selected < cur_p.size` before accessing `cur_p.data[cur_p.selected]` — grammar constraints can create empty candidate sets
10. **Validate integer casts**: After casting `size_t` to `int32_t`, verify cast didn't truncate: `static_cast<size_t>(result) == original_size`
11. **Validate string buffers**: For llama.cpp API calls that write to string buffers, verify returned size matches expected and explicitly null-terminate
12. **Use `nb::bytes` for binary data**: state save/load uses `nb::bytes` directly (not `std::vector<uint8_t>`) to avoid per-element Python↔C++ conversion; manage GIL manually when mixing Python object construction with heavy C++ calls
13. **Logprobs path uses `cur_p.selected` directly**: In `generate_tokens_with_details`, `llama_sampler_apply` is called explicitly for logprob computation, then the selected token is read from `cur_p.data[cur_p.selected].id` — do NOT call `generate_next`/`llama_sampler_sample` after an explicit apply, as it would re-apply the chain and advance the dist sampler's RNG

### When Modifying Python Wrappers (`src/llama_cpp/llama.py`, `unified.py`)

1. **LoRA persistence**: Always call `_reapply_lora_adapters()` after context reset
2. **Embeddings**: Validate `config.embeddings=True` before embedding operations
3. **Token counting**: Use `add_special=True` when calculating max_tokens for chat
4. **Tokenized prompt validation**: After tokenizing prompt, validate `len(prompt_tokens) <= n_ctx() * 2` to prevent OOM from high-compression text (e.g., `"a" * 10MB` could tokenize to massive token counts)
5. **Stop sequences validation**: Always call `_validate_stop_sequences(stop)` on all entry points that accept stop sequences. This validates count (max 20) and length (max 500 chars per sequence). All generation entry points use this: `generate()`, `generate_stream()`, `create_chat_completion()`
6. **Stop sequences implementation**: Use `generate_tokens_multi_stop()` when stop sequences present and details not needed (fast path)
7. **Streaming UTF-8 decoding**: Use `_token_to_text_incremental(tokens)` for all streaming paths — handles multi-byte UTF-8 characters split across token boundaries with automatic final flush
8. **Grammar samplers are stateful**: Never cache or reuse them — `llama_sampler_accept` mutates internal state, so each generation must create a fresh sampler
9. **Falsy-value defaults**: Use `if x is not None else fallback` (not `x or fallback`) when `0` or `0.0` is a valid value
10. **BOS auto-detection**: `LlamaConfig.add_bos` defaults to `None` (auto-detected from model metadata via `llama_vocab_get_add_bos` after model load). The C++ generation functions have a guard `if (add_bos && front != bos)` as a safety net against double BOS
11. **Negative value validation**: Always validate that token counts and similar integer values are non-negative before arithmetic operations
12. **Training context awareness**: `UnifiedLLM` logs warning when `n_ctx > model.n_ctx_train()` — generation quality may degrade beyond training context
13. **Close checks**: All public methods that access `self.model` or `self.ctx` must call `_check_closed()` first to provide clear error messages instead of `AttributeError` on closed instances

### Model File Requirement

Default test model: `./models/Qwen3-8B-Q6_K.gguf`

Update `conftest.py` if using different model paths.

## Performance Considerations

### Build Optimizations (Release)
- `-O3 -march=native -mtune=native -flto=auto -ffast-math -funroll-loops`
- LTO enabled if supported

### Runtime Optimizations
- GIL released during C++ operations (v0.3.0), including streaming generation (v0.3.5)
- Per-token batch allocation eliminated (v0.3.0)
- Fast stop sequence path (v0.3.0)
- Session-style continuation with `reset_kv_cache=False` (v0.3.1)
- True incremental streaming via background thread (v0.3.2)
  - Tokens yielded as generated, not buffered
  - Low time-to-first-token for responsive UIs
  - Perfect for SSE/WebSocket streaming endpoints
  - GIL released during C++ decode/sample, re-acquired only for Python callback (v0.3.5)
  - Multi-token stop sequence buffering prevents partial stop tokens in output
- Grammar samplers always created fresh (stateful — caching causes incorrect results)
- O(n_vocab) candidate vector allocated once per generation call, not per token (avoids repeated 32K–128K element allocations)
- State save/load uses `nb::bytes` buffer protocol — single memcpy instead of per-element Python↔C++ conversion
- RAII `BatchGuard` in `Context::decode` replaces manual try/catch for `llama_batch` cleanup
- Validation overhead < 0.1%: All safety checks (token counts, buffer sizes, sampler selection) are O(1) operations on hot paths

## Testing Strategy

Test files organized by concern:
- `test_inference.py`: Core generation, chat, embeddings, state management, logprobs
- `test_async.py`: Async API correctness
- `test_optimizations.py`: Embedding context reuse, KV cache, multi-token stops
- `test_regressions.py`: State load position tracking, grammar sampling, LoRA persistence
- `test_unified.py`: UnifiedLLM multi-model support
- `test_pool.py`: LlamaPool parallel inference and model warmup
- `test_streaming.py`: True incremental streaming API (requires model)
- `test_streaming_logic.py`: Streaming threading logic (no model required)

Key test fixture: `conftest.py` provides `model_path` and `test_model` fixtures.

### Memory Safety Verification

`examples/verify_double_free.py` exercises 20 resource cleanup scenarios for both `Llama` and `UnifiedLLM`:
- Double `close()`, context manager + close, `close()` then `del`
- State save/load round-trips then close
- GC pressure interleaved with close
- Use-after-close (must raise, not crash)
- Multi-instance close in different orders
- Rapid create-close loops
- `del` without close (RAII / `__del__` paths)
- Mixed `UnifiedLLM` + `Llama` instance close

Run with glibc heap checking for allocator-level corruption detection:
```bash
MALLOC_CHECK_=3 python examples/verify_double_free.py
```

## Common Pitfalls

1. **Empty embeddings**: Ensure `LlamaConfig(embeddings=True)` when using `embed()` or `create_embedding()`
2. **Lost LoRA adapters**: After `reset()`, adapters are automatically reapplied
3. **Single-token stop sequences**: Use `generate_tokens_multi_stop()` for multi-token stops like `<|end_of_turn|>`
4. **Context overflow**: `UnifiedLLM` validates `max_tokens > 0` and raises on overflow
5. **Stale KV position**: State load/save automatically maintains `cur_pos_` with rollback on failure
6. **Thread safety**: Do NOT call methods concurrently on same instance - use multiple instances or LlamaPool
7. **Global logging**: `verbose=False` affects ALL instances (llama.cpp limitation); concurrent logging configuration protected by mutex
8. **Grammar sampler reuse**: Never cache grammar samplers — they are stateful and must be created fresh each generation
9. **Falsy-value traps**: Use `is not None` checks (not `or`) when `0`/`0.0` are valid parameter values
10. **Logprobs token bounds**: In the logprobs/details path (`generate_tokens_with_details`), sampler selection is validated before access — empty candidate sets raise explicit error
11. **High-compression prompts**: Text like `"a" * 10_000_000` can tokenize to massive token counts even though text length is < 10MB limit. Validation now rejects prompts > 2×n_ctx tokens after tokenization
12. **Training context limits**: `UnifiedLLM` warns when `n_ctx` exceeds model's training context — generation quality may degrade beyond this limit
13. **Translation hallucination**: LLMs may editorialize, reverse sentiment, or inject opinions when translating opinionated/sarcastic text. Mitigate with: low temperature (0.1–0.3), explicit system prompt rules against editorializing/moral hedging, and structured prompt sections (FAITHFULNESS/FLUENCY/STYLE) that models attend to better than flat rule lists
14. **14B+ models on 16GB Apple Silicon**: Expect ~10 tok/s generation (memory-bandwidth-limited). Default context 10240 may crash Metal; use `--ctx 4096`. Performance is near hardware ceiling — use 8B or 4B models for better throughput
15. **Streaming thread leaks**: If `generate_stream()` is terminated early and C++ generation is stuck, a warning is logged. Avoid reusing the instance until thread completes

## Integration with llama.cpp

This project links against system-installed llama.cpp. When updating:

1. Build llama.cpp with GPU support (`-DGGML_CUDA=ON` on Linux, `-DGGML_METAL=ON` on macOS)
2. Install to system: `sudo cmake --install build`
3. Rebuild the extension: `uv pip install -e .`
4. Update C++ bindings if API changed

On macOS, `brew install llama.cpp` provides headers and libraries that CMake auto-detects.

## Model Support

UnifiedLLM auto-detects model families by filename patterns:
- Aya (Cohere's multilingual model, Tiny Aya variants, 70+ languages)
- Qwen3 (with thinking/non-thinking mode detection)
- Qwen3.5 (hybrid attention, 262K context / 1M via YaRN, thinking default-on for large models, disabled for 0.8B-9B)
- Gemma
- TranslateGemma (Google's 55-language translation model, 128K context)
- Mistral
- GPT-OSS
- Phi
- GLM4 (with GLM-4.7 thinking mode support, REAP compression variants)
- Granite
- MiniCPM

See `src/llama_cpp/unified.py` for family detection logic and `examples/translategemma_example.py` for translation usage.

## Translation Example (`examples/translate.py`)

General-purpose translation tool using `UnifiedLLM` with configurable target language:

```bash
python examples/translate.py --model models/Qwen3-8B-Q6_K.gguf --ctx 8192
python examples/translate.py --model models/Qwen3-8B-Q6_K.gguf -t Japanese
python examples/translate.py --model models/Qwen3-8B-Q6_K.gguf --thinking
python examples/translate.py --model models/Qwen3-8B-Q6_K.gguf --temperature 0.1 -o
```

Key design decisions:
- **`-t` / `--target-lang`**: Target language (default: "Simplified Chinese"). System prompt and user prompt are templated with `{target_lang}` at runtime
- **Structured system prompt**: Three named sections (FAITHFULNESS, FLUENCY, STYLE) + SPECIFICS for Markdown/proper nouns/numbers/cultural references. Structured sections get better model attention than flat numbered lists
- **Sentiment fidelity**: Explicit anti-hedging rules ("never soften, editorialize, or moral-hedge") counter LLM tendency to neutralize critical/sarcastic/controversial text
- **Markdown-aware**: Prompt explicitly instructs to preserve link/image/heading syntax and translate display text only, never URLs
- **Low default temperature (0.3)**: Reduces hallucination and sentiment drift in translations
- **`--temperature` CLI arg**: Overrides the model's default temperature after construction via `llm.model_config.temperature`
- **VRAM check**: Estimates GPU memory usage before loading and warns if insufficient

## Recent Improvements (v0.3.6)

### Validation & Safety Enhancements
- **Tokenized prompt limits**: Rejects prompts that tokenize to > 2×n_ctx tokens, preventing OOM from high-compression text (DoS protection)
- **State load integrity**: `set_state_data()` rolls back `cur_pos_` if load fails, maintaining context position consistency
- **Sampler validation**: Bounds-checking before accessing sampler selection, with explicit error for empty candidate sets
- **Integer overflow protection**: Validates `size_t` to `int32_t` casts in tokenization paths
- **String buffer validation**: Explicit null-termination and size verification for all llama.cpp string API calls
- **Negative value guards**: Defense against corrupted token counts from C++ binding

### Concurrency & Thread Safety
- **Logging mutex**: `g_log_mutex` protects `llama_log_set()` calls, preventing races during concurrent initialization
- **Pool timeout handling**: `LlamaPool._checkout_instance()` distinguishes "pool closed" from "pool busy" timeouts
- **Thread leak detection**: `generate_stream()` logs warning if worker thread doesn't stop within 5s timeout

### Quality of Life
- **Training context warnings**: `UnifiedLLM` warns when requested `n_ctx` exceeds model's training context
- **Better error messages**: Validation errors include token counts, limits, and actionable guidance

### Performance Impact
- All validation overhead < 0.1% (O(1) checks on hot paths)
- No changes to core generation algorithms
- Backward compatible: no breaking changes to well-formed code

## Recent Improvements (2026-03-31)

### Code Review Fixes - Second Review Cycle

**HIGH Priority** (2 issues fixed):
- **Missing close guards**: Added `_check_closed()` to 27 public methods (token accessors, model info, state management, LoRA, performance metrics). Users now get clear `LlamaError` instead of confusing `AttributeError` when calling methods on closed instances
- **Missing 'mistral' config**: Added `MODEL_CONFIGS["mistral"]` entry to fix `KeyError` when Mistral models detected via GGUF metadata

**MEDIUM Priority** (5 issues fixed):
- **Backend free race**: Added `g_resource_mutex` lock in `backend_free()` to prevent race between check and free
- **Dead code cleanup**: Replaced `getattr(config, 'max_prompt_multiplier', 2)` with module constant `_MAX_PROMPT_MULTIPLIER` (slots dataclass can't have dynamic attributes)
- **Queue init race**: Fixed lazy initialization in `LlamaPool._ensure_queue_initialized()` to be safe for free-threaded Python 3.13+
- **Config key typos**: Fixed `MODEL_CONFIGS["glm4"]` → `"glm-4"` and `MODEL_CONFIGS["phi"]` → `"phi-4"`
- **Cleanup registration race**: Removed unprotected first check in `_register_unified_cleanup()` for Python 3.13+ safety

**LOW Priority** (2 improvements):
- **Loop variable clarity**: Changed `for const int i : priming` → `for const llama_token tok : priming` in C++ for correct type and clarity
- **Type safety**: Replaced `hasattr()` with `isinstance()` check in `UnifiedLLM.strip_thinking()`

**Documentation**: `docs/CODE_REVIEW_FIXES_2026-03-31_v2.md`

### Code Quality Refactoring

**Improvement #1: Stop-Sequence Validation Helper**
- Extracted duplicate validation logic from `generate_stream()` and `generate()`
- Added `_validate_stop_sequences()` static method (single source of truth)
- **Bug fix**: Added missing validation to `create_chat_completion()` (was completely unprotected)
- Impact: 18 lines reduced, consistent validation across all APIs

**Improvement #2: UTF-8 Streaming Helper**
- Extracted duplicate UTF-8 decoding from 3 locations: `generate_stream()`, `generate(stream=True)`, `create_chat_completion(stream=True)`
- Added `_token_to_text_incremental()` method with guaranteed final flush
- **Bug fix**: Ensures `decoder.decode(b"", final=True)` always called (was inconsistent)
- Impact: 32 lines reduced, consistent multi-byte UTF-8 handling (emoji, CJK)

**Total Impact**:
- 50 lines of duplication eliminated
- 2 bugs fixed (missing validation + UTF-8 flush)
- 130/130 tests passing (0 regressions)
- All changes backward compatible

**Documentation**: 
- `docs/IMPROVEMENT_SUGGESTIONS_ANALYSIS.md` - Analysis of 5 suggestions (2 implemented, 3 rejected with rationale)
- `docs/IMPROVEMENTS_2026-03-31_v2.md` - Implementation details with before/after code
