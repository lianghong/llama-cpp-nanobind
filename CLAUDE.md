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
# Format code
ruff format src/ tests/ examples/ tools/

# Sort imports
isort src/ tests/ examples/ tools/

# Lint
ruff check src/ tests/ examples/ tools/

# Type checking
mypy src/llama_cpp/
```

### Build Configuration

The project uses scikit-build-core with CMake. Key environment variables:

```bash
# Custom build type
CMAKE_BUILD_TYPE=RelWithDebInfo uv pip install -e .

# Custom lib/include paths
LLAMA_LIB_DIR=$(pwd)/lib LLAMA_INCLUDE_DIR=$(pwd)/include uv pip install -e .

# Portable build (no -march=native)
CMAKE_ARGS="-DLLAMA_PORTABLE=ON" uv pip install -e .
```

## Architecture

### Component Structure

1. **External Dependencies** (`lib/`, `include/`, `models/`)
   - **Not included in repo** - must be obtained separately
   - `include/`: llama.cpp headers for C++ bindings
   - `lib/`: Prebuilt llama.cpp shared libraries (`.so` on Linux/CUDA, `.dylib` on macOS/Metal)
   - `models/`: GGUF model files
   - On macOS, Homebrew llama.cpp is auto-detected by CMake
   - See README.md for setup instructions

2. **C++ Bindings** (`src/bindings/llama_cpp.cpp`)
   - Single-file nanobind extension module
   - Exposes llama.cpp C API to Python
   - **Critical**: GIL released during heavy C++ operations (decode, generate, tokenize)
   - Maintains internal state: `cur_pos_` for KV cache position tracking
   - Reuses `single_batch_` to eliminate per-token allocations

3. **Python Wrappers** (`src/llama_cpp/`)
   - `llama.py`: Core `Llama` class with high-level inference API
   - `unified.py`: `UnifiedLLM` class for multi-model support (auto-detects Qwen3, Gemma, Mistral, etc.)
   - `__init__.py`: Preloads shared libraries with `RTLD_GLOBAL` to avoid soname issues

4. **Library Preloading**
   - `_preload_shared_libs()` in `__init__.py` ensures CUDA/Metal/ggml libraries load correctly
   - Works for both editable installs (from `./lib`) and wheel installs (`llama_cpp/lib`)
   - RPATH: `$ORIGIN/lib` on Linux, `@loader_path/lib` on macOS

### Key Design Patterns

#### State Management
- **KV Cache Position**: `cur_pos_` tracks context position, updated by:
  - `load_state()`, `set_state_data()`: Sync from KV cache after load
  - `kv_cache_seq_rm()`, `kv_cache_seq_keep()`, `kv_cache_seq_add()`: Update when modifying sequence 0
- **State Save/Load**: Uses `nb::bytes` buffer protocol for zero-copy transfer between Python and C++ (no per-element conversion); GIL managed manually — released during heavy llama.cpp calls, held for Python object construction
- **LoRA Adapters**: Tracked in `_lora_configs` list, reapplied via `_reapply_lora_adapters()` after `reset()`

#### Sampling Pipeline
- Grammar constraints apply **before** sampler chain
- Sampler chain respects temperature/top_p/top_k even with grammar
- Use `cur_p.selected` from sampler, not argmax

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
  - **Multi-token stop sequence buffering**: tokens are buffered up to `max_stop_len` before yielding to prevent partial stop sequence tokens from reaching the consumer
- **`generate(..., stream=True)`**: Buffered streaming
  - All tokens generated first, then yielded (higher latency)
  - Simpler implementation, no threading overhead
  - Suitable when latency not critical

#### Memory Safety (Double-Free Prevention)
- **C++ classes**: All destructors check `if (ptr_)` before free, then set `ptr_ = nullptr`
- **Thread-safe close**: `Model::close()` and `Context::close()` hold `g_init_mutex` to prevent races between GC/`__del__` and explicit `close()`
- **Backend ref-counting**: `g_model_count` atomic prevents backend double-free
- **Nanobind `keep_alive`**: Context, SamplerChain, LoraAdapter keep Model alive via `nb::keep_alive<1, 2>()`
- **LoRA adapters**: Freed automatically by llama.cpp with the model (destructor is `= default`)
- **No `__del__`**: Neither `Llama` nor `UnifiedLLM` uses `__del__` (avoids GIL issues during shutdown); cleanup via atexit + RAII
- **`close()` idempotency**: `Llama._closed` flag and `UnifiedLLM`'s `self.llm is None` check
- **Destruction order**: Context freed before Model (C++ dependency)
- **Instance tracking**: `weakref.ref` sets prevent circular references; atexit handler calls `close()` on all live instances

#### Parallel Inference (LlamaPool)
- **Purpose**: True concurrent processing with multiple model instances
- **Architecture**:
  - Creates `pool_size` independent `Llama` instances
  - `asyncio.Queue`-based checkout/return ensures exclusive instance access (Llama is not thread-safe)
  - Each instance is checked out by at most one request at a time
  - Instances returned to pool after use via try/finally
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

1. **Always release GIL** for long operations: `nb::call_guard<nb::gil_scoped_release>()` or manual `nb::gil_scoped_release`
2. **Update `cur_pos_`** when modifying KV cache or loading state
3. **Reuse buffers** (like `single_batch_`) instead of per-call allocation
4. **Use RAII for temporary resources**: `Context::decode` uses `BatchGuard` struct for `llama_batch` cleanup instead of try/catch
5. **Respect sampler chain** after grammar constraints
6. **Hold `g_init_mutex`** when freeing resources in `close()` methods (prevents races with GC)
7. **Validate token range before indexing logits**: sampler can return `LLAMA_TOKEN_NULL` (-1); always check `token >= 0 && token < n_vocab` before `logits[token]`
8. **Use `nb::bytes` for binary data**: state save/load uses `nb::bytes` directly (not `std::vector<uint8_t>`) to avoid per-element Python↔C++ conversion; manage GIL manually when mixing Python object construction with heavy C++ calls

### When Modifying Python Wrappers (`src/llama_cpp/llama.py`, `unified.py`)

1. **LoRA persistence**: Always call `_reapply_lora_adapters()` after context reset
2. **Embeddings**: Validate `config.embeddings=True` before embedding operations
3. **Token counting**: Use `add_special=True` when calculating max_tokens for chat
4. **Stop sequences**: Use `generate_tokens_multi_stop()` when stop sequences present and details not needed
5. **Grammar samplers are stateful**: Never cache or reuse them — `llama_sampler_accept` mutates internal state, so each generation must create a fresh sampler
6. **Falsy-value defaults**: Use `if x is not None else fallback` (not `x or fallback`) when `0` or `0.0` is a valid value

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
5. **Stale KV position**: State load/save automatically maintains `cur_pos_`
6. **Thread safety**: Do NOT call methods concurrently on same instance - use multiple instances or LlamaPool
7. **Global logging**: `verbose=False` affects ALL instances (llama.cpp limitation)
8. **Grammar sampler reuse**: Never cache grammar samplers — they are stateful and must be created fresh each generation
9. **Falsy-value traps**: Use `is not None` checks (not `or`) when `0`/`0.0` are valid parameter values
10. **Logprobs token bounds**: In the logprobs/details path (`generate_tokens_with_details`), always validate token range before indexing `logits[]` — sampler can return `LLAMA_TOKEN_NULL` (-1)
11. **Translation hallucination**: LLMs may editorialize, reverse sentiment, or inject opinions when translating opinionated/sarcastic text. Mitigate with: low temperature (0.1–0.3), few-shot examples demonstrating faithful translation, and explicit system prompt rules against inserting commentary
12. **14B+ models on 16GB Apple Silicon**: Expect ~10 tok/s generation (memory-bandwidth-limited). Default context 10240 may crash Metal; use `--ctx 4096`. Performance is near hardware ceiling — use 8B or 4B models for better throughput

## Integration with llama.cpp

This project uses **prebuilt** llama.cpp libraries in `lib/`. When updating llama.cpp:

1. Build llama.cpp with GPU support (`-DGGML_CUDA=ON` on Linux, `-DGGML_METAL=ON` on macOS)
2. Copy headers to `include/`
3. Copy shared libraries to `lib/` (`.so` on Linux, `.dylib` on macOS)
4. Verify RPATH and soname compatibility
5. Update C++ bindings if API changed

On macOS, `brew install llama.cpp` provides headers and libraries that CMake auto-detects.

## Model Support

UnifiedLLM auto-detects model families by filename patterns:
- Aya (Cohere's multilingual model, Tiny Aya variants, 70+ languages)
- Qwen3 (with thinking/non-thinking mode detection)
- Qwen3.5 (hybrid attention, 262K context, thinking mode default-on)
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

English-to-Chinese translation script using `UnifiedLLM` with optimized settings:

```bash
python examples/translate.py --model models/Qwen3-8B-Q6_K.gguf --ctx 8192
python examples/translate.py --model models/Qwen3-8B-Q6_K.gguf --thinking
python examples/translate.py --model models/Qwen3-8B-Q6_K.gguf --temperature 0.1 -o
```

Key design decisions:
- **Low default temperature (0.3)**: Reduces hallucination and sentiment drift in translations
- **Few-shot example in system prompt**: Demonstrates faithful translation of editorially charged/sarcastic text, preventing the model from editorializing or reversing the author's sentiment
- **`--temperature` CLI arg**: Overrides the model's default temperature after construction via `llm.model_config.temperature`
- **VRAM check**: Estimates GPU memory usage before loading and warns if insufficient
