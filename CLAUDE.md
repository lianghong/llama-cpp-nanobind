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
black src/ tests/

# Lint
ruff check src/ tests/

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
   - `lib/`: CUDA-enabled llama.cpp shared libraries
   - `models/`: GGUF model files
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
   - `_preload_shared_libs()` in `__init__.py` ensures CUDA/ggml libraries load correctly
   - Works for both editable installs (from `./lib`) and wheel installs (`llama_cpp/lib`)

### Key Design Patterns

#### State Management
- **KV Cache Position**: `cur_pos_` tracks context position, updated by:
  - `load_state()`, `set_state_data()`: Sync from KV cache after load
  - `kv_cache_seq_rm()`, `kv_cache_seq_keep()`, `kv_cache_seq_add()`: Update when modifying sequence 0
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
  - Background worker thread runs C++ generation
  - Main thread yields tokens from queue as they arrive
  - Exceptions propagated from worker thread to caller
  - Early termination supported (daemon thread)
- **`generate(..., stream=True)`**: Buffered streaming
  - All tokens generated first, then yielded (higher latency)
  - Simpler implementation, no threading overhead
  - Suitable when latency not critical

#### Parallel Inference (LlamaPool)
- **Purpose**: True concurrent processing with multiple model instances
- **Architecture**:
  - Creates `pool_size` independent `Llama` instances
  - Round-robin load balancing across instances
  - Asyncio semaphore limits concurrent access to pool_size
  - Each instance processes one request at a time
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

1. **Always release GIL** for long operations: `nb::call_guard<nb::gil_scoped_release>()`
2. **Update `cur_pos_`** when modifying KV cache or loading state
3. **Reuse buffers** (like `single_batch_`) instead of per-call allocation
4. **Respect sampler chain** after grammar constraints

### When Modifying Python Wrappers (`src/llama_cpp/llama.py`, `unified.py`)

1. **LoRA persistence**: Always call `_reapply_lora_adapters()` after context reset
2. **Embeddings**: Validate `config.embeddings=True` before embedding operations
3. **Token counting**: Use `add_special=True` when calculating max_tokens for chat
4. **Stop sequences**: Use `generate_tokens_multi_stop()` when stop sequences present and details not needed

### Model File Requirement

Default test model: `./models/Qwen3-8B-Q6_K.gguf`

Update `conftest.py` if using different model paths.

## Performance Considerations

### Build Optimizations (Release)
- `-O3 -march=native -mtune=native -flto=auto -ffast-math -funroll-loops`
- LTO enabled if supported

### Runtime Optimizations
- GIL released during C++ operations (v0.3.0)
- Per-token batch allocation eliminated (v0.3.0)
- Fast stop sequence path (v0.3.0)
- Grammar sampler caching for repeated schemas (v0.3.1)
- Session-style continuation with `reset_kv_cache=False` (v0.3.1)
- True incremental streaming via background thread (v0.3.2)
  - Tokens yielded as generated, not buffered
  - Low time-to-first-token for responsive UIs
  - Perfect for SSE/WebSocket streaming endpoints

## Testing Strategy

Test files organized by concern:
- `test_inference.py`: Core generation, chat, embeddings, state management
- `test_async.py`: Async API correctness
- `test_optimizations.py`: Embedding context reuse, KV cache, multi-token stops
- `test_regressions.py`: State load position tracking, grammar sampling, LoRA persistence
- `test_unified.py`: UnifiedLLM multi-model support
- `test_pool.py`: LlamaPool parallel inference and model warmup
- `test_streaming.py`: True incremental streaming API (requires model)
- `test_streaming_logic.py`: Streaming threading logic (no model required)

Key test fixture: `conftest.py` provides `model_path` and `test_model` fixtures.

## Common Pitfalls

1. **Empty embeddings**: Ensure `LlamaConfig(embeddings=True)` when using `embed()` or `create_embedding()`
2. **Lost LoRA adapters**: After `reset()`, adapters are automatically reapplied
3. **Single-token stop sequences**: Use `generate_tokens_multi_stop()` for multi-token stops like `<|end_of_turn|>`
4. **Context overflow**: `UnifiedLLM` validates `max_tokens > 0` and raises on overflow
5. **Stale KV position**: State load/save automatically maintains `cur_pos_`
6. **Thread safety**: Do NOT call methods concurrently on same instance - use multiple instances or LlamaPool
7. **Global logging**: `verbose=False` affects ALL instances (llama.cpp limitation), triggers runtime warning

## Integration with llama.cpp

This project uses **prebuilt** llama.cpp libraries in `lib/`. When updating llama.cpp:

1. Build llama.cpp with CUDA support
2. Copy headers to `include/`
3. Copy shared libraries to `lib/`
4. Verify RPATH and soname compatibility
5. Update C++ bindings if API changed

## Model Support

UnifiedLLM auto-detects model families by filename patterns:
- Qwen3 (with thinking/non-thinking mode detection)
- Gemma
- Mistral
- GPT-OSS
- Phi
- GLM4
- MiniCPM

See `src/llama_cpp/unified.py` for family detection logic.
