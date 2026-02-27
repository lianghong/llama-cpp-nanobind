# llama-cpp-nanobind

High-performance nanobind bindings for `llama.cpp`, packaged as a wheel-ready Python library with CUDA enabled by default. The interface mirrors `llama-cpp-python` where it does not conflict with upstream `llama.cpp`.

## Platform Support

- **Linux** (x86_64): NVIDIA CUDA GPU (compute capability 6.0+, CUDA 13.x)
- **macOS** (Apple Silicon & Intel): Metal GPU acceleration via Homebrew llama.cpp
- **Python**: 3.14+

## Project layout

- `src/llama_cpp/llama.py` – Pythonic wrapper (`Llama` class)
- `src/llama_cpp/unified.py` – Multi-model wrapper (`UnifiedLLM` class)
- `src/bindings/` – nanobind extension (C++)
- `examples/` – runnable scripts
- `tests/` – pytest-based smoke tests

**External dependencies (not included, see setup below):**
- `include/` – headers from llama.cpp
- `lib/` – precompiled shared libraries
- `models/` – GGUF model files

## Prerequisites

- Python 3.14+
- `uv` package manager (https://github.com/astral-sh/uv)
- llama.cpp headers and libraries (see setup below)

**Linux:**
- GCC/G++ 15 at `/usr/local/bin/gcc-15` and `/usr/local/bin/g++-15`
- CUDA-capable GPU

**macOS:**
- Xcode Command Line Tools (`xcode-select --install`)
- Homebrew llama.cpp (`brew install llama.cpp`) or manually built libraries

## External Dependencies Setup

Before building, you need to obtain the llama.cpp headers and libraries:

### Option 1: Homebrew (macOS)

```bash
brew install llama.cpp
# CMake auto-detects Homebrew paths — no manual copying needed
```

### Option 2: Build llama.cpp from source

```bash
# Clone and build llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Linux (CUDA)
cmake -B build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON
cmake --build build --config Release
cp -r include/ /path/to/llama-cpp-nanobind/include/
cp build/lib*.so /path/to/llama-cpp-nanobind/lib/

# macOS (Metal)
cmake -B build -DGGML_METAL=ON -DBUILD_SHARED_LIBS=ON
cmake --build build --config Release
cp -r include/ /path/to/llama-cpp-nanobind/include/
cp build/lib*.dylib /path/to/llama-cpp-nanobind/lib/
```

### Option 3: Use prebuilt release

Download prebuilt libraries from [llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases) and extract to `include/` and `lib/`.

### Model files

Download GGUF models from [Hugging Face](https://huggingface.co/models?search=gguf) and place in `models/`:

```bash
# Example: Download Qwen3-8B
huggingface-cli download Qwen/Qwen3-8B-GGUF Qwen3-8B-Q6_K.gguf --local-dir models/
```

## Build & install

```bash
# Create virtual environment with uv
uv venv --python 3.14 .venv
source .venv/bin/activate

# Install build deps and the package (editable)
uv pip install -e .
```

`scikit-build-core` drives the build; it automatically links against the prebuilt libraries in `./lib` and installs them into the wheel. RPATH is set so the extension finds `llama_cpp/lib` at runtime.

**Note**: Release builds use aggressive optimizations (`-O3`, `-march=native`, `-flto=auto`, `-ffast-math`) for maximum performance.

**Recent Optimizations**: v0.3.0 includes significant performance and correctness improvements:
- GIL released during heavy C++ operations for better async/threading performance
- State load/save correctly maintains KV cache position bookkeeping
- Grammar-constrained generation now respects sampling parameters (temperature, top_p, etc.)
- Stop sequences use fast C++ path (no O(n_vocab) overhead per token)
- Per-token batch allocations eliminated via reusable buffer
- LoRA adapters persist correctly across `reset()` calls
- `UnifiedLLM.kv_cache_clear()` now works correctly

**v0.3.1 Optimizations:**
- True incremental streaming via `generate_stream()` - yields tokens as generated in background thread
- Session-style continuation with `reset_kv_cache=False` to reduce recompute
- Backend shutdown guard prevents race conditions
- `n_seq_max` now configurable in `LlamaConfig`

**v0.3.2 Streaming Improvements:**
- `generate_stream()` now uses background thread for genuine token-by-token streaming
- Tokens are yielded immediately as generated, not buffered (low latency)
- Early termination supported without waiting for completion
- Perfect for SSE endpoints, WebSocket streaming, and responsive UIs
- Optional model warmup for `LlamaPool` to eliminate cold-start latency

**v0.3.3 Stability & Documentation:**
- Fixed race condition in global state initialization (thread safety)
- Improved thread safety documentation with prominent warnings
- Enhanced error handling and state synchronization

**v0.3.4 Cleanup:**
- Removed redundant internal documentation (AGENTS.md, CMAKE_OPTIMIZATIONS.md, OPTIMIZATIONS.md)
- Synchronized version across all project files

**v0.3.5 Bug Fixes, Python 3.14, TranslateGemma:**
- Fixed grammar sampler cache bug — stateful samplers were reused across generations producing incorrect results; cache removed, fresh samplers created each time
- Fixed `generate_stream()` thread cancellation — background thread now stops promptly via `threading.Event` when consumer closes generator early
- Fixed GIL management in streaming — GIL released during C++ decode/sample operations, re-acquired only for Python callback
- Fixed `Model::close()` and `Context::close()` thread safety — both now hold mutex to prevent races with GC/`__del__`
- Fixed falsy-value bug in `UnifiedLLM` thinking parameter defaults (`0.0` was treated as `None`)
- Removed `UnifiedLLM.__del__` — cleanup handled by atexit handler, consistent with `Llama` design
- Upgraded to Python 3.14 — removed `from __future__ import annotations` (PEP 649), `uuid7()`, PEP 758 bracketless except
- Added TranslateGemma model support (Google's 55-language translation model based on Gemma 3)
- Added `examples/verify_double_free.py` — 20-scenario memory safety verification script
- Added `tools/` — `url2md.py` (web-to-markdown) and `md_translator.py` (LLM-powered markdown translation)
- Code quality: all ruff, ruff format, isort, and mypy issues resolved across codebase

**Post-v0.3.5 Fixes & Optimizations:**
- Fixed out-of-bounds read in logprobs path — `generate_tokens_with_details()` now validates token range before indexing logits (prevents UB when sampler returns `LLAMA_TOKEN_NULL`)
- Fixed streaming stop sequence leakage — `generate_tokens_streaming()` now buffers tokens up to the longest stop sequence length before yielding, preventing partial stop tokens from reaching the consumer
- Fixed `LlamaPool` concurrent instance safety — replaced round-robin + semaphore with `asyncio.Queue` checkout/return to guarantee exclusive instance access (Llama is not thread-safe)
- State save/load uses `nb::bytes` buffer protocol — eliminates per-element Python↔C++ conversion for `get_state_data()`/`set_state_data()`, reducing memory overhead from ~28x to ~1x for large KV cache states
- RAII `BatchGuard` in `Context::decode` replaces manual try/catch for `llama_batch` cleanup
- O(n_vocab) candidate vector allocated once per generation call instead of per token in grammar/logprobs paths
- Added logprobs test coverage (structure validation, short prompts, stop sequence interaction)
- Added `examples/translate.py` — English-to-Chinese translation example with few-shot prompting to prevent hallucination/sentiment reversal, configurable `--temperature` (default 0.3 for faithful output)
- Fixed mypy `no-any-return` in `Llama.get_state()` — explicit type annotation for C++ binding return value

### Optional build flags

```bash
# Custom build type or different lib/include roots
CMAKE_BUILD_TYPE=RelWithDebInfo \
LLAMA_LIB_DIR=$(pwd)/lib \
LLAMA_INCLUDE_DIR=$(pwd)/include \
uv pip install -e .

# Portable build without -march=native (for distributable wheels)
CMAKE_ARGS="-DLLAMA_PORTABLE=ON" uv pip install -e .
```

## Usage

```python
from llama_cpp import Llama, SamplingParams, LlamaConfig

# Basic usage
llm = Llama("models/Qwen3-8B-Q6_K.gguf")
text = llm.generate("Hello, world!", max_tokens=64)
print(text)

# Context manager for automatic cleanup
with Llama("models/Qwen3-8B-Q6_K.gguf") as llm:
    text = llm.generate("Hello", max_tokens=32)
    print(text)

# Custom sampling
sampling = SamplingParams(temperature=0.7, top_p=0.9, repeat_penalty=1.05)
stream = llm.generate("Tell me a haiku", max_tokens=48, sampling=sampling, stream=True)
for chunk in stream:
    print(chunk, end="", flush=True)

# True incremental streaming (yields tokens as generated - LOW LATENCY)
# Tokens arrive immediately, perfect for SSE/WebSocket/live UIs
for chunk in llm.generate_stream("Tell me a story", max_tokens=100):
    print(chunk, end="", flush=True)
# Note: generate(..., stream=True) buffers all tokens first (higher latency)

# OpenAI-compatible chat endpoint
chat = llm.create_chat_completion(
    [{"role": "user", "content": "Give me one word describing the ocean"}],
    max_tokens=8,
)
print(chat["choices"][0]["message"]["content"])

# Session-style continuation (reuse KV cache)
llm.generate("Hello", max_tokens=10, reset_kv_cache=True)
llm.generate("Continue", max_tokens=10, reset_kv_cache=False)  # Faster
```

### UnifiedLLM (Multi-Model Support)

For working with multiple model families (Qwen3, Qwen3.5, Gemma, TranslateGemma, Aya, GLM-4/4.7, Mistral, GPT-OSS, Phi, etc.):

```python
from llama_cpp.unified import UnifiedLLM

# Auto-detects model family from path
llm = UnifiedLLM("models/Qwen3-30B-A3B-Instruct-2507-Q4_K_S.gguf")
print(f"Model family: {llm.family.name}")

# Basic generation (Instruct-2507 is non-thinking by default)
response = llm.generate("Explain quantum computing briefly")
print(response)

# Enable thinking mode for Qwen3 (hybrid models only, not Instruct-2507)
# llm = UnifiedLLM("models/Qwen3-8B-Q6_K.gguf")
# response = llm.generate("Solve: x^2 - 4 = 0", thinking=True)
```

### Translation Example

`examples/translate.py` provides English-to-Chinese translation with optimized settings for faithful output:

```bash
# Basic translation (default temperature=0.3 for accuracy)
python examples/translate.py --model models/Qwen3-8B-Q6_K.gguf

# With thinking mode for complex text
python examples/translate.py --model models/Qwen3-8B-Q6_K.gguf --thinking

# Custom temperature and output file
python examples/translate.py --model models/Qwen3-8B-Q6_K.gguf --temperature 0.1 -o

# Custom input file
python examples/translate.py --model models/Qwen3-8B-Q6_K.gguf --file input.txt --ctx 8192
```

The system prompt uses few-shot examples to prevent common LLM translation failures (hallucinated content, sentiment reversal, editorializing).

### Error Handling

Custom exceptions for better error handling:

```python
from llama_cpp import Llama, LlamaError, ModelLoadError, ValidationError

try:
    llm = Llama("nonexistent.gguf")
except ModelLoadError as e:
    print(f"Failed to load model: {e}")

try:
    llm.generate("test", max_tokens=0)  # Invalid
except ValidationError as e:
    print(f"Invalid parameter: {e}")
```

### Async API

Async wrappers for FastAPI, asyncio applications (runs inference in thread pool):

```python
import asyncio
from llama_cpp import Llama

async def main():
    llm = Llama("models/Qwen3-8B-Q6_K.gguf")
    
    # Async generation
    text = await llm.generate_async("Hello", max_tokens=32)
    
    # Async streaming
    async for chunk in await llm.generate_async("Test", max_tokens=16, stream=True):
        print(chunk, end="", flush=True)
    
    # Async chat completion
    response = await llm.create_chat_completion_async(
        [{"role": "user", "content": "Hi"}],
        max_tokens=16
    )

asyncio.run(main())
```

**⚠️ Thread Safety Warning:**
- The `Llama` class is **NOT thread-safe** - do not call methods concurrently from multiple threads on the same instance
- Async methods use a lock to prevent crashes, but concurrent calls serialize (no parallelism benefit)
- For true parallel inference, use `LlamaPool` with multiple independent instances
- `verbose=False` in `LlamaConfig` affects logging **globally** for all instances (llama.cpp limitation; see class docstring for details)

### Parallel Inference with LlamaPool

For true concurrent processing across multiple requests, use `LlamaPool`. This creates multiple independent Llama instances that can process requests in parallel:

```python
from llama_cpp import LlamaPool
import asyncio

async def main():
    # Create pool with 4 parallel workers
    async with LlamaPool("model.gguf", pool_size=4) as pool:
        # These run in TRUE PARALLEL (not serialized)
        results = await pool.generate_batch([
            "What is artificial intelligence?",
            "Explain quantum computing",
            "Tell me about Python",
            "What is machine learning?"
        ], max_tokens=64)

        for i, result in enumerate(results, 1):
            print(f"Result {i}: {result}")

asyncio.run(main())
```

**Performance Comparison:**

```python
# Single instance (serialized)
llm = Llama("model.gguf")
results = await asyncio.gather(
    llm.generate_async("Q1"),  # ← Runs first
    llm.generate_async("Q2"),  # ← Waits for Q1
    llm.generate_async("Q3"),  # ← Waits for Q2
)
# Total time: ~3x single query time

# Pool (parallel)
pool = LlamaPool("model.gguf", pool_size=3)
results = await asyncio.gather(
    pool.generate("Q1"),  # ← Runs
    pool.generate("Q2"),  # ← Runs in parallel
    pool.generate("Q3"),  # ← Runs in parallel
)
# Total time: ~1x single query time (3x speedup!)
pool.close()
```

**GPU Memory Planning:**
- Each instance loads the full model separately
- Required VRAM ≈ `model_size × pool_size`
- Example: 8GB model with `pool_size=3` needs ~24GB VRAM
- Adjust `pool_size` based on available GPU memory

**Model Warmup (Optional):**
```python
# Enable warmup for production deployments with strict SLA requirements
async with LlamaPool("model.gguf", pool_size=4, warmup=True) as pool:
    # All instances are pre-warmed, first request has consistent latency
    results = await pool.generate_batch([...], max_tokens=64)
```

Warmup benefits:
- ✓ Eliminates cold-start latency variability on first request
- ✓ Pre-loads GPU caches and compiles CUDA kernels
- ✓ Ensures predictable performance for production SLAs

Warmup tradeoffs:
- ✗ Adds 1-3 seconds to pool initialization time
- ✗ May not provide significant benefit for llama.cpp (overhead typically <50ms)
- ℹ️ Recommended only for services with strict latency requirements

**Chat Completions:**

```python
async with LlamaPool("model.gguf", pool_size=2) as pool:
    conversations = [
        [{"role": "user", "content": "Hello!"}],
        [{"role": "user", "content": "Hi there!"}],
    ]
    responses = await pool.create_chat_completion_batch(
        conversations, max_tokens=32
    )
```

See `examples/parallel_inference.py` for a complete demonstration with benchmarks.

### Chat Templates

Use `chat_format` to apply model-specific chat templates:

```python
config = LlamaConfig(
    model_path="models/Qwen3-8B-Q6_K.gguf",
    chat_format="gemma"  # Uses llama.cpp built-in template
)
llm = Llama("models/Qwen3-8B-Q6_K.gguf", config=config)
```

### Embeddings

```python
# Simple embedding
vec = llm.embed("embedding me softly")

# OpenAI-compatible embedding API (requires embeddings=True)
config = LlamaConfig(model_path="model.gguf", embeddings=True)
llm = Llama("model.gguf", config=config)
result = llm.create_embedding("Hello world")
```

### JSON Mode / Constrained Generation

```python
# Force valid JSON output
response = llm.create_chat_completion(
    [{"role": "user", "content": "Return JSON with name and age"}],
    max_tokens=32,
    response_format={"type": "json_object"}
)

# With JSON schema constraint
response = llm.create_chat_completion(
    [{"role": "user", "content": "Generate user data"}],
    max_tokens=32,
    response_format={
        "type": "json_object",
        "schema": {"type": "object", "properties": {"name": {"type": "string"}}}
    }
)
```

### Custom Grammar

```python
from llama_cpp import LlamaGrammar

grammar = LlamaGrammar.from_string('root ::= "yes" | "no"')
response = llm.create_chat_completion(messages=[...], grammar=grammar)
```

## Tests

```bash
uv pip install -e .[test]
uv run pytest -q
```

### Memory Safety Verification

The project includes a double-free verification script that exercises all resource cleanup paths under glibc's heap checker:

```bash
# Basic run (crash detection via signal handlers)
python examples/verify_double_free.py

# With glibc heap checking (detects silent corruption)
MALLOC_CHECK_=3 python examples/verify_double_free.py
```

The script tests 20 scenarios across both `Llama` and `UnifiedLLM`: double `close()`, context manager + close, state save/load + close, GC pressure, use-after-close, multi-instance close ordering, rapid create-close loops, `del` without close, `__del__` + close interactions, and mixed instance types.

## Notes

- Server-specific llama.cpp features are intentionally excluded; the bindings focus on efficient local inference.
- CUDA offload is enabled by default (`n_gpu_layers=-1`, `offload_kqv=True`). Adjust `LlamaConfig` for CPU-only operation.
- The sampler pipeline mirrors llama.cpp's sampler chain, keeping sampling inside C++ for speed.

## API surface

**Core Classes:**
- `Llama` – main class for model loading and inference (supports context manager)
- `LlamaPool` – pool manager for parallel inference with multiple instances
- `LlamaConfig` – configuration (chat_format, embeddings, GPU settings, n_seq_max)
- `SamplingParams` – temperature, top_k, top_p, penalties
- `LlamaGrammar` – constrained generation via GBNF or JSON schema
- `UnifiedLLM` – multi-model wrapper with auto-detection

**Exceptions:**
- `LlamaError` – base exception
- `ModelLoadError` – model loading failures
- `ValidationError` – invalid parameters
- `GenerationError` – generation failures

**Key Methods:**
- `generate()`, `generate_async()` – text generation
- `generate_stream()` – true streaming generation (yields as tokens decode)
- `create_chat_completion()`, `create_chat_completion_async()` – chat API
- `create_embedding()`, `create_embedding_async()` – embeddings
- `tokenize()`, `detokenize()`, `n_tokens()` – tokenization
- `save_state()`, `load_state()`, `get_state()`, `set_state()` – state management
- `load_lora()`, `remove_lora()`, `clear_lora()` – LoRA adapters
- `perf()`, `perf_reset()` – performance metrics

**Utilities:**
- `print_system_info()` – llama.cpp build/CPU info
- `set_log_level()`, `disable_logging()`, `reset_logging()` – logging control
- `shutdown()` – explicit cleanup of all instances before program exit

## License

This project is licensed under the MIT License.

This package includes prebuilt libraries from [llama.cpp](https://github.com/ggerganov/llama.cpp), which is also MIT licensed. See the llama.cpp repository for full license details and attribution requirements.
