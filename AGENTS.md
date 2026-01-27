# Repository Guidelines

## Project Structure & Module Organization
- `src/llama_cpp/` – Python package: `llama.py` (Llama class), `unified.py` (UnifiedLLM), and nanobind extension `_llama`.
- `src/bindings/` – C++ nanobind bindings mapping the llama.cpp API to Python.
- `include/` – shipped llama.cpp headers used during extension builds.
- `lib/` – precompiled shared libraries (`libllama.so`, `libggml*.so`, `libmtm*.so`) with CUDA enabled.
- `examples/` – runnable demos (e.g., `examples/basic.py`, `examples/unified_demo.py`).
- `tests/` – pytest smoke tests that load the bundled Qwen3-8B model.
- `models/` – expected test model `Qwen3-8B-Q6_K.gguf` (not tracked by git).
- `docs/` – API documentation (`API.md`).

## Key APIs
- `Llama` – main class for model loading and inference (supports context manager)
- `LlamaConfig` – configuration including `chat_format`, `embeddings`, GPU settings
- `SamplingParams` – temperature, top_k, top_p, penalties
- `LlamaGrammar` – constrained generation via GBNF or JSON schema
- `UnifiedLLM` – multi-model wrapper with auto-detection for Qwen3, Gemma, Mistral, GPT-OSS, Phi, etc.

**Exceptions:**
- `LlamaError` – base exception
- `ModelLoadError` – model loading failures
- `ValidationError` – invalid parameters
- `GenerationError` – generation failures

**Key Methods:**
- `generate()`, `generate_async()` – text generation
- `create_chat_completion()`, `create_chat_completion_async()` – chat API
- `create_embedding()`, `create_embedding_async()` – embeddings
- `model_size()`, `n_params()`, `n_layer()` – model info
- `n_tokens(text, add_special=False)` – count tokens
- `kv_cache_clear()` – clear KV cache
- `kv_cache_seq_rm()`, `kv_cache_seq_cp()`, `kv_cache_seq_pos_max()` – KV cache management
- `save_state()`, `load_state()`, `get_state()`, `set_state()` – state persistence
- `load_lora()`, `remove_lora()`, `clear_lora()` – LoRA adapter management
- `reset()` – reset context (reapplies LoRA adapters)
- `print_system_info()` – llama.cpp build/CPU info
- `shutdown()` – explicit cleanup of all instances before program exit

## Thread Safety
- Sync methods are NOT thread-safe; don't call concurrently on same instance
- Async methods serialize via internal lock
- GIL is released during heavy C++ operations (decode, generate, tokenize)
- `verbose=False` affects logging globally, not per-instance
- For parallelism, use multiple Llama/UnifiedLLM instances

## Build, Test, and Development Commands
- Create env: `uv venv --python 3.14 .venv && source .venv/bin/activate`
- Editable install: `uv pip install -e .`
- Run tests: `uv run pytest -q` or `make test`
- Build wheel: `make wheel`
- Run example: `uv run python examples/basic.py`

## Code Quality Tools
- Linting: `ruff check src/llama_cpp/ tests/`
- Formatting: `black src/llama_cpp/ tests/` and `isort src/llama_cpp/ tests/`
- Type checking: `mypy src/llama_cpp/ --ignore-missing-imports`
- Dead code: `vulture src/llama_cpp/ --min-confidence 80`

## Coding Style & Naming Conventions
- Python: PEP 8/257/484; type hints required for public APIs; use dataclasses for configs.
- C++: C++17, prefer RAII wrappers; avoid copying llama model/context; use explicit error handling.
- Indentation: 4 spaces (Python), 4 spaces (C++).
- Filenames: snake_case for Python, lower_snake for headers/sources; tests start with `test_`.

## Testing Guidelines
- Framework: pytest.
- Scope: keep tests fast; smoke tests only load the Qwen3-8B model.
- Shared fixtures in `tests/conftest.py` with proper cleanup.
- Add new tests under `tests/`, named `test_*.py`; ensure they skip gracefully if `models/` is missing.
- Run `uv run pytest -q` before sending PRs.

## Commit & Pull Request Guidelines
- Commits: concise present-tense subject (≤72 chars), e.g., `Add chat completion helper`.
- PRs: include summary of changes, testing performed, and relevant issue links.

## Security & Configuration Tips
- CUDA is assumed; if testing CPU-only, set `n_gpu_layers=0`.
- Do not vendor new models in git; keep large artifacts in `models/` locally.
