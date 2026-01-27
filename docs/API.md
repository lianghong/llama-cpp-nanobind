# API Reference

## llama_cpp.Llama

High-level client wrapping the nanobind extension. Compatible with the `llama_cpp_python.Llama` surface where feasible. Supports context manager protocol for automatic resource cleanup.

**Constructor**

```python
Llama(model_path: str, config: LlamaConfig | None = None, sampling: SamplingParams | None = None)
```

- `model_path`: Path to a GGUF model file.
- `config`: Optional `LlamaConfig` to fine-tune context/model options.
- `sampling`: Default `SamplingParams` used for generation when no override is passed.

**Context Manager**

```python
with Llama("model.gguf") as llm:
    text = llm.generate("Hello", max_tokens=32)
# Resources automatically cleaned up
```

**Methods**

- `generate(prompt, max_tokens=128, sampling=None, stop=None, echo=False, logprobs=None, stream=False, seed=None, reset_kv_cache=True)` → `str | Iterator[str] | dict`
- `generate_stream(prompt, max_tokens=128, sampling=None, stop=None, seed=None, reset_kv_cache=True)` → `Iterator[str]` – True streaming (yields as tokens decode)
- `generate_async(...)` → Async version of generate
- `create_chat_completion(messages, max_tokens=128, stream=False, stop=None, response_format=None, grammar=None, tools=None, tool_choice=None, reset_kv_cache=True, **kwargs)` → Chat completion dict or stream
- `create_chat_completion_async(...)` → Async version
- `create_embedding(input, model=None)` → OpenAI-compatible embedding API (requires `embeddings=True`)
- `create_embedding_async(...)` → Async version
- `embed(text)` → `List[float]` – Simple embedding (requires `embeddings=True`)
- `embed_async(text)` → Async version
- `tokenize(text, add_special=True, parse_special=False)` → `List[int]`
- `detokenize(tokens, remove_special=True, unparse_special=False)` → `str`
- `n_tokens(text, add_special=False)` → `int` – Count tokens (set add_special=True to include BOS)
- `token_bos()`, `token_eos()`, `token_eot()` → Token IDs
- `n_ctx()`, `n_vocab()`, `n_embd()` → Model dimensions
- `model_size()`, `n_params()`, `n_layer()` → Model info
- `metadata` → `dict` – Model metadata (property)
- `get_chat_template(name="")` → `str`
- `token_to_piece(token)` → `str`
- `reset()` → Reset context/KV cache
- `close()` → Release resources (Context freed before Model for safety)
- `save_state(path)`, `load_state(path)` → State persistence
- `get_state()`, `set_state(data)` → State as bytes
- `load_lora(path, scale=1.0)`, `remove_lora(adapter)`, `clear_lora()` → LoRA management
- `perf()`, `perf_reset()` → Performance metrics
- `kv_cache_clear()` → Clear KV cache without recreating context
- `kv_cache_seq_rm()`, `kv_cache_seq_cp()`, `kv_cache_seq_keep()`, `kv_cache_seq_add()`, `kv_cache_seq_pos_max()` → KV cache management

### True Streaming

Use `generate_stream()` for true incremental streaming (yields tokens as they're decoded):

```python
for chunk in llm.generate_stream("Tell me a story", max_tokens=100):
    print(chunk, end="", flush=True)
```

### Session-Style Continuation

Use `reset_kv_cache=False` to preserve KV cache between calls for multi-turn sessions:

```python
# First turn
llm.generate("Hello", max_tokens=10, reset_kv_cache=True)

# Continue without clearing cache (faster, reuses computed KV)
llm.generate("How are you?", max_tokens=10, reset_kv_cache=False)
```

## Exceptions

```python
from llama_cpp import LlamaError, ModelLoadError, ValidationError, GenerationError
```

- `LlamaError` – Base exception for all llama-cpp-nanobind errors
- `ModelLoadError` – Failed to load model file
- `ValidationError` – Invalid input parameters
- `GenerationError` – Text generation failed

**Example:**

```python
from llama_cpp import Llama, ModelLoadError, ValidationError

try:
    llm = Llama("nonexistent.gguf")
except ModelLoadError as e:
    print(f"Model error: {e}")

try:
    llm.generate("test", max_tokens=0)
except ValidationError as e:
    print(f"Validation error: {e}")
```

## Module-level functions

- `print_system_info()` → `str` – llama.cpp system info
- `set_log_level(level)` → None – Set log level
- `disable_logging()` → None – Silence logging
- `reset_logging()` → None – Restore default logging
- `shutdown()` → None – Explicitly shutdown all Llama instances and free backend resources

### shutdown()

Call at the end of your program before `exit()` to avoid segfaults when using logging or other modules that hold references during cleanup.

```python
from llama_cpp import Llama, shutdown

def main():
    with Llama("model.gguf") as llm:
        text = llm.generate("Hello", max_tokens=32)
    shutdown()  # Clean up before Python's shutdown sequence

if __name__ == "__main__":
    main()
```

## Configuration dataclasses

### LlamaConfig

| Field | Default | Notes |
| --- | --- | --- |
| `model_path` | required | GGUF model path |
| `n_ctx` | 4096 | Context window (must be ≥ 1) |
| `n_batch` | 2048 | Logical batch (must be ≥ 1) |
| `n_ubatch` | `n_batch` | Physical micro-batch |
| `n_seq_max` | 1 | Max parallel sequences (1 = single sequence) |
| `n_threads` | `os.cpu_count()` | Threads for generation |
| `n_threads_batch` | `n_threads` | Threads for prompt/batch |
| `n_gpu_layers` | -1 | GPU layers (-1 = all) |
| `main_gpu` | 0 | Primary GPU index |
| `split_mode` | 0 | `llama_split_mode` enum |
| `use_mmap` | True | Memory-map model |
| `use_mlock` | False | mlock model into RAM |
| `offload_kqv` | True | Offload K/Q/V to GPU |
| `flash_attn` | 1 | Flash-attention mode |
| `embeddings` | False | Enable embeddings (required for `embed()` and `create_embedding()`) |
| `add_bos` | True | Add BOS during tokenization |
| `parse_special` | False | Parse special tokens |
| `chat_format` | None | Chat template name |
| `verbose` | True | Control logging |
| `seed` | -1 | RNG seed (-1 = random) |

Raises `ValidationError` if `n_ctx < 1`, `n_batch < 1`, `n_seq_max < 1`, or `n_gpu_layers < -1`.

### SamplingParams

| Field | Default |
| --- | --- |
| `temperature` | 0.8 |
| `top_k` | 40 |
| `top_p` | 0.95 |
| `min_p` | 0.0 |
| `min_keep` | 1 |
| `repeat_penalty` | 1.1 |
| `repeat_last_n` | 64 |
| `presence_penalty` | 0.0 |
| `frequency_penalty` | 0.0 |
| `seed` | None |

## llama_cpp.LlamaGrammar

Grammar for constrained text generation.

```python
from llama_cpp import LlamaGrammar

# From GBNF string
grammar = LlamaGrammar.from_string('root ::= "yes" | "no"')

# From JSON schema
grammar = LlamaGrammar.from_json_schema({"type": "object", "properties": {"name": {"type": "string"}}})

response = llm.create_chat_completion(messages=[...], grammar=grammar)
```

## llama_cpp.unified.UnifiedLLM

Unified interface for multiple LLM families with automatic detection and family-specific optimizations.

**Constructor**

```python
UnifiedLLM(
    model_path: str,
    n_ctx: int = 8192,
    n_batch: int = 2048,
    n_ubatch: int = 512,
    n_gpu_layers: int = -1,
    verbose: bool = False,
    family: str | ModelFamily | None = None
)
```

- `model_path`: Path to GGUF model file.
- `n_ctx`: Context size (clamped to model's max).
- `n_batch`: Batch size for prompt processing.
- `n_ubatch`: Micro-batch size.
- `n_gpu_layers`: Layers to offload to GPU (-1 = all).
- `verbose`: Enable verbose logging.
- `family`: Explicit model family override (auto-detects if None).

**Context Manager**

```python
with UnifiedLLM("models/Qwen3-30B-A3B-Instruct-2507-Q4_K_S.gguf") as llm:
    response = llm.generate("Hello")
# Resources automatically cleaned up
```

**Properties**

- `family` → `ModelFamily` – Detected model family enum
- `supports_thinking` → `bool` – Whether model supports thinking mode

**Methods**

- `generate(prompt, system_prompt=None, max_tokens=None, thinking=False, stop=None)` → `str`
- `generate_with_thinking(prompt, system_prompt=None, max_tokens=None, stop=None)` → `tuple[str, str]` – Returns (thinking, answer)
- `set_reasoning_level(level)` → None – Set GPT-OSS reasoning ("low", "medium", "high")
- `strip_thinking(text)` → `str` – Remove thinking tags from text
- `n_tokens(text)` → `int` – Count tokens
- `n_ctx()` → `int` – Get context size
- `kv_cache_clear()` → None – Clear KV cache
- `close()` → None – Release resources

**Example**

```python
from llama_cpp.unified import UnifiedLLM

# Qwen3-Instruct-2507 (non-thinking variant)
llm = UnifiedLLM("models/Qwen3-30B-A3B-Instruct-2507-Q4_K_S.gguf")
print(llm.family.name)  # QWEN3
print(llm.supports_thinking)  # False (Instruct-2507 is non-thinking)

# Basic generation
response = llm.generate("Hello")

# For thinking mode, use hybrid Qwen3 models:
# llm = UnifiedLLM("models/Qwen3-8B-Q6_K.gguf")
# response = llm.generate("Solve x^2 = 4", thinking=True)
# thinking, answer = llm.generate_with_thinking("Explain gravity")
```

### ModelFamily

Enum of supported model families:

- `AYA`, `GEMMA`, `GLM4`, `GRANITE`, `MINICPM`, `PHI`, `MISTRAL`, `QWEN3`, `GPT_OSS`

### ModelConfig

Model-specific configuration (auto-selected based on family):

| Field | Description |
| --- | --- |
| `family` | ModelFamily enum |
| `chat_format` | llama.cpp chat format name |
| `temperature` | Default sampling temperature |
| `top_p` | Default nucleus sampling |
| `top_k` | Default top-k sampling |
| `min_p` | Default min-p threshold |
| `max_ctx` | Maximum context length |
| `supports_thinking` | Thinking mode support |
| `stop_sequences` | Default stop sequences |

### detect_model_family

```python
from llama_cpp.unified import detect_model_family

config = detect_model_family("models/Qwen3-30B-A3B-Instruct-2507-Q4_K_S.gguf")
print(config.family)  # ModelFamily.QWEN3
print(config.supports_thinking)  # False (Instruct-2507 variant)
```

Auto-detects model family from file path. Raises `ValueError` if unknown.

## JSON Mode

```python
response = llm.create_chat_completion(
    messages=[{"role": "user", "content": "Return JSON"}],
    response_format={"type": "json_object"}
)

# With schema
response = llm.create_chat_completion(
    messages=[...],
    response_format={
        "type": "json_object",
        "schema": {"type": "object", "properties": {"name": {"type": "string"}}}
    }
)
```

## Function Calling

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
    }
}]

response = llm.create_chat_completion(
    messages=[{"role": "user", "content": "Weather in Tokyo?"}],
    tools=tools,
    tool_choice="auto"
)
```

## LoRA Adapters

```python
adapter = llm.load_lora("adapter.gguf", scale=1.0)
response = llm.generate("Hello", max_tokens=32)
llm.remove_lora(adapter)  # or llm.clear_lora()
```

## Resource Management

Both `Llama` and `UnifiedLLM` support proper resource cleanup:

**Context Manager (Recommended)**

```python
# Resources automatically released on exit
with Llama("model.gguf") as llm:
    text = llm.generate("Hello", max_tokens=32)

with UnifiedLLM("model.gguf") as llm:
    text = llm.generate("Hello")
```

**Explicit Close**

```python
llm = Llama("model.gguf")
try:
    text = llm.generate("Hello", max_tokens=32)
finally:
    llm.close()  # Always call close() when done
```

**Notes:**
- `close()` is safe to call multiple times
- After `close()`, the instance cannot be used for inference
- Native resources (Context, Model) are freed in the correct order to prevent segfaults
- For `UnifiedLLM`, the backend reference is cleared before closing the underlying `Llama` instance

## Thread Safety

- **Sync methods** (`generate()`, `create_chat_completion()`, etc.) are NOT thread-safe. Do not call them concurrently from multiple threads on the same instance.
- **Async methods** (`generate_async()`, etc.) use an internal lock and serialize concurrent calls.
- For true parallelism, use multiple `Llama` or `UnifiedLLM` instances.
- The `verbose=False` setting affects logging globally, not per-instance.
- The GIL is released during heavy C++ operations (decode, generate, tokenize) to allow other Python threads to run.
