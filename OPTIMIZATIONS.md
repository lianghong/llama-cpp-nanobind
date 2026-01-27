# Optimization Summary

This document summarizes the optimizations applied based on expert review.

## v0.3.0 Optimizations

### 6. GIL Release for Async Operations ✅

**Problem**: Heavy C++ operations (decode, generate, tokenize) held the GIL, blocking other Python threads and stalling async event loops.

**Solution**:
- Added `nb::call_guard<nb::gil_scoped_release>()` to heavy operations:
  - Model loading and tokenize/detokenize
  - Context decode, decode_one, generate_next
  - State save/load operations
  - All generate_tokens* functions

**Files Modified**:
- `src/bindings/llama_cpp.cpp`

### 7. State/KV Cache Position Bookkeeping ✅

**Problem**: `load_state()`, `set_state_data()`, and KV cache operations didn't update `cur_pos_`, causing position corruption on subsequent decodes.

**Solution**:
- `load_state()` and `set_state_data()` now update `cur_pos_` from KV cache after loading
- `kv_cache_seq_rm()`, `kv_cache_seq_keep()`, `kv_cache_seq_add()` update `cur_pos_` when modifying sequence 0

**Files Modified**:
- `src/bindings/llama_cpp.cpp`

### 8. Grammar-Constrained Generation Respects Sampling ✅

**Problem**: `generate_tokens_with_grammar` and `generate_tokens_grammar_multi_stop` used argmax after grammar, ignoring temperature/top_p/top_k.

**Solution**:
- Apply sampler chain after grammar constraint
- Use `cur_p.selected` from sampler instead of argmax

**Files Modified**:
- `src/bindings/llama_cpp.cpp`

### 9. LoRA Persistence Across reset() ✅

**Problem**: `reset()` recreated context but didn't reapply LoRA adapters despite `_reapply_lora_adapters()` existing.

**Solution**:
- `reset()` now calls `_reapply_lora_adapters()` after context reset

**Files Modified**:
- `src/llama_cpp/llama.py`

### 10. Stop Sequences Use Fast Path ✅

**Problem**: Any stop sequence forced the expensive "details" path with O(n_vocab) work per token.

**Solution**:
- `generate()` now uses `generate_tokens_multi_stop()` when stop sequences are provided but logprobs/echo aren't needed

**Files Modified**:
- `src/llama_cpp/llama.py`

### 11. Per-Token Batch Allocation Eliminated ✅

**Problem**: `decode_one()` allocated/freed a batch every token.

**Solution**:
- Added `single_batch_` member to Context class
- `decode_one()` reuses pre-allocated batch

**Files Modified**:
- `src/bindings/llama_cpp.cpp`

### 12. UnifiedLLM Fixes ✅

**Problem**: Multiple issues in UnifiedLLM:
- `kv_cache_clear()` was broken (Llama had no such method)
- `_calc_max_tokens()` didn't account for BOS token
- `max_tokens=0` was silently treated as "auto"
- Resource leak if backend construction failed

**Solution**:
- Added `kv_cache_clear()` method to Llama class
- `_calc_max_tokens()` now uses `add_special=True` for accurate counting
- Validates `max_tokens > 0` and raises clear error for context overflow
- Added try/except in `__init__` to close llm if backend fails
- Added `_prepare_chat()` to avoid double tokenization

**Files Modified**:
- `src/llama_cpp/llama.py`
- `src/llama_cpp/unified.py`

---

## v0.2.0 Optimizations

### 1. LoRA Adapter Persistence ✅

**Problem**: LoRA adapters were silently dropped after the first call because `ctx.reset()` recreated the C++ context without reapplying adapters.

**Solution**:
- Added `_lora_configs` list to track adapter paths and scales
- Created `_reapply_lora_adapters()` method to reapply all loaded adapters after context reset
- Updated `load_lora()` to store adapter configuration
- Updated `remove_lora()` and `clear_lora()` to maintain config list

**Files Modified**:
- `src/llama_cpp/llama.py`

### 2. Multi-Token Stop Sequences ✅

**Problem**: Chat stop sequences were reduced to a single token, so multi-token stops like `<|end_of_turn|>` never triggered.

**Solution**:
- Added two new C++ functions:
  - `generate_tokens_multi_stop()` - supports multi-token stop sequences without grammar
  - `generate_tokens_grammar_multi_stop()` - supports both grammar and multi-token stop sequences
- Changed `create_chat_completion()` to use `stop_sequences: List[List[int]]` instead of `stop_tokens: List[int]`

**Files Modified**:
- `src/bindings/llama_cpp.cpp`
- `src/llama_cpp/llama.py`

### 3. Embedding Context Recreation ✅

**Problem**: Embedding path recreated the whole llama context for every embedding request, thrashing CUDA memory.

**Solution**:
- Added `kv_cache_clear()` method to Context class
- Changed `embed()` and `create_embedding()` to use `kv_cache_clear()` instead of `ctx.reset()`

**Files Modified**:
- `src/bindings/llama_cpp.cpp`
- `src/llama_cpp/llama.py`

### 4. Embedding Validation ✅

**Problem**: Embeddings could silently return empty vectors when `embeddings=False`.

**Solution**:
- Added validation in `embed()` and `create_embedding()` to check `self.config.embeddings`
- Raises `ValidationError` with clear message if embeddings not enabled

**Files Modified**:
- `src/llama_cpp/llama.py`

### 5. Backend Initialization Thread Safety ✅

**Problem**: Backend init was not thread-safe.

**Solution**:
- Replaced plain bool with `std::once_flag` and `std::call_once`
- Added `std::atexit()` handler to call `llama_backend_free()` on exit

**Files Modified**:
- `src/bindings/llama_cpp.cpp`

---

## Testing

Run the test suite to verify all optimizations:

```bash
uv run pytest tests/ -v
```

Key test files:
- `tests/test_regressions.py` - Tests for state load, grammar sampling, LoRA persistence
- `tests/test_optimizations.py` - Tests for embedding, multi-token stops, KV cache
- `tests/test_unified.py` - Tests for UnifiedLLM fixes

## Performance Impact

| Optimization | Impact |
|-------------|--------|
| GIL Release | Better async/threading performance |
| State Bookkeeping | Fixes correctness issue |
| Grammar Sampling | Fixes correctness issue |
| LoRA Persistence | Fixes correctness issue |
| Fast Stop Path | Significant speedup for stop sequences |
| Batch Reuse | Reduced allocations per token |
| UnifiedLLM Fixes | Correctness + performance |

## v0.3.3 Stability & Thread Safety

### 1. Global State Race Condition Fix ✅

**Problem**: The `mark_llama_initialized()` and `_cleanup_all()` functions accessed global state variables (`_llama_initialized`, `_shutdown_called`) without proper lock protection, creating race conditions during concurrent initialization that could cause double-free errors, resource leaks, or segfaults during interpreter shutdown.

**Solution**:
- Protected all global state access with `_cleanup_lock` in `mark_llama_initialized()`
- Added lock protection in `_cleanup_all()` for flag checks
- Performed cleanup outside lock to avoid deadlock

**Files Modified**:
- `src/llama_cpp/llama.py:59-79`

**Impact**: Eliminates critical race condition that could cause crashes in multi-threaded applications.

### 2. Global Logging Warning ✅

**Problem**: The `verbose=False` setting in `LlamaConfig` affects logging globally (llama.cpp limitation) but was not clearly communicated, causing confusion in multi-instance applications where the last instance's setting would affect all instances.

**Solution**:
- Updated `LlamaConfig.verbose` field comment to warn about global effect
- Added runtime `RuntimeWarning` when `verbose=False` is used
- Warning explains the limitation and its impact on multi-instance applications

**Files Modified**:
- `src/llama_cpp/llama.py:207, 262-272`

**Impact**: Clear user communication prevents confusion and debugging difficulties.

### 3. Thread Safety Documentation ✅

**Problem**: Thread safety warnings were buried in docstring, making it easy for users to miss critical information that concurrent access can cause crashes.

**Solution**:
- Made thread safety warning prominent at top of `Llama` class docstring
- Added bold `**WARNING: NOT THREAD-SAFE**` header
- Provided clear guidance on proper usage patterns
- Updated README.md with dedicated thread safety section

**Files Modified**:
- `src/llama_cpp/llama.py:227-243`
- `README.md`
- `CLAUDE.md`

**Impact**: Prevents user errors that could cause crashes or data corruption.

---

## Testing

All changes verified with full test suite:
```bash
uv run pytest tests/ -q
# Result: 19 passed, 86 skipped in 0.74s
```

No regressions introduced.

## Documentation

See also:
- `CODE_REVIEW_FIXES.md` - Detailed analysis of code review issues
- `STREAMING_IMPROVEMENTS.md` - True incremental streaming implementation (v0.3.2)
- `WARMUP_FEATURE.md` - Model warmup for LlamaPool (v0.3.2)
