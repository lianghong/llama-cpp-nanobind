# Code Review Fixes - 2026-03-31

## Summary

Fixed 6 issues from comprehensive code review:
- **HIGH (3):** 2 fixed, 1 clarified as intentional behavior
- **MEDIUM (1):** 1 fixed
- **LOW (2):** 2 fixed

---

## HIGH Priority Fixes

### ✅ HIGH #1: Backend Initialization Race Condition

**Issue:** `llama_model_load_from_file()` called in member initializer list BEFORE `llama_backend_init()` in constructor body.

**File:** `src/bindings/llama_cpp.cpp:51-64`

**Impact:** CUDA/Metal backends not initialized before model loading, causing undefined behavior on GPU paths.

**Root Cause:**
```cpp
// BEFORE (BUGGY):
explicit Model(const std::string& path, const ModelParams& params)
    : model_(llama_model_load_from_file(...))  // Line 52 - runs FIRST
{
    std::call_once(g_backend_init_flag, init_backend);  // Line 54 - too late!
}
```

**Fix Applied:** Backend now initialized in constructor body before model load.

**Result:** Backend initialization happens before any model loading attempts.

---

### ✅ HIGH #2: Context::reset() Double-Free Risk

**Issue:** `ctx_` freed but not set to nullptr before `llama_init_from_model()`.

**File:** `src/bindings/llama_cpp.cpp:527-537`

**Fix:** Set `ctx_ = nullptr` immediately after `llama_free(ctx_)`.

**Result:** No dangling pointer if init fails.

---

### ✅ HIGH #3: Stop Token KV Cache - NOT A BUG

**Analysis:** Current behavior is CORRECT - stop tokens intentionally not decoded to keep them out of conversation history.

**Fix:** Added clarifying comment explaining intentional design.

---

## MEDIUM & LOW Priority Fixes

### ✅ LlamaPool asyncio.Queue Binding
- Lazy initialization on first async use
- Prevents "no running event loop" errors

### ✅ UnifiedLLM Error Handling
- Added `_check_closed()` guards to helper methods
- Consistent LlamaError messages

### ✅ Logging Performance  
- Changed f-strings to %-formatting
- Lazy evaluation when log level disabled

---

**Status:** ✅ All critical issues fixed, production ready
