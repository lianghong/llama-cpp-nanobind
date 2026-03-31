# PEP 758 & PEP 765 Compliance Report

**Date:** 2026-03-31  
**Python Version:** 3.14  
**Status:** ✅ **FULLY COMPLIANT**

---

## Executive Summary

The codebase has been audited for compliance with Python 3.14's two new exception handling PEPs:

- **PEP 758**: Unparenthesized exception lists in `except` clauses
- **PEP 765**: Control flow restrictions in `finally` blocks

**Result:** All Python files pass compilation with `-We` (warnings-as-errors) flag. Zero violations detected.

---

## PEP 758: Unparenthesized Exception Lists

### Specification

PEP 758 permits three valid exception handling forms:

```python
# Form 1: Parenthesized (existing, always valid)
except (ValueError, TypeError):
    ...

# Form 2: Unparenthesized (NEW - allowed without 'as')
except ValueError, TypeError:
    ...

# Form 3: With capture (REQUIRES parentheses)
except (ValueError, TypeError) as e:
    ...
```

**Critical Rule:** Parentheses are REQUIRED when using the `as` keyword.

### Audit Results

**Files Checked:** 50+ Python files across `src/`, `tests/`, `examples/`, `tools/`

**Violations Found:** 0

#### Exception Patterns Identified

| File | Line | Pattern | Status |
|------|------|---------|--------|
| `src/llama_cpp/llama.py:1823` | `except (KeyError, TypeError, ValueError) as e:` | ✅ Correct (parentheses with `as`) |
| `examples/verify_double_free.py:258` | `except LlamaError, AttributeError:` | ✅ Correct (unparenthesized without `as`) |
| Multiple files | `except Exception as e:` | ✅ Correct (single exception with `as`) |
| Multiple files | `except (RuntimeError, OSError):` | ✅ Correct (parenthesized) |

All exception handling syntax follows PEP 758 requirements.

---

## PEP 765: Control Flow in Finally Blocks

### Specification

PEP 765 makes it a `SyntaxWarning` (future `SyntaxError`) to use control flow statements that exit a `finally` block:

```python
# ❌ DISALLOWED: return from finally
def f():
    try:
        ...
    finally:
        return 42  # SyntaxWarning

# ❌ DISALLOWED: break from finally
for x in items:
    try:
        ...
    finally:
        break  # SyntaxWarning

# ❌ DISALLOWED: continue from finally
for x in items:
    try:
        ...
    finally:
        continue  # SyntaxWarning

# ✅ ALLOWED: control flow in nested scope
try:
    ...
finally:
    def inner():
        return 42  # OK - exits inner function, not finally
    
    for x in items:
        break  # OK - exits inner loop, not finally
```

### Audit Results

**Files with `finally:` blocks:** 6

1. `src/llama_cpp/llama.py`
2. `src/llama_cpp/pool.py`
3. `tools/url2md.py`
4. `tools/md_translator.py`
5. `examples/model_helper_utils.py`
6. `tests/test_regressions.py`

**Violations Found:** 0

#### Finally Block Analysis

All `finally:` blocks were analyzed for control flow statements. Representative examples:

**Example 1: `examples/model_helper_utils.py:63-64`**
```python
finally:
    gguf.GGUFReader._build_tensors = orig  # type: ignore[method-assign]
```
**Status:** ✅ No control flow - simple assignment restoration

**Example 2: `src/llama_cpp/pool.py` (context manager cleanup)**
```python
finally:
    self._closed = True
```
**Status:** ✅ No control flow - state flag update

**Example 3: `src/llama_cpp/llama.py` (streaming cleanup)**
```python
finally:
    cancel_flag.set()
    worker_thread.join(timeout=5.0)
    if worker_thread.is_alive():
        logging.warning(...)
```
**Status:** ✅ No control flow exiting finally - all statements execute sequentially

No `return`, `break`, or `continue` statements found in any `finally` block.

---

## Verification Methodology

### 1. Static Analysis

Compiled all Python files with warnings-as-errors:

```bash
python3.14 -We -m py_compile <file>
```

**Result:** Zero warnings emitted for all 50+ files.

### 2. Pattern Matching

Searched for violation patterns:

```bash
# Check for control flow in finally blocks
grep -A 20 'finally:' **/*.py | grep -E '^\s+(return|break|continue)\b'

# Check for unparenthesized exceptions with 'as'
grep -n 'except.*,.*as\s' **/*.py | grep -v 'except\s*('
```

**Result:** Zero matches.

### 3. Manual Code Review

Inspected all exception handlers and finally blocks individually.

**Result:** All syntax patterns comply with PEP 758 and PEP 765.

---

## Compliance Summary Table

| PEP | Feature | Files Checked | Violations | Status |
|-----|---------|--------------|------------|--------|
| 758 | Unparenthesized exception lists | 50+ | 0 | ✅ Compliant |
| 765 | Control flow in finally blocks | 6 | 0 | ✅ Compliant |

---

## Risk Assessment

### Current Risk: **NONE**

- All code compiles without warnings under Python 3.14's strict mode
- No deprecated syntax patterns identified
- No syntax that will become errors in future Python versions

### Future Maintenance

To maintain compliance:

1. **CI Integration**: Add `-We` flag to test suite:
   ```yaml
   - run: python3.14 -We -m pytest tests/
   ```

2. **Pre-commit Hook**: Compile with warnings:
   ```bash
   python3.14 -We -m py_compile $(git diff --name-only --cached | grep '\.py$')
   ```

3. **Code Review Checklist**:
   - [ ] No `return`/`break`/`continue` exiting `finally` blocks
   - [ ] Exception lists with `as` use parentheses: `except (A, B) as e:`
   - [ ] Unparenthesized exceptions only when no `as` clause

---

## Examples from Codebase

### ✅ Correct Exception Handling

**Multiple exceptions with capture (parentheses required):**
```python
# src/llama_cpp/llama.py:1823
try:
    data = json.loads(text)
except (KeyError, TypeError, ValueError) as e:
    logging.debug("Failed to parse tool call: %s", e)
```

**Multiple exceptions without capture (parentheses optional):**
```python
# examples/verify_double_free.py:258
try:
    llm.generate("Should fail", max_tokens=4)
except LlamaError, AttributeError:  # PEP 758 allows this
    ok = True
```

**Single exception with capture:**
```python
# src/llama_cpp/llama.py:378
try:
    self.ctx.kv_cache_seq_rm(-1, start, end)
except RuntimeError as e:
    raise ValidationError(f"KV cache operation failed: {e}") from e
```

### ✅ Correct Finally Block Usage

**Cleanup without control flow:**
```python
# examples/model_helper_utils.py:63-64
try:
    orig = gguf.GGUFReader._build_tensors
    gguf.GGUFReader._build_tensors = lambda self: None
    reader = gguf.GGUFReader(path)
    # ...
finally:
    gguf.GGUFReader._build_tensors = orig  # Restore monkeypatch
```

**Resource cleanup:**
```python
# src/llama_cpp/llama.py (streaming)
try:
    # Generate tokens...
finally:
    cancel_flag.set()  # Signal worker thread
    worker_thread.join(timeout=5.0)
    if worker_thread.is_alive():
        logging.warning("Worker thread did not stop within timeout")
```

---

## Conclusion

The llama-cpp-nanobind codebase demonstrates **strict compliance** with Python 3.14's PEP 758 and PEP 765 specifications. All exception handling and finally block patterns follow the new requirements correctly.

**No code changes required.**

---

**Audited By:** Claude Code (Sonnet 4.5)  
**Verification:** Python 3.14 `-We` compilation + manual review  
**Files Analyzed:** 50+ Python files  
**Violations:** 0
