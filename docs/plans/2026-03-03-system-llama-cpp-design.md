# Design: System-Installed llama.cpp Integration

**Date**: 2026-03-03
**Status**: Approved

## Goal

Remove local `include/` and `lib/` directory support from the build system. Use system-installed llama.cpp (e.g., from `/usr/local`) discovered via CMake's `find_library()` and `find_path()`. Wheels no longer bundle shared libraries — runtime depends on system-installed llama.cpp.

## Decisions

1. **System-only discovery**: No local `include/` or `lib/` fallback
2. **No wheel bundling**: Extension links against system libs at runtime
3. **macOS Homebrew kept**: Auto-detect via `brew --prefix` as a system path
4. **Standard CMake override**: Users use `CMAKE_PREFIX_PATH` instead of `LLAMA_LIB_DIR`/`LLAMA_INCLUDE_DIR`

## Changes

### CMakeLists.txt

- Replace `LLAMA_LIB_DIR`/`LLAMA_INCLUDE_DIR` env-var-based discovery with:
  - `find_path(LLAMA_INCLUDE_DIR llama.h)` for headers
  - `find_library()` for `llama`, `ggml`, `ggml-base`, `ggml-cpu`, `ggml-cuda`, `ggml-blas`
- On macOS, prepend Homebrew llama.cpp prefix to `CMAKE_PREFIX_PATH` before search
- Remove `$ORIGIN/lib` / `@loader_path/lib` RPATH (system libs found via default linker paths)
- Remove `install(FILES ${LLAMA_SHARED_LIBS} ...)` — no library bundling

### pyproject.toml

- Remove `"include", "lib"` from `sdist.include`

### src/llama_cpp/__init__.py

- Update docstring to reflect system library linking

### .gitignore

- Remove `include/*`, `!include/README.md`, `lib/*`, `!lib/README.md` entries

### CLAUDE.md

- Update build instructions and architecture docs to reflect system-installed llama.cpp
