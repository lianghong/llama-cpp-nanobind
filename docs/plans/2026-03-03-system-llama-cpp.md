# System llama.cpp Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove local `include/` and `lib/` directory support; use system-installed llama.cpp discovered via CMake `find_library()`/`find_path()`, with no library bundling in wheels.

**Architecture:** CMake uses `find_path()` for headers and `find_library()` for each llama.cpp shared library. On macOS, Homebrew prefix is prepended to `CMAKE_PREFIX_PATH` before the search. RPATH is removed since system libraries are found via default linker paths. Wheels contain only the extension module.

**Tech Stack:** CMake 3.26+, scikit-build-core, nanobind

---

### Task 1: Rewrite CMakeLists.txt library discovery

**Files:**
- Modify: `CMakeLists.txt:63-115` (replace local dir discovery with find_library/find_path)

**Step 1: Replace lines 63-115 with system discovery**

Replace this section (lines 63-115):
```cmake
# Paths for headers and pre-built libraries
if(NOT DEFINED LLAMA_LIB_DIR)
  set(LLAMA_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib")
endif()
if(NOT DEFINED LLAMA_INCLUDE_DIR)
  set(LLAMA_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
endif()

# On macOS, auto-detect Homebrew llama.cpp if local dirs are empty
if(APPLE)
  ...entire block...
endif()

# Collect all shared libraries including versioned symlinks
...entire glob block...

if(NOT LLAMA_SHARED_LIBS)
  message(FATAL_ERROR "No prebuilt llama.cpp libraries found in ${LLAMA_LIB_DIR}")
endif()
```

With:
```cmake
# On macOS, add Homebrew llama.cpp to search paths
if(APPLE)
  execute_process(
    COMMAND brew --prefix llama.cpp
    OUTPUT_VARIABLE BREW_LLAMA_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE BREW_RESULT
  )
  if(BREW_RESULT EQUAL 0 AND EXISTS "${BREW_LLAMA_PREFIX}")
    list(PREPEND CMAKE_PREFIX_PATH "${BREW_LLAMA_PREFIX}")
    message(STATUS "Added Homebrew llama.cpp to search paths: ${BREW_LLAMA_PREFIX}")
  endif()
endif()

# Find llama.cpp headers
find_path(LLAMA_INCLUDE_DIR llama.h REQUIRED)
message(STATUS "Found llama.h: ${LLAMA_INCLUDE_DIR}")

# Find llama.cpp shared libraries
find_library(LLAMA_LIB llama REQUIRED)
find_library(GGML_LIB ggml REQUIRED)
find_library(GGML_BASE_LIB ggml-base REQUIRED)
find_library(GGML_CPU_LIB ggml-cpu REQUIRED)

# Optional GPU backend libraries
find_library(GGML_CUDA_LIB ggml-cuda)
find_library(GGML_BLAS_LIB ggml-blas)

set(LLAMA_SHARED_LIBS ${LLAMA_LIB} ${GGML_LIB} ${GGML_BASE_LIB} ${GGML_CPU_LIB})
foreach(_optlib GGML_CUDA_LIB GGML_BLAS_LIB)
  if(${_optlib})
    list(APPEND LLAMA_SHARED_LIBS ${${_optlib}})
  endif()
endforeach()

message(STATUS "llama.cpp libraries: ${LLAMA_SHARED_LIBS}")
```

**Step 2: Verify the change compiles**

Run: `cd /data/storage/Projects/github_projects/llama-cpp-nanobind && uv pip install -e . 2>&1 | tail -20`
Expected: Successful build finding libs from /usr/local

**Step 3: Commit**

```bash
git add CMakeLists.txt
git commit -m "build: use find_library/find_path for system llama.cpp discovery"
```

---

### Task 2: Remove RPATH and library bundling from CMakeLists.txt

**Files:**
- Modify: `CMakeLists.txt:176-197` (remove RPATH config and library install)

**Step 1: Remove RPATH configuration (lines 178-187)**

Replace:
```cmake
# Ensure the runtime loader finds packaged libs
if(APPLE)
  set(_RPATH "@loader_path/lib")
else()
  set(_RPATH "$ORIGIN/lib")
endif()
set_target_properties(_llama PROPERTIES
  OUTPUT_NAME "_llama"
  INSTALL_RPATH "${_RPATH}"
  BUILD_WITH_INSTALL_RPATH TRUE)
```

With:
```cmake
set_target_properties(_llama PROPERTIES
  OUTPUT_NAME "_llama")
```

**Step 2: Remove library bundling install (lines 194-197)**

Delete:
```cmake
# Ship the prebuilt libraries alongside the extension
install(FILES ${LLAMA_SHARED_LIBS}
  DESTINATION llama_cpp/lib
  COMPONENT python)
```

**Step 3: Verify build still works**

Run: `cd /data/storage/Projects/github_projects/llama-cpp-nanobind && uv pip install -e . 2>&1 | tail -20`
Expected: Successful build, no RPATH warnings

**Step 4: Verify the extension loads**

Run: `python -c "from llama_cpp import Llama; print('OK')"`
Expected: `OK`

**Step 5: Commit**

```bash
git add CMakeLists.txt
git commit -m "build: remove RPATH and library bundling (system libs at runtime)"
```

---

### Task 3: Update pyproject.toml

**Files:**
- Modify: `pyproject.toml:53`

**Step 1: Remove include and lib from sdist.include**

Change line 53 from:
```toml
sdist.include = ["include", "lib", "examples", "tests"]
```
To:
```toml
sdist.include = ["examples", "tests"]
```

**Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "build: remove include/lib from sdist (system-installed llama.cpp)"
```

---

### Task 4: Update __init__.py docstring

**Files:**
- Modify: `src/llama_cpp/__init__.py:1-6`

**Step 1: Update module docstring**

Change:
```python
"""llama_cpp_nanobind package initializer.

High-performance nanobind bindings for llama.cpp.
The extension uses RUNPATH ($ORIGIN/lib on Linux, @loader_path/lib on macOS)
to locate bundled shared libraries, so no manual preloading is required.
"""
```
To:
```python
"""llama_cpp_nanobind package initializer.

High-performance nanobind bindings for llama.cpp.
The extension links against system-installed llama.cpp shared libraries
(e.g., from /usr/local/lib or Homebrew on macOS).
"""
```

**Step 2: Commit**

```bash
git add src/llama_cpp/__init__.py
git commit -m "docs: update init docstring for system library linking"
```

---

### Task 5: Update .gitignore

**Files:**
- Modify: `.gitignore:51-55`

**Step 1: Remove include/ and lib/ gitignore entries**

Remove these lines (51-55):
```gitignore
# External dependencies (obtain separately, keep README.md)
include/*
!include/README.md
lib/*
!lib/README.md
```

**Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: remove include/lib from gitignore (no longer project-local)"
```

---

### Task 6: Update README.md

**Files:**
- Modify: `README.md` (multiple sections)

**Step 1: Update project layout section (lines 19-22)**

Remove:
```markdown
**External dependencies (not included, see setup below):**
- `include/` – headers from llama.cpp
- `lib/` – precompiled shared libraries
- `models/` – GGUF model files
```

Replace with:
```markdown
**External dependencies:**
- System-installed llama.cpp (headers + shared libraries)
- `models/` – GGUF model files (not included)
```

**Step 2: Update External Dependencies Setup section (lines 38-71)**

Replace the entire section with:
```markdown
## External Dependencies Setup

Before building, you need llama.cpp installed on your system. CMake uses `find_library()` and `find_path()` to discover headers and libraries from standard system paths.

### Option 1: Homebrew (macOS)

\`\`\`bash
brew install llama.cpp
# CMake auto-detects Homebrew paths — no manual setup needed
\`\`\`

### Option 2: Build and install llama.cpp from source

\`\`\`bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Linux (CUDA)
cmake -B build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON
cmake --build build --config Release
sudo cmake --install build

# macOS (Metal)
cmake -B build -DGGML_METAL=ON -DBUILD_SHARED_LIBS=ON
cmake --build build --config Release
sudo cmake --install build
\`\`\`

### Custom install prefix

If llama.cpp is installed to a non-standard location, pass it via `CMAKE_PREFIX_PATH`:

\`\`\`bash
CMAKE_ARGS="-DCMAKE_PREFIX_PATH=/opt/llama" uv pip install -e .
\`\`\`
```

**Step 3: Update build description (line 93)**

Change:
```markdown
`scikit-build-core` drives the build; it automatically links against the prebuilt libraries in `./lib` and installs them into the wheel. RPATH is set so the extension finds `llama_cpp/lib` at runtime.
```
To:
```markdown
`scikit-build-core` drives the build; it links against system-installed llama.cpp libraries found via CMake's `find_library()`.
```

**Step 4: Update optional build flags section (lines 166-177)**

Replace:
```markdown
### Optional build flags

\`\`\`bash
# Custom build type or different lib/include roots
CMAKE_BUILD_TYPE=RelWithDebInfo \
LLAMA_LIB_DIR=$(pwd)/lib \
LLAMA_INCLUDE_DIR=$(pwd)/include \
uv pip install -e .

# Portable build without -march=native (for distributable wheels)
CMAKE_ARGS="-DLLAMA_PORTABLE=ON" uv pip install -e .
\`\`\`
```
With:
```markdown
### Optional build flags

\`\`\`bash
# Custom build type
CMAKE_BUILD_TYPE=RelWithDebInfo uv pip install -e .

# Custom llama.cpp install prefix
CMAKE_ARGS="-DCMAKE_PREFIX_PATH=/opt/llama" uv pip install -e .

# Portable build without -march=native
CMAKE_ARGS="-DLLAMA_PORTABLE=ON" uv pip install -e .
\`\`\`
```

**Step 5: Update license section (line 516)**

Change:
```markdown
This package includes prebuilt libraries from [llama.cpp](https://github.com/ggerganov/llama.cpp), which is also MIT licensed. See the llama.cpp repository for full license details and attribution requirements.
```
To:
```markdown
This package links against [llama.cpp](https://github.com/ggerganov/llama.cpp), which is also MIT licensed. See the llama.cpp repository for full license details and attribution requirements.
```

**Step 6: Commit**

```bash
git add README.md
git commit -m "docs: update README for system-installed llama.cpp"
```

---

### Task 7: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md` (multiple sections)

**Step 1: Update Build Configuration section**

Replace the env var examples:
```bash
# Custom lib/include paths
LLAMA_LIB_DIR=$(pwd)/lib LLAMA_INCLUDE_DIR=$(pwd)/include uv pip install -e .
```
With:
```bash
# Custom llama.cpp install prefix
CMAKE_ARGS="-DCMAKE_PREFIX_PATH=/opt/llama" uv pip install -e .
```

**Step 2: Update Architecture > Component Structure section**

Replace "External Dependencies" item:
```markdown
1. **External Dependencies** (`lib/`, `include/`, `models/`)
   - **Not included in repo** - must be obtained separately
   - `include/`: llama.cpp headers for C++ bindings
   - `lib/`: Prebuilt llama.cpp shared libraries (`.so` on Linux/CUDA, `.dylib` on macOS/Metal)
   - `models/`: GGUF model files
   - On macOS, Homebrew llama.cpp is auto-detected by CMake
   - See README.md for setup instructions
```
With:
```markdown
1. **External Dependencies** (`models/`)
   - llama.cpp must be installed on the system (headers + shared libraries)
   - CMake discovers headers via `find_path()` and libraries via `find_library()`
   - On macOS, Homebrew llama.cpp prefix is auto-added to `CMAKE_PREFIX_PATH`
   - `models/`: GGUF model files (not included)
```

**Step 3: Update Library Preloading section**

Replace:
```markdown
4. **Library Preloading**
   - `_preload_shared_libs()` in `__init__.py` ensures CUDA/Metal/ggml libraries load correctly
   - Works for both editable installs (from `./lib`) and wheel installs (`llama_cpp/lib`)
   - RPATH: `$ORIGIN/lib` on Linux, `@loader_path/lib` on macOS
```
With:
```markdown
4. **Library Linking**
   - Extension links against system-installed llama.cpp shared libraries
   - No RPATH or library bundling — relies on system linker paths
   - On macOS, Homebrew llama.cpp is auto-detected via `brew --prefix`
```

**Step 4: Update "Integration with llama.cpp" section**

Replace:
```markdown
## Integration with llama.cpp

This project uses **prebuilt** llama.cpp libraries in `lib/`. When updating llama.cpp:

1. Build llama.cpp with GPU support (`-DGGML_CUDA=ON` on Linux, `-DGGML_METAL=ON` on macOS)
2. Copy headers to `include/`
3. Copy shared libraries to `lib/` (`.so` on Linux, `.dylib` on macOS)
4. Verify RPATH and soname compatibility
5. Update C++ bindings if API changed
```
With:
```markdown
## Integration with llama.cpp

This project links against system-installed llama.cpp. When updating:

1. Build llama.cpp with GPU support (`-DGGML_CUDA=ON` on Linux, `-DGGML_METAL=ON` on macOS)
2. Install to system: `sudo cmake --install build`
3. Rebuild the extension: `uv pip install -e .`
4. Update C++ bindings if API changed
```

**Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for system-installed llama.cpp"
```

---

### Task 8: Verify end-to-end

**Step 1: Clean build and verify**

Run:
```bash
cd /data/storage/Projects/github_projects/llama-cpp-nanobind
uv pip install -e . 2>&1 | tail -20
```
Expected: Build succeeds, finds libs from /usr/local

**Step 2: Verify import works**

Run: `python -c "from llama_cpp import Llama; print('OK')"`
Expected: `OK`

**Step 3: Run tests**

Run: `uv run pytest -q`
Expected: All tests pass
