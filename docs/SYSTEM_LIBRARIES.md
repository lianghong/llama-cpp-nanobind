# Using System-Installed llama.cpp Libraries

**Purpose:** This document explains how llama-cpp-nanobind uses system-installed llama.cpp from `/usr/local` and how to verify the setup.

---

## Design Philosophy

This project **does not bundle or manage llama.cpp**. Instead, it links against system-installed libraries:

✅ **Advantages:**
- Single llama.cpp installation shared across multiple Python projects
- Easy to update llama.cpp independently (just rebuild and reinstall)
- Smaller Python wheel size (no embedded libraries)
- System package manager can track dependencies
- GPU support (CUDA/Metal) configured at llama.cpp build time

❌ **Disadvantages:**
- Requires llama.cpp to be installed before building Python bindings
- Version compatibility must be maintained manually
- Different systems may have different llama.cpp configurations

---

## Installation Workflow

### 1. Install llama.cpp to `/usr/local`

```bash
# Clone llama.cpp repository
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Create build directory
mkdir build && cd build

# Configure with CUDA support (Linux with NVIDIA GPU)
cmake .. \
  -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native

# Or for macOS with Metal support
cmake .. \
  -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON

# Build (use all CPU cores)
make -j$(nproc)

# Install to /usr/local (requires sudo)
sudo make install
```

**Installed Files:**
```
/usr/local/include/llama.h           # C API header
/usr/local/include/ggml.h            # GGML header
/usr/local/lib/libllama.so           # Main library
/usr/local/lib/libggml.so            # GGML library
/usr/local/lib/libggml-base.so       # GGML base
/usr/local/lib/libggml-cpu.so        # CPU backend
/usr/local/lib/libggml-cuda.so       # CUDA backend (if enabled)
/usr/local/lib/pkgconfig/llama.pc    # pkg-config metadata
```

### 2. Verify Installation

```bash
# Check headers
ls -lh /usr/local/include/llama.h

# Check libraries
ls -lh /usr/local/lib/libllama.so
ls -lh /usr/local/lib/libggml*.so

# Verify library linkage
ldd /usr/local/lib/libllama.so

# Check pkg-config (optional)
pkg-config --cflags llama
pkg-config --libs llama
```

### 3. Build Python Bindings

```bash
# Clone this project
git clone https://github.com/yourusername/llama-cpp-nanobind.git
cd llama-cpp-nanobind

# Create virtual environment
uv venv --python 3.14 .venv
source .venv/bin/activate

# Install in editable mode
# CMake will automatically find /usr/local
uv pip install -e .
```

**Expected CMake Output:**
```
-- Using system llama.cpp from /usr/local
-- CMAKE_PREFIX_PATH: /usr/local
-- Found llama.h: /usr/local/include
-- Found libllama: /usr/local/lib/libllama.so
-- Found libggml: /usr/local/lib/libggml.so
-- Found libggml-base: /usr/local/lib/libggml-base.so
-- Found libggml-cpu: /usr/local/lib/libggml-cpu.so
-- Found libggml-cuda: /usr/local/lib/libggml-cuda.so
-- Linking against: /usr/local/lib/libllama.so;/usr/local/lib/libggml.so;...
```

---

## Library Search Order

CMake searches for llama.cpp in this order:

1. **User override**: `CMAKE_PREFIX_PATH` environment variable
   ```bash
   CMAKE_ARGS="-DCMAKE_PREFIX_PATH=/opt/llama" uv pip install -e .
   ```

2. **Linux**: `/usr/local` (automatically added if `/usr/local/include/llama.h` exists)

3. **macOS**: Homebrew location (auto-detected via `brew --prefix llama.cpp`)

4. **Standard paths**: `/usr`, `/opt/local`, etc.

**Override Example:**
```bash
# Use llama.cpp from a custom location
CMAKE_ARGS="-DCMAKE_PREFIX_PATH=/home/user/llama-install" uv pip install -e .
```

---

## Verification After Build

### Check Python Extension Linkage

```bash
# Find the extension module
EXTENSION=$(find .venv/lib/python3.14/site-packages/llama_cpp -name "_llama*.so")

# Check which libraries it's linked against
ldd $EXTENSION

# Expected output:
#   libllama.so => /usr/local/lib/libllama.so
#   libggml.so => /usr/local/lib/libggml.so
#   libggml-cuda.so => /usr/local/lib/libggml-cuda.so
#   ...
```

### Runtime Test

```python
# test_system_libs.py
from llama_cpp import Llama, LlamaConfig

# This will use libraries from /usr/local
llm = Llama(
    "models/Qwen3-8B-Q6_K.gguf",
    config=LlamaConfig(
        model_path="models/Qwen3-8B-Q6_K.gguf",
        n_ctx=512,
        n_gpu_layers=99,  # GPU offloading works if CUDA/Metal enabled
        verbose=True,
    )
)

response = llm.generate("Hello", max_tokens=10)
print(response)
llm.close()
```

---

## Troubleshooting

### Problem: "llama.h not found"

**Cause:** llama.cpp not installed or not in search path.

**Fix:**
```bash
# Verify installation
ls /usr/local/include/llama.h

# If missing, install llama.cpp to /usr/local
cd /path/to/llama.cpp
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc)
sudo make install
```

### Problem: "libllama.so: cannot open shared object file"

**Cause:** Library path not in `LD_LIBRARY_PATH` (runtime).

**Fix:**
```bash
# Add to shell profile (~/.bashrc or ~/.zshrc)
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Or use ldconfig (requires sudo)
echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/llama.conf
sudo ldconfig

# Verify
ldconfig -p | grep llama
```

### Problem: CMake finds wrong llama.cpp version

**Cause:** Multiple llama.cpp installations, wrong one found first.

**Fix:**
```bash
# Explicitly set prefix
CMAKE_ARGS="-DCMAKE_PREFIX_PATH=/usr/local" uv pip install -e .

# Or remove conflicting installations
sudo rm -rf /usr/lib/libllama.so  # Example: remove old system package
```

### Problem: GPU not working (CUDA/Metal)

**Cause:** llama.cpp was built without GPU support.

**Fix:**
```bash
# Rebuild llama.cpp with GPU support
cd /path/to/llama.cpp
rm -rf build && mkdir build && cd build

# For CUDA:
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DGGML_CUDA=ON
make -j$(nproc)
sudo make install

# For Metal (macOS):
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DGGML_METAL=ON
make -j$(nproc)
sudo make install

# Rebuild Python bindings
cd /path/to/llama-cpp-nanobind
uv pip install -e . --force-reinstall --no-cache-dir
```

---

## Updating llama.cpp

When llama.cpp releases a new version:

```bash
# 1. Update llama.cpp
cd /path/to/llama.cpp
git pull
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DGGML_CUDA=ON
make -j$(nproc)
sudo make install

# 2. Rebuild Python bindings
cd /path/to/llama-cpp-nanobind
uv pip install -e . --force-reinstall --no-cache-dir

# 3. Verify
python -c "from llama_cpp import Llama; print('OK')"
```

---

## Alternative: Using System Package Managers

### Homebrew (macOS)

```bash
# Install llama.cpp via Homebrew
brew install llama.cpp

# Build Python bindings (auto-detects Homebrew)
uv pip install -e .
```

### Custom Prefix (e.g., `/opt/llama`)

```bash
# Install llama.cpp to custom location
cmake .. -DCMAKE_INSTALL_PREFIX=/opt/llama
make -j$(nproc)
sudo make install

# Build Python bindings with custom prefix
CMAKE_ARGS="-DCMAKE_PREFIX_PATH=/opt/llama" uv pip install -e .

# Add to runtime library path
export LD_LIBRARY_PATH=/opt/llama/lib:$LD_LIBRARY_PATH
```

---

## CMakeLists.txt Implementation

The build system explicitly checks for `/usr/local` on Linux:

```cmake
# From CMakeLists.txt lines 63-83
if(APPLE)
  # macOS: check Homebrew
  execute_process(COMMAND brew --prefix llama.cpp ...)
  list(PREPEND CMAKE_PREFIX_PATH "${BREW_LLAMA_PREFIX}")
else()
  # Linux: explicitly prioritize /usr/local
  if(EXISTS "/usr/local/include/llama.h")
    list(PREPEND CMAKE_PREFIX_PATH "/usr/local")
    message(STATUS "Using system llama.cpp from /usr/local")
  endif()
endif()

# Find headers and libraries
find_path(LLAMA_INCLUDE_DIR llama.h REQUIRED)
find_library(LLAMA_LIB llama REQUIRED)
find_library(GGML_LIB ggml REQUIRED)
# ...
```

This ensures `/usr/local` is checked **before** other standard paths like `/usr`.

---

## Summary

✅ **System library approach:**
- llama.cpp installed to `/usr/local`
- Python bindings link against system libs
- CMake automatically discovers `/usr/local`
- Single source of truth for llama.cpp version
- Easy updates via reinstall

✅ **Verification steps:**
1. Check `/usr/local/include/llama.h` exists
2. Check CMake output during build
3. Run `ldd` on extension module
4. Test GPU offloading works

✅ **Override if needed:**
```bash
CMAKE_ARGS="-DCMAKE_PREFIX_PATH=/custom/path" uv pip install -e .
```

---

**Document Version:** 1.0  
**Last Updated:** 2026-03-31  
**Tested On:** Linux (Ubuntu 24.04), macOS (Homebrew)
