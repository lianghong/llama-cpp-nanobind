# lib/

This directory contains prebuilt llama.cpp shared libraries.

## Required Files

### Linux (CUDA)

Core libraries:
- `libllama.so` / `libllama.so.0` - Main llama.cpp library
- `libggml.so` / `libggml.so.0` - GGML tensor library

Backend libraries:
- `libggml-base.so` / `libggml-base.so.0` - Base GGML backend
- `libggml-cpu.so` / `libggml-cpu.so.0` - CPU backend
- `libggml-cuda.so` / `libggml-cuda.so.0` - CUDA backend (GPU acceleration)
- `libggml-blas.so` / `libggml-blas.so.0` - BLAS backend (optional)

### macOS (Metal)

Core libraries:
- `libllama.dylib` - Main llama.cpp library
- `libggml.dylib` - GGML tensor library

Backend libraries:
- `libggml-base.dylib` - Base GGML backend
- `libggml-cpu.dylib` - CPU backend
- `libggml-metal.dylib` - Metal backend (GPU acceleration)

On macOS, Homebrew llama.cpp (`brew install llama.cpp`) is auto-detected by CMake — no manual copying needed.

## Build Requirements

**Linux:**
- CUDA 13.x support enabled
- Compute capability 6.0+ (Pascal and newer GPUs)
- C++17 compatible compiler (GCC 11+ recommended)

**macOS:**
- Xcode Command Line Tools (Apple Clang)
- Metal-capable GPU (Apple Silicon or supported Intel Mac)

## Updating Libraries

When updating llama.cpp:

1. Clone and build llama.cpp:
   ```bash
   # Linux (CUDA)
   cmake -B build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON
   cmake --build build

   # macOS (Metal)
   cmake -B build -DGGML_METAL=ON -DBUILD_SHARED_LIBS=ON
   cmake --build build
   ```
2. Copy shared libraries to this directory:
   - Linux: `.so` and `.so.0` files
   - macOS: `.dylib` files
3. Update headers in `include/` to match
4. Verify RPATH compatibility

## Notes

- Libraries are bundled into the wheel with RPATH set to `$ORIGIN/lib` (Linux) or `@loader_path/lib` (macOS)
- Linux: symlinks (`.so` -> `.so.0`) should be preserved
- Version numbers in sonames must match what the extension expects
