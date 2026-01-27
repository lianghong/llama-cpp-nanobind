# lib/

This directory contains prebuilt llama.cpp shared libraries with CUDA support.

## Required Files

Core libraries:
- `libllama.so` / `libllama.so.0` - Main llama.cpp library
- `libggml.so` / `libggml.so.0` - GGML tensor library

Backend libraries:
- `libggml-base.so` / `libggml-base.so.0` - Base GGML backend
- `libggml-cpu.so` / `libggml-cpu.so.0` - CPU backend
- `libggml-cuda.so` / `libggml-cuda.so.0` - CUDA backend (GPU acceleration)
- `libggml-blas.so` / `libggml-blas.so.0` - BLAS backend (optional)

## Build Requirements

Libraries should be built with:
- CUDA 12.x support enabled
- Compute capability 6.0+ (Pascal and newer GPUs)
- C++17 compatible compiler (GCC 11+ recommended)

## Updating Libraries

When updating llama.cpp:

1. Clone and build llama.cpp with CUDA enabled:
   ```bash
   cmake -B build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON
   cmake --build build
   ```
2. Copy shared libraries (`.so` and `.so.0` files) to this directory
3. Update headers in `include/` to match
4. Verify RPATH compatibility

## Notes

- Libraries are bundled into the wheel with RPATH set to `$ORIGIN/lib`
- Symlinks (`.so` -> `.so.0`) should be preserved
- Version numbers in sonames must match what the extension expects
