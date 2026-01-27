# include/

This directory contains C/C++ header files from llama.cpp required for building the nanobind extension.

## Required Files

- `llama.h` - Main llama.cpp API header
- `llama-cpp.h` - C++ wrapper header (if available)
- `ggml.h` - GGML tensor library header
- `ggml-backend.h` - GGML backend interface
- `ggml-cuda.h` - CUDA backend header (for GPU support)
- `ggml-alloc.h` - GGML memory allocation
- `common.h` - Common utilities (if needed)

## Updating Headers

When updating llama.cpp:

1. Build llama.cpp from source with your desired configuration
2. Copy the public headers from llama.cpp's `include/` directory
3. Ensure header versions match the shared libraries in `lib/`

## Notes

- Headers must be compatible with the prebuilt libraries in `lib/`
- Version mismatch between headers and libraries will cause build failures or runtime errors
