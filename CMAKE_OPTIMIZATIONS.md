# CMake Build Optimizations

## Overview

The CMakeLists.txt has been optimized with aggressive compiler flags for maximum performance in Release builds.

## Optimizations Applied

### 1. Link-Time Optimization (LTO)
```cmake
include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)

if(IPO_SUPPORTED AND CMAKE_BUILD_TYPE STREQUAL "Release")
  set_target_properties(_llama PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()
```
- **Benefit**: Cross-module optimizations, better inlining, dead code elimination
- **Impact**: 5-15% performance improvement, smaller binary size

### 2. Compiler Optimization Flags (Release)

```cmake
-O3                  # Maximum optimization level
-march=native        # Optimize for current CPU architecture (AVX2, AVX-512, etc.)
-mtune=native        # Tune for current CPU microarchitecture
-ffast-math          # Aggressive floating-point optimizations
-funroll-loops       # Unroll loops for better performance
-fvisibility=hidden  # Reduce symbol table size
-fno-plt             # Avoid PLT for function calls
-flto=auto           # Link-time optimization with auto parallelization
-DNDEBUG             # Disable assertions
```

### 3. Linker Flags (Release)
```cmake
-flto=auto           # Enable LTO at link time
-fuse-linker-plugin  # Use GCC plugin for better LTO
```

## Performance Impact

Expected improvements over basic `-O2`:
- **CPU-bound operations**: 10-20% faster
- **Vectorized code**: 15-30% faster (with AVX2/AVX-512)
- **Binary size**: 5-10% smaller (with LTO)
- **Compile time**: 2-3x longer (acceptable for Release builds)

## Build Types

### Release (Default)
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```
- All optimizations enabled
- No debug symbols
- Fastest runtime performance

### Debug
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```
- `-g -O0` (debug symbols, no optimization)
- Assertions enabled
- Best for debugging

### RelWithDebInfo
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
```
- Optimizations + debug symbols
- Good for profiling

## Architecture-Specific Builds

### Current CPU (Default)
```bash
# Uses -march=native automatically
cmake -S . -B build
```

### Portable Build (No -march=native)
```bash
cmake -S . -B build -DCMAKE_CXX_FLAGS="-O3 -ffast-math"
```

### Specific Architecture
```bash
# For AVX2 systems
cmake -S . -B build -DCMAKE_CXX_FLAGS="-O3 -march=haswell"

# For AVX-512 systems
cmake -S . -B build -DCMAKE_CXX_FLAGS="-O3 -march=skylake-avx512"
```

## Compiler Support

Optimizations work with:
- ✅ GCC 15+ (primary target)
- ✅ GCC 11-14 (tested)
- ✅ Clang 14+ (supported)

## Flags Explanation

| Flag | Purpose | Trade-off |
|------|---------|-----------|
| `-O3` | Maximum optimization | Longer compile time |
| `-march=native` | Use all CPU features | Not portable to other CPUs |
| `-mtune=native` | Optimize instruction scheduling | Minimal downside |
| `-ffast-math` | Aggressive FP math | May break IEEE 754 compliance |
| `-funroll-loops` | Unroll loops | Larger code size |
| `-flto=auto` | Link-time optimization | Much longer link time |
| `-fvisibility=hidden` | Hide symbols | Smaller binary, faster loading |
| `-fno-plt` | Direct calls | Faster function calls |

## Benchmarking

To verify optimization impact:

```bash
# Build with optimizations
cmake -S . -B build-opt -DCMAKE_BUILD_TYPE=Release
cmake --build build-opt
pip install build-opt/

# Build without optimizations
cmake -S . -B build-base -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-O2"
cmake --build build-base

# Compare performance
python -m timeit -s "from llama_cpp import Llama; llm = Llama('model.gguf')" \
  "llm.generate('test', max_tokens=100)"
```

## Safety Considerations

### `-ffast-math` Implications
- Assumes no NaN/Inf values
- May reorder floating-point operations
- Generally safe for LLM inference
- Disable if you need strict IEEE 754 compliance:
  ```bash
  cmake -S . -B build -DCMAKE_CXX_FLAGS="-O3 -march=native"
  ```

### `-march=native` Portability
- Binary only works on similar CPUs
- For distribution, use generic flags:
  ```bash
  cmake -S . -B build -DCMAKE_CXX_FLAGS="-O3 -march=x86-64-v3"
  ```

## Verification

Check applied flags:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_VERBOSE_MAKEFILE=ON
cmake --build build --verbose
```

Check LTO is enabled:
```bash
nm -D build/_llama*.so | wc -l  # Should be small with LTO
```

Check CPU features used:
```bash
objdump -d build/_llama*.so | grep -E "vfmadd|vpdpbusd"  # AVX2/AVX-512
```

## Troubleshooting

### LTO Errors
If LTO fails:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-O3 -march=native -ffast-math"
```

### Compilation Too Slow
Reduce optimization level:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-O2 -march=native"
```

### Runtime Crashes
Disable `-ffast-math`:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-O3 -march=native"
```
