cmake_minimum_required(VERSION 3.26)

# Enforce specific compilers before project():
# - macOS: Apple Clang (Homebrew LLVM has libc++ ABI mismatches)
# - Linux: prefer gcc-15/g++-15 wherever they live; fall back to gcc/g++
if(APPLE)
  if(NOT DEFINED CMAKE_C_COMPILER AND "$ENV{CC}" STREQUAL "")
    find_program(_LLAMA_CC NAMES clang PATHS /usr/bin NO_DEFAULT_PATH)
    if(_LLAMA_CC)
      set(CMAKE_C_COMPILER "${_LLAMA_CC}")
    endif()
  endif()
  if(NOT DEFINED CMAKE_CXX_COMPILER AND "$ENV{CXX}" STREQUAL "")
    find_program(_LLAMA_CXX NAMES clang++ PATHS /usr/bin NO_DEFAULT_PATH)
    if(_LLAMA_CXX)
      set(CMAKE_CXX_COMPILER "${_LLAMA_CXX}")
    endif()
  endif()
else()
  if(NOT DEFINED CMAKE_C_COMPILER AND "$ENV{CC}" STREQUAL "")
    find_program(_LLAMA_CC NAMES gcc-15 gcc)
    if(_LLAMA_CC)
      set(CMAKE_C_COMPILER "${_LLAMA_CC}")
    endif()
  endif()
  if(NOT DEFINED CMAKE_CXX_COMPILER AND "$ENV{CXX}" STREQUAL "")
    find_program(_LLAMA_CXX NAMES g++-15 g++)
    if(_LLAMA_CXX)
      set(CMAKE_CXX_COMPILER "${_LLAMA_CXX}")
    endif()
  endif()
endif()

project(llama_cpp_nanobind VERSION 0.6.0 LANGUAGES C CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(LLAMA_PORTABLE "Build portable wheel without -march=native" OFF)
# -ffast-math enables -funsafe-math-optimizations and can set FTZ/DAZ
# process-wide, altering softmax/sampling numerics. Off by default for
# an inference library; opt-in for users who prefer throughput over fidelity.
option(LLAMA_FAST_MATH "Enable -ffast-math (may alter numerics)" OFF)

# Enable IPO/LTO if supported
include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)

# Compiler feature detection
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-march=native HAS_MARCH_NATIVE)
check_cxx_compiler_flag(-mtune=native HAS_MTUNE_NATIVE)
check_cxx_compiler_flag(-ffast-math HAS_FFAST_MATH)
check_cxx_compiler_flag(-funroll-loops HAS_FUNROLL_LOOPS)
check_cxx_compiler_flag(-fno-plt HAS_FNO_PLT)

set(CMAKE_C_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# Prefer maximum optimization; users can override with -DCMAKE_BUILD_TYPE
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Validate build type
set(VALID_BUILD_TYPES Debug Release RelWithDebInfo MinSizeRel)
if(NOT CMAKE_BUILD_TYPE IN_LIST VALID_BUILD_TYPES)
  message(FATAL_ERROR "Invalid CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}. Must be one of: ${VALID_BUILD_TYPES}")
endif()

# Find Python 3.14+ (matches requires-python in pyproject.toml)
find_package(Python 3.14 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE NANOBIND_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
find_package(nanobind CONFIG REQUIRED PATHS "${NANOBIND_CMAKE_DIR}")

# Configure system library search paths
# Priority: user override > /usr/local > Homebrew (macOS) > system defaults
if(APPLE)
  # On macOS, check for Homebrew llama.cpp
  execute_process(
    COMMAND brew --prefix llama.cpp
    OUTPUT_VARIABLE BREW_LLAMA_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE BREW_RESULT
  )
  if(BREW_RESULT EQUAL 0 AND EXISTS "${BREW_LLAMA_PREFIX}")
    list(APPEND CMAKE_PREFIX_PATH "${BREW_LLAMA_PREFIX}")
    message(STATUS "Added Homebrew llama.cpp to search paths: ${BREW_LLAMA_PREFIX}")
  endif()
else()
  # /usr/local is already a CMake system prefix. Adding it to the variable
  # would override a custom CMAKE_PREFIX_PATH supplied through the environment.
  if(EXISTS "/usr/local/include/llama.h")
    message(STATUS "System llama.cpp available in /usr/local")
  endif()
endif()

# User can override with: cmake -DCMAKE_PREFIX_PATH=/custom/path
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

# Find llama.cpp headers
find_path(LLAMA_INCLUDE_DIR llama.h REQUIRED)
message(STATUS "Found llama.h: ${LLAMA_INCLUDE_DIR}")
find_path(LLAMA_COMMON_INCLUDE_DIR speculative.h
  HINTS "${LLAMA_INCLUDE_DIR}" PATH_SUFFIXES llama common REQUIRED)

# Find llama.cpp shared libraries
find_library(LLAMA_LIB llama REQUIRED)
find_library(GGML_LIB ggml REQUIRED)
find_library(GGML_BASE_LIB ggml-base REQUIRED)
find_library(GGML_CPU_LIB ggml-cpu REQUIRED)
find_library(LLAMA_COMMON_LIB llama-common REQUIRED)

message(STATUS "Found libllama: ${LLAMA_LIB}")
message(STATUS "Found libggml: ${GGML_LIB}")
message(STATUS "Found libggml-base: ${GGML_BASE_LIB}")
message(STATUS "Found libggml-cpu: ${GGML_CPU_LIB}")
message(STATUS "Found libllama-common: ${LLAMA_COMMON_LIB}")

# common/speculative and llama-ext are evolving C++ interfaces. Fail at
# configure time with a useful message instead of building against stale
# headers (or accidentally mixing headers and libraries from two installs).
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES "${LLAMA_INCLUDE_DIR}" "${LLAMA_COMMON_INCLUDE_DIR}")
set(CMAKE_REQUIRED_LIBRARIES "${LLAMA_LIB}" "${LLAMA_COMMON_LIB}")
# Recheck on every configure: system libraries can be upgraded in place while
# the CMake build directory (and its previously successful probe) survives.
unset(LLAMA_HAS_CURRENT_BATCH_API CACHE)
check_cxx_source_compiles("
  #include <llama.h>
  #include <llama-ext.h>
  #include <speculative.h>
  int main() {
    auto m = llama_model_default_params();
    m.load_mode = LLAMA_LOAD_MODE_MMAP_MLOCK;
    m.load_mtp = true;
    auto c = llama_context_default_params();
    c.ctx_other = nullptr;
    c.n_outputs_max = 1;
    c.moe_cache_size = 0;
    common_speculative_draft_params d{};
    d.pos0 = 0;
    auto * penalties = llama_sampler_init_penalties(32, 8, 1.0f, 0.0f, 0.0f);
    auto * dry = llama_sampler_init_dry(nullptr, 0.0f, 1.75f, 2, 8, nullptr, 0);
    common_batch batch(nullptr);
    batch.add(0, 0, 0, true);
    auto rc = llama_process(nullptr, LLAMA_PROCESS_TYPE_DECODE, batch.get());
    return rc + common_speculative_process(nullptr, batch) + llama_model_n_layer_nextn(nullptr);
  }" LLAMA_HAS_CURRENT_BATCH_API)
unset(CMAKE_REQUIRED_INCLUDES)
unset(CMAKE_REQUIRED_LIBRARIES)
if(NOT LLAMA_HAS_CURRENT_BATCH_API)
  message(FATAL_ERROR
    "Incompatible llama.cpp headers/libraries. Install a matching current llama.cpp "
    "build including llama-common, common_batch, llama_process, and llama-ext.h "
    "(validated with b11552 / 23b0202a1, 2026-10-10), then rebuild this extension. "
    "See docs/SYSTEM_LIBRARIES.md.")
endif()

# Optional backends must come from the selected ggml installation. A CPU-only
# custom prefix must not pick up a system CUDA/BLAS DSO with a different ABI.
get_filename_component(GGML_LIBRARY_DIR "${GGML_LIB}" DIRECTORY)
find_library(GGML_CUDA_LIB ggml-cuda HINTS "${GGML_LIBRARY_DIR}" NO_DEFAULT_PATH)
find_library(GGML_METAL_LIB ggml-metal HINTS "${GGML_LIBRARY_DIR}" NO_DEFAULT_PATH)
find_library(GGML_BLAS_LIB ggml-blas HINTS "${GGML_LIBRARY_DIR}" NO_DEFAULT_PATH)

if(GGML_CUDA_LIB)
  message(STATUS "Found libggml-cuda: ${GGML_CUDA_LIB}")
endif()
if(GGML_METAL_LIB)
  message(STATUS "Found libggml-metal: ${GGML_METAL_LIB}")
endif()
if(GGML_BLAS_LIB)
  message(STATUS "Found libggml-blas: ${GGML_BLAS_LIB}")
endif()

set(LLAMA_SHARED_LIBS ${LLAMA_LIB} ${LLAMA_COMMON_LIB} ${GGML_LIB} ${GGML_BASE_LIB} ${GGML_CPU_LIB})
foreach(_optlib GGML_CUDA_LIB GGML_METAL_LIB GGML_BLAS_LIB)
  if(${_optlib})
    list(APPEND LLAMA_SHARED_LIBS ${${_optlib}})
  endif()
endforeach()

message(STATUS "Linking against: ${LLAMA_SHARED_LIBS}")

set(BINDINGS_SOURCES
  src/bindings/llama_cpp.cpp)

nanobind_add_module(
  _llama
  SHARED
  ${BINDINGS_SOURCES}
)

target_include_directories(_llama
  PRIVATE
  ${LLAMA_INCLUDE_DIR}
  ${LLAMA_COMMON_INCLUDE_DIR}
)

# Enable LTO if supported
if(IPO_SUPPORTED AND CMAKE_BUILD_TYPE STREQUAL "Release")
  set_target_properties(_llama PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Optimization flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(_llama PRIVATE -g -O0 -fvisibility=hidden)
  else()
    # Base optimization flags
    set(OPT_FLAGS -O3 -fvisibility=hidden -DNDEBUG)
    
    # Add native arch flags only if not building portable
    if(NOT LLAMA_PORTABLE)
      if(HAS_MARCH_NATIVE)
        list(APPEND OPT_FLAGS -march=native)
      endif()
      if(HAS_MTUNE_NATIVE)
        list(APPEND OPT_FLAGS -mtune=native)
      endif()
    endif()
    
    # Additional optimization flags
    if(LLAMA_FAST_MATH AND HAS_FFAST_MATH)
      list(APPEND OPT_FLAGS -ffast-math)
    endif()
    if(HAS_FUNROLL_LOOPS)
      list(APPEND OPT_FLAGS -funroll-loops)
    endif()
    if(HAS_FNO_PLT)
      list(APPEND OPT_FLAGS -fno-plt)
    endif()
    target_compile_options(_llama PRIVATE ${OPT_FLAGS})
    # CMake's INTERPROCEDURAL_OPTIMIZATION supplies compiler-specific LTO
    # flags only when check_ipo_supported succeeds (also works with Clang).
  endif()
endif()

target_link_libraries(_llama PRIVATE ${LLAMA_SHARED_LIBS})

set_target_properties(_llama PROPERTIES
  OUTPUT_NAME "_llama")

# Install extension into the package folder
install(TARGETS _llama
  LIBRARY DESTINATION llama_cpp
  COMPONENT python)
