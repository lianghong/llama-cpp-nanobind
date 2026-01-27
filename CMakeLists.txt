cmake_minimum_required(VERSION 3.26)

# Enforce GCC/G++ 15 as required by the environment (set before project())
if(NOT DEFINED CMAKE_C_COMPILER)
  set(CMAKE_C_COMPILER /usr/local/bin/gcc-15)
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER /usr/local/bin/g++-15)
endif()

project(llama_cpp_nanobind VERSION 0.3.0 LANGUAGES C CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(LLAMA_PORTABLE "Build portable wheel without -march=native" OFF)

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

# Find Python 3.12+ (matches requires-python in pyproject.toml)
find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

# Paths for headers and pre-built libraries
if(NOT DEFINED LLAMA_LIB_DIR)
  set(LLAMA_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib")
endif()
if(NOT DEFINED LLAMA_INCLUDE_DIR)
  set(LLAMA_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
endif()

# Collect all shared libraries including versioned symlinks
# Using glob here is acceptable since these are prebuilt external libs
file(GLOB LLAMA_SHARED_LIBS
  "${LLAMA_LIB_DIR}/libllama.so*"
  "${LLAMA_LIB_DIR}/libggml*.so*"
  "${LLAMA_LIB_DIR}/libmtm*.so*"
)
list(REMOVE_DUPLICATES LLAMA_SHARED_LIBS)

if(NOT LLAMA_SHARED_LIBS)
  message(FATAL_ERROR "No prebuilt llama.cpp libraries found in ${LLAMA_LIB_DIR}")
endif()

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
    if(HAS_FFAST_MATH)
      list(APPEND OPT_FLAGS -ffast-math)
    endif()
    if(HAS_FUNROLL_LOOPS)
      list(APPEND OPT_FLAGS -funroll-loops)
    endif()
    if(HAS_FNO_PLT)
      list(APPEND OPT_FLAGS -fno-plt)
    endif()
    list(APPEND OPT_FLAGS -flto=auto)
    
    target_compile_options(_llama PRIVATE ${OPT_FLAGS})
    target_link_options(_llama PRIVATE -flto=auto -fuse-linker-plugin)
  endif()
endif()

target_link_libraries(_llama PRIVATE ${LLAMA_SHARED_LIBS})

# Ensure the runtime loader finds packaged libs
set_target_properties(_llama PROPERTIES
  OUTPUT_NAME "_llama"
  INSTALL_RPATH "$ORIGIN/lib"
  BUILD_WITH_INSTALL_RPATH TRUE)

# Install extension into the package folder
install(TARGETS _llama
  LIBRARY DESTINATION llama_cpp
  COMPONENT python)

# Ship the prebuilt libraries alongside the extension
install(FILES ${LLAMA_SHARED_LIBS}
  DESTINATION llama_cpp/lib
  COMPONENT python)
