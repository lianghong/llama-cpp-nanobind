#!/usr/bin/env bash
# Build a wheel for llama-cpp-nanobind.
#
# Usage:
#   scripts/build_wheel.sh                 # native build (uses -march=native)
#   scripts/build_wheel.sh --portable      # no -march=native (redistributable)
#   scripts/build_wheel.sh --clean         # rm -rf build/ dist/ first
#   scripts/build_wheel.sh --install       # also `uv pip install dist/*.whl`
#   scripts/build_wheel.sh --fast-math     # opt-in to -ffast-math
#
# Environment overrides:
#   PYTHON=python3.14        Interpreter used to create/refresh .venv
#   LLAMA_PREFIX=/usr/local  Where llama.cpp is installed (headers + libs)
#   CMAKE_BUILD_TYPE=Release Release | RelWithDebInfo | Debug | MinSizeRel
#   CC=gcc-15  CXX=g++-15    Compiler overrides (default: find_program in CMake)
#   JOBS=$(nproc)            Parallel build jobs
#
# Requirements:
#   - uv (https://github.com/astral-sh/uv) in PATH
#   - Python 3.14+
#   - llama.cpp installed (default: /usr/local)
#   - ninja (installed on demand if missing via scikit-build-core fallback)

set -euo pipefail

# ---- Paths ------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV="${VENV:-$REPO_ROOT/.venv}"
PYTHON="${PYTHON:-python3.14}"
LLAMA_PREFIX="${LLAMA_PREFIX:-/usr/local}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"

# ---- Options ----------------------------------------------------------------

PORTABLE=0
CLEAN=0
INSTALL=0
FAST_MATH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --portable)  PORTABLE=1 ;;
    --clean)     CLEAN=1 ;;
    --install)   INSTALL=1 ;;
    --fast-math) FAST_MATH=1 ;;
    -h|--help)
      sed -n '2,22p' "$0"
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done

# ---- Output helpers ---------------------------------------------------------

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m  %s\n' "$*" >&2; }
fail() { printf '\033[1;31mxx\033[0m  %s\n' "$*" >&2; exit 1; }

# ---- Preflight --------------------------------------------------------------

command -v uv >/dev/null 2>&1 || fail "uv not found in PATH (install: https://github.com/astral-sh/uv)"

if [[ ! -f "$LLAMA_PREFIX/include/llama.h" ]]; then
  fail "llama.h not found under $LLAMA_PREFIX/include — set LLAMA_PREFIX or install llama.cpp"
fi

if [[ ! -f "$LLAMA_PREFIX/lib/libllama.so" && ! -f "$LLAMA_PREFIX/lib/libllama.dylib" ]]; then
  fail "libllama not found under $LLAMA_PREFIX/lib"
fi

log "Using llama.cpp from: $LLAMA_PREFIX"
log "Build type:          $CMAKE_BUILD_TYPE"
log "Parallel jobs:       $JOBS"
[[ $PORTABLE  -eq 1 ]] && log "Portable build:      ON (no -march=native)"
[[ $FAST_MATH -eq 1 ]] && log "Fast-math:           ON"

# ---- Clean ------------------------------------------------------------------

if [[ $CLEAN -eq 1 ]]; then
  log "Cleaning build/ dist/ *.egg-info"
  rm -rf build build-debug dist ./*.egg-info
  find src/llama_cpp -maxdepth 1 \( -name '*.so' -o -name '*.dylib' \) -delete 2>/dev/null || true
fi

# ---- Venv -------------------------------------------------------------------

if [[ ! -x "$VENV/bin/python" ]]; then
  log "Creating venv at $VENV with $PYTHON"
  uv venv --python "$PYTHON" "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

log "Ensuring build tooling is present in venv"
uv pip install --quiet build

# ---- CMake args -------------------------------------------------------------
#
# Do NOT set CMAKE_PREFIX_PATH here: scikit-build-core injects its own value
# so that find_package(nanobind) locates nanobind inside the isolated build
# environment. Overriding CMAKE_PREFIX_PATH would shadow that injection and
# break the build. CMakeLists.txt already prepends $LLAMA_PREFIX (/usr/local)
# on Linux when it detects /usr/local/include/llama.h, and auto-detects
# Homebrew on macOS.
#
# For non-standard llama.cpp installs, use: LLAMA_PREFIX=/path ./build_wheel.sh
# and let CMake discover it via find_library/find_path (llama.cpp installs
# with no CMake config package, so CMAKE_PREFIX_PATH + find_path is the path).

CMAKE_ARGS_ACCUM=()

if [[ $PORTABLE  -eq 1 ]]; then CMAKE_ARGS_ACCUM+=("-DLLAMA_PORTABLE=ON");  fi
if [[ $FAST_MATH -eq 1 ]]; then CMAKE_ARGS_ACCUM+=("-DLLAMA_FAST_MATH=ON"); fi

# Forward LLAMA_PREFIX via env CMAKE_PREFIX_PATH only if it's not /usr/local
# (which CMakeLists.txt already handles natively). Env CMAKE_PREFIX_PATH is
# appended to by CMake, so this composes with scikit-build-core's injection.
if [[ "$LLAMA_PREFIX" != "/usr/local" ]]; then
  if [[ -n "${CMAKE_PREFIX_PATH:-}" ]]; then
    export CMAKE_PREFIX_PATH="$LLAMA_PREFIX:$CMAKE_PREFIX_PATH"
  else
    export CMAKE_PREFIX_PATH="$LLAMA_PREFIX"
  fi
fi

# Compose CMAKE_ARGS: append to any pre-existing value (scalar env var).
_PREEXISTING_CMAKE_ARGS="${CMAKE_ARGS:-}"
if [[ ${#CMAKE_ARGS_ACCUM[@]} -gt 0 ]]; then
  export CMAKE_ARGS="${_PREEXISTING_CMAKE_ARGS} ${CMAKE_ARGS_ACCUM[*]}"
fi
export CMAKE_BUILD_TYPE
export CMAKE_BUILD_PARALLEL_LEVEL="$JOBS"

log "CMAKE_ARGS: ${CMAKE_ARGS:-<none>}"
log "CMAKE_PREFIX_PATH (env): ${CMAKE_PREFIX_PATH:-<default>}"

# ---- Build ------------------------------------------------------------------

log "Building wheel (isolated; re-uses CCache if available)"
python -m build --wheel

# ---- Report -----------------------------------------------------------------

shopt -s nullglob
WHEELS=(dist/*.whl)
shopt -u nullglob

if [[ ${#WHEELS[@]} -eq 0 ]]; then
  fail "build succeeded but no wheel in dist/"
fi

log "Built:"
for w in "${WHEELS[@]}"; do
  printf '    %s  (%s)\n' "$w" "$(du -h "$w" | cut -f1)"
done

# ---- Optional install -------------------------------------------------------

if [[ $INSTALL -eq 1 ]]; then
  LATEST="${WHEELS[-1]}"
  log "Installing $LATEST into $VENV"
  uv pip install --force-reinstall --no-deps "$LATEST"
fi

log "Done"
