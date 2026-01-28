"""llama_cpp_nanobind package initializer.

High-performance nanobind bindings for llama.cpp with CUDA enabled by default.
The extension uses RUNPATH ($ORIGIN/lib) to locate bundled shared libraries,
so no manual preloading is required.
"""

from __future__ import annotations

from ._about import __version__

# ruff: noqa: E402
from .llama import (
    GenerationError,
    Llama,
    LlamaConfig,
    LlamaError,
    LlamaGrammar,
    ModelLoadError,
    SamplingParams,
    ValidationError,
    disable_logging,
    print_system_info,
    reset_logging,
    set_log_level,
    shutdown,
)
from .pool import LlamaPool

__all__ = [
    "Llama",
    "LlamaConfig",
    "SamplingParams",
    "LlamaGrammar",
    "LlamaPool",
    "set_log_level",
    "disable_logging",
    "reset_logging",
    "print_system_info",
    "LlamaError",
    "ModelLoadError",
    "GenerationError",
    "ValidationError",
    "shutdown",
    "__version__",
]
