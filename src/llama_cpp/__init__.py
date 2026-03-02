"""llama_cpp_nanobind package initializer.

High-performance nanobind bindings for llama.cpp.
The extension links against system-installed llama.cpp shared libraries
(e.g., from /usr/local/lib or Homebrew on macOS).
"""

from ._about import __version__

# ruff: noqa: E402
from .llama import disable_logging
from .llama import GenerationError
from .llama import Llama
from .llama import LlamaConfig
from .llama import LlamaError
from .llama import LlamaGrammar
from .llama import ModelLoadError
from .llama import print_system_info
from .llama import reset_logging
from .llama import SamplingParams
from .llama import set_log_level
from .llama import shutdown
from .llama import ValidationError
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
