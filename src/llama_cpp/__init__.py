"""llama_cpp_nanobind package initializer.

High-performance nanobind bindings for llama.cpp.
The extension links against system-installed llama.cpp shared libraries
(e.g., from /usr/local/lib or Homebrew on macOS).
"""

from ._about import __version__

# ruff: noqa: E402
from .llama import disable_logging
from .llama import GenerationError
from .llama import GGML_TYPE_BF16
from .llama import GGML_TYPE_F16
from .llama import GGML_TYPE_F32
from .llama import GGML_TYPE_IQ4_NL
from .llama import GGML_TYPE_Q4_0
from .llama import GGML_TYPE_Q4_1
from .llama import GGML_TYPE_Q5_0
from .llama import GGML_TYPE_Q5_1
from .llama import GGML_TYPE_Q8_0
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
    "GGML_TYPE_F32",
    "GGML_TYPE_F16",
    "GGML_TYPE_BF16",
    "GGML_TYPE_Q4_0",
    "GGML_TYPE_Q4_1",
    "GGML_TYPE_Q5_0",
    "GGML_TYPE_Q5_1",
    "GGML_TYPE_Q8_0",
    "GGML_TYPE_IQ4_NL",
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
