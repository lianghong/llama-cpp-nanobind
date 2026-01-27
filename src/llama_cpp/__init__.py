"""llama_cpp_nanobind package initializer.

High-performance nanobind bindings for llama.cpp with CUDA enabled by default.
We preload bundled ggml/llama shared libraries before importing the extension
to avoid missing-soname issues in editable installs.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from ._about import __version__


def _preload_shared_libs() -> None:
    """Load bundled llama.cpp shared libraries with RTLD_GLOBAL.

    Works both for editable installs (loading from source ``./lib``) and
    wheel installs (``llama_cpp/lib`` packaged with the extension). This avoids
    `ImportError: libggml-blas.so.0: cannot open shared object file` when the
    loader cannot find transitive deps.
    """
    import logging
    import warnings

    candidates = []
    here = Path(__file__).resolve().parent
    candidates.append(here / "lib")
    candidates.append(
        Path(__file__).resolve().parent.parent / "lib"
    )  # project root in editable

    names = [
        # primary sonames
        "libllama.so",
        "libggml.so",
        "libggml-base.so",
        "libggml-cpu.so",
        "libggml-blas.so",
        "libggml-cuda.so",
        "libmtmd.so",
        "libmtmb.so",
        # common linker-resolved aliases used by the prebuilt libs
        "libllama.so.0",
        "libggml.so.0",
        "libggml-base.so.0",
        "libggml-cpu.so.0",
        "libggml-blas.so.0",
        "libggml-cuda.so.0",
    ]

    flags = ctypes.RTLD_GLOBAL if hasattr(ctypes, "RTLD_GLOBAL") else None
    loaded_any = False
    failed_libs: list[tuple[str, str]] = []

    for root in candidates:
        if not root.exists():
            continue
        for name in names:
            lib_path = root / name
            if lib_path.exists():
                try:
                    ctypes.CDLL(
                        str(lib_path), mode=flags or 0
                    )  # keep handle alive via ctypes cache
                    loaded_any = True
                except OSError as e:
                    # Track failures for potential warning
                    failed_libs.append((name, str(e)))

    # Log individual failures at debug level for troubleshooting
    for name, error in failed_libs:
        logging.debug(f"Could not preload {name}: {error}")

    # Warn if critical libraries failed to load
    critical_libs = {"libllama.so", "libggml.so"}
    critical_failures = [name for name, _ in failed_libs if name in critical_libs]
    if critical_failures:
        warnings.warn(
            f"Failed to preload critical libraries: {', '.join(critical_failures)}. "
            "This may cause import errors.",
            RuntimeWarning,
            stacklevel=2,
        )
    elif failed_libs and not loaded_any:
        lib_names = ", ".join(name for name, _ in failed_libs[:3])
        warnings.warn(
            f"Failed to preload llama.cpp libraries ({lib_names}). "
            "This may cause import errors. Check CUDA/library installation.",
            RuntimeWarning,
            stacklevel=2,
        )


_preload_shared_libs()

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
