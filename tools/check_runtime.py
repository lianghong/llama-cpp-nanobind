"""Verify Linux native-library provenance before testing an upstream build."""

from __future__ import annotations

import argparse
import ctypes
import json
import platform
import sys
from pathlib import Path
from typing import Any

import llama_cpp._llama as _llama


def runtime_info() -> dict[str, Any]:
    """Report loaded native paths and build identity from the loaded common DSO."""
    if sys.platform != "linux":
        raise RuntimeError("native path verification currently requires Linux /proc")
    libraries: set[Path] = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        path = Path(fields[5])
        if path.name.startswith(("libllama", "libggml")) and ".so" in path.name:
            libraries.add(path.resolve())
    common = [path for path in libraries if path.name.startswith("libllama-common.")]
    if len(common) != 1:
        raise RuntimeError(f"expected one loaded libllama-common, found {common}")
    library = ctypes.CDLL(str(common[0]))

    def string_symbol(name: str) -> str:
        value = ctypes.c_char_p.in_dll(library, name).value
        if value is None:
            raise RuntimeError(f"null native build identity: {name}")
        return value.decode()

    return {
        "commit": string_symbol("LLAMA_COMMIT"),
        "build_number": ctypes.c_int.in_dll(library, "LLAMA_BUILD_NUMBER").value,
        "compiler": string_symbol("LLAMA_COMPILER"),
        "libraries": sorted(str(path) for path in libraries),
        "extension": _llama.__file__,
        "system_info": _llama.print_system_info(),
        "python": sys.version,
        "platform": platform.platform(),
    }


def verify_runtime(info: dict[str, Any], prefix: Path, expected_commit: str) -> None:
    prefix = prefix.resolve()
    commit = info["commit"]
    if len(commit) < 7 or not expected_commit.startswith(commit):
        raise RuntimeError(
            f"loaded commit {commit!r} differs from expected {expected_commit!r}"
        )
    outside = [
        path for path in info["libraries"] if not Path(path).is_relative_to(prefix)
    ]
    if outside:
        raise RuntimeError(f"native libraries outside {prefix}: {outside}")
    for name in ("libllama.", "libggml.", "libggml-base.", "libggml-cpu."):
        if not any(Path(path).name.startswith(name) for path in info["libraries"]):
            raise RuntimeError(f"required native library not loaded: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    info = runtime_info()
    text = json.dumps(info, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    verify_runtime(info, args.prefix, args.expected_commit)


if __name__ == "__main__":
    main()
