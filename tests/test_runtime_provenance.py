"""The compatibility gate must reject a mixed or unexpected native runtime."""

from pathlib import Path

import pytest

from tools.check_runtime import verify_runtime

COMMIT = "23b0202a189c44a54625aadcb37a946dd1d6278d"


def _info(prefix: Path) -> dict:
    return {
        "commit": COMMIT[:7],
        "libraries": [
            str(prefix / "lib" / name)
            for name in (
                "libllama.so.0.6.0",
                "libllama-common.so.0.6.0",
                "libggml.so.0.26.0",
                "libggml-base.so.0.26.0",
                "libggml-cpu.so.0.26.0",
            )
        ],
    }


def test_runtime_accepts_matching_isolated_prefix(tmp_path: Path):
    verify_runtime(_info(tmp_path), tmp_path, COMMIT)


@pytest.mark.parametrize("commit", ["cbe28bb3", "unknown", "23"])
def test_runtime_rejects_unexpected_commit(tmp_path: Path, commit: str):
    info = _info(tmp_path)
    info["commit"] = commit
    with pytest.raises(RuntimeError, match="differs from expected"):
        verify_runtime(info, tmp_path, COMMIT)


def test_runtime_rejects_system_library_mixed_into_prefix(tmp_path: Path):
    info = _info(tmp_path)
    info["libraries"].append("/usr/local/lib/libggml-cuda.so.0")
    with pytest.raises(RuntimeError, match="outside"):
        verify_runtime(info, tmp_path, COMMIT)


def test_runtime_rejects_missing_native_library(tmp_path: Path):
    info = _info(tmp_path)
    info["libraries"] = [
        path for path in info["libraries"] if "libggml-cpu" not in path
    ]
    with pytest.raises(RuntimeError, match="not loaded"):
        verify_runtime(info, tmp_path, COMMIT)
