"""Linux NVIDIA process-memory sampling without a Python NVML dependency."""

from __future__ import annotations

import ctypes
import os
import threading
import time
from typing import Any


class _ProcessInfo(ctypes.Structure):
    # nvmlProcessInfo_t, shared by ComputeRunningProcesses_v2 and _v3.
    _fields_ = [
        ("pid", ctypes.c_uint),
        ("used_gpu_memory", ctypes.c_ulonglong),
        ("gpu_instance_id", ctypes.c_uint),
        ("compute_instance_id", ctypes.c_uint),
    ]


class GpuMemorySampler:
    """Observe this process across physical GPUs during one inference request."""

    def __init__(self, interval_seconds: float = 0.01):
        if interval_seconds <= 0:
            raise ValueError("memory sampling interval must be positive")
        self.interval = interval_seconds
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        self.lib.nvmlInit_v2.argtypes = []
        self.lib.nvmlInit_v2.restype = ctypes.c_int
        self.lib.nvmlShutdown.argtypes = []
        self.lib.nvmlShutdown.restype = ctypes.c_int
        self._check(self.lib.nvmlInit_v2())
        try:
            count = ctypes.c_uint()
            self.lib.nvmlDeviceGetCount_v2.argtypes = [ctypes.POINTER(ctypes.c_uint)]
            self.lib.nvmlDeviceGetCount_v2.restype = ctypes.c_int
            self._check(self.lib.nvmlDeviceGetCount_v2(ctypes.byref(count)))
            self.lib.nvmlDeviceGetHandleByIndex_v2.argtypes = [
                ctypes.c_uint,
                ctypes.POINTER(ctypes.c_void_p),
            ]
            self.lib.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
            self.handles = []
            for index in range(count.value):
                handle = ctypes.c_void_p()
                self._check(
                    self.lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle))
                )
                self.handles.append(handle)
            if not self.handles:
                raise RuntimeError("no NVIDIA GPUs available for memory sampling")
            self.query = self.lib.nvmlDeviceGetComputeRunningProcesses_v3
            self.query.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint),
                ctypes.POINTER(_ProcessInfo),
            ]
            self.query.restype = ctypes.c_int
        except BaseException:
            self.lib.nvmlShutdown()
            raise
        self.done = threading.Event()
        self.thread: threading.Thread | None = None
        self.readings: list[tuple[float, int]] = []
        self.error: Exception | None = None

    @staticmethod
    def _check(code: int) -> None:
        if code:
            raise RuntimeError(f"NVML returned error {code}")

    def _sample(self) -> None:
        total = 0
        for handle in self.handles:
            capacity = 16
            while True:
                count = ctypes.c_uint(capacity)
                entries = (_ProcessInfo * capacity)()
                code = self.query(handle, ctypes.byref(count), entries)
                if code == 7:  # NVML_ERROR_INSUFFICIENT_SIZE: process list grew.
                    capacity = max(capacity * 2, count.value)
                    continue
                self._check(code)
                for entry in entries[: count.value]:
                    if entry.pid == os.getpid():
                        if entry.used_gpu_memory == 2**64 - 1:
                            raise RuntimeError("NVML process memory is unavailable")
                        total += entry.used_gpu_memory
                break
        self.readings.append((time.perf_counter(), total))

    def _poll(self) -> None:
        try:
            while not self.done.wait(self.interval):
                self._sample()
        except Exception as exc:
            self.error = exc
            self.done.set()

    def start(self) -> None:
        if self.thread is not None:
            raise RuntimeError("memory sampler is already running")
        self.readings = []
        self.error = None
        self.done.clear()
        self._sample()
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()

    def finish(self) -> dict[str, Any]:
        self.done.set()
        if self.thread is None:
            raise RuntimeError("memory sampler was not started")
        self.thread.join()
        self.thread = None
        if self.error is not None:
            raise self.error
        self._sample()
        return {
            "sampled_peak_gpu_process_bytes": max(n for _, n in self.readings),
            "samples": len(self.readings),
            "interval_seconds": self.interval,
            "max_observed_gap_seconds": max(
                (
                    b[0] - a[0]
                    for a, b in zip(self.readings, self.readings[1:], strict=False)
                ),
                default=0.0,
            ),
        }

    def close(self) -> None:
        self.done.set()
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        self._check(self.lib.nvmlShutdown())
