#!/usr/bin/env python3
"""Optional accelerator utilities with safe CPU fallback.

This module keeps GPU dependencies optional. If CuPy/CUDA are unavailable,
callers automatically fall back to NumPy execution.
"""
from __future__ import annotations

from typing import Any, Tuple

import numpy as np

try:
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover - optional dependency path
    cp = None  # type: ignore


def detect_accelerator(requested: str = "auto") -> Tuple[str, str]:
    """Resolve requested accelerator to ('cpu'|'gpu', reason)."""
    mode = requested.lower()
    if mode not in {"auto", "cpu", "gpu"}:
        raise ValueError("accelerator must be one of: auto, cpu, gpu")
    if mode == "cpu":
        return "cpu", "forced_cpu"

    if cp is None:
        if mode == "gpu":
            return "cpu", "gpu_requested_but_cupy_unavailable"
        return "cpu", "auto_cpu_cupy_unavailable"

    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:  # pragma: no cover - runtime-dependent
        if mode == "gpu":
            return "cpu", f"gpu_requested_but_cuda_probe_failed:{exc.__class__.__name__}"
        return "cpu", f"auto_cpu_cuda_probe_failed:{exc.__class__.__name__}"

    if device_count > 0:
        return "gpu", f"cuda_devices={device_count}"

    if mode == "gpu":
        return "cpu", "gpu_requested_but_no_cuda_devices"
    return "cpu", "auto_cpu_no_cuda_devices"


def xp_for(accelerator: str):
    """Return array module for accelerator ('numpy' or 'cupy')."""
    if accelerator == "gpu" and cp is not None:
        return cp
    return np


def as_backend_array(
    array_like: Any,
    *,
    accelerator: str,
    dtype: Any | None = None,
):
    """Create array in backend memory."""
    xp = xp_for(accelerator)
    return xp.asarray(array_like, dtype=dtype)


def to_cpu_array(array_like: Any):
    """Convert backend array to NumPy array."""
    if cp is not None and isinstance(array_like, cp.ndarray):  # type: ignore[attr-defined]
        return cp.asnumpy(array_like)
    return np.asarray(array_like)


def rbf_kernel_backend(
    x_left: Any,
    x_right: Any | None = None,
    *,
    gamma: float,
    accelerator: str,
):
    """Compute RBF kernel on CPU/GPU backend and return NumPy array."""
    xp = xp_for(accelerator)
    x_l = as_backend_array(x_left, accelerator=accelerator)
    if x_right is None:
        x_r = x_l
    else:
        x_r = as_backend_array(x_right, accelerator=accelerator)

    left_norm = xp.sum(x_l * x_l, axis=1, keepdims=True)
    right_norm = xp.sum(x_r * x_r, axis=1, keepdims=True).T
    sq_dist = left_norm + right_norm - 2.0 * (x_l @ x_r.T)
    sq_dist = xp.maximum(sq_dist, 0.0)
    kernel = xp.exp(-gamma * sq_dist)
    return to_cpu_array(kernel)

