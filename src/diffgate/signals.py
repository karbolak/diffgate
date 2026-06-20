"""Numerical signal helpers for diffusion trajectory analysis.

The functions in this module intentionally accept either NumPy arrays or
PyTorch tensors. PyTorch is not imported at module import time so that the core
package can be unit-tested on machines without a GPU installation.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np


def to_numpy(x: Any) -> np.ndarray:
    """Convert an array-like object or torch tensor to a float64 NumPy array."""

    if hasattr(x, "detach") and hasattr(x, "cpu"):
        x = x.detach().cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x, dtype=np.float64)


def finite_values(values: Iterable[Any]) -> list[float]:
    """Return finite floats from a mixed sequence."""

    out: list[float] = []
    for value in values:
        try:
            f = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def safe_mean(values: Iterable[Any]) -> float:
    vals = finite_values(values)
    return float(np.mean(vals)) if vals else 0.0


def safe_std(values: Iterable[Any]) -> float:
    vals = finite_values(values)
    return float(np.std(vals)) if vals else 0.0


def safe_max(values: Iterable[Any]) -> float:
    vals = finite_values(values)
    return float(np.max(vals)) if vals else 0.0


def safe_last(values: Iterable[Any]) -> float:
    vals = finite_values(values)
    return float(vals[-1]) if vals else 0.0


def linear_slope(values: Iterable[Any]) -> float:
    """Slope of a first-degree polynomial over finite values."""

    vals = finite_values(values)
    if len(vals) < 2:
        return 0.0
    x = np.arange(len(vals), dtype=np.float64)
    y = np.asarray(vals, dtype=np.float64)
    return float(np.polyfit(x, y, 1)[0])


def tensor_rms(x: Any) -> float:
    arr = to_numpy(x)
    return float(np.sqrt(np.mean(arr**2))) if arr.size else 0.0


def tensor_abs_mean(x: Any) -> float:
    arr = to_numpy(x)
    return float(np.mean(np.abs(arr))) if arr.size else 0.0


def tensor_std(x: Any) -> float:
    arr = to_numpy(x)
    return float(np.std(arr)) if arr.size else 0.0


def cosine_similarity(a: Any, b: Any, eps: float = 1e-8) -> float:
    av = to_numpy(a).reshape(-1)
    bv = to_numpy(b).reshape(-1)
    if av.shape != bv.shape:
        raise ValueError(f"cosine inputs must have the same shape, got {av.shape} and {bv.shape}")
    denom = np.linalg.norm(av) * np.linalg.norm(bv) + eps
    return float(np.dot(av, bv) / denom)


def latent_updates(latents: list[Any]) -> list[np.ndarray]:
    """Return step-to-step latent updates z_t - z_{t-1}."""

    return [to_numpy(latents[i]) - to_numpy(latents[i - 1]) for i in range(1, len(latents))]


def normalized_residual(delta: Any, pred: Any, eps: float = 1e-8) -> float:
    """Scheduler-agnostic update-prediction residual.

    Fits the scalar alpha for alpha * pred ≈ delta and returns
    ||delta - alpha pred|| / ||delta||.
    """

    d = to_numpy(delta).reshape(-1)
    p = to_numpy(pred).reshape(-1)
    if d.shape != p.shape:
        raise ValueError(f"residual inputs must have the same shape, got {d.shape} and {p.shape}")
    denom = float(np.dot(p, p) + eps)
    alpha = float(np.dot(d, p) / denom)
    residual = d - alpha * p
    return float(np.linalg.norm(residual) / (np.linalg.norm(d) + eps))


def denoising_consistency_residual(delta: Any, pred: Any, eps: float = 1e-8) -> float:
    """Alias used by older package code."""

    return normalized_residual(delta, pred, eps=eps)


def prefix_summary(values: Iterable[Any]) -> dict[str, float]:
    """Return the compact summary statistics used by DiffGate.

    The thesis mainly used mean, slope, and final prefix value. DiffGate also
    reports max and standard deviation because they are cheap, useful for logs,
    and match some of the original trajectory-generation scripts.
    """

    vals = finite_values(values)
    return {
        "mean": round(safe_mean(vals), 8),
        "max": round(safe_max(vals), 8),
        "std": round(safe_std(vals), 8),
        "slope": round(linear_slope(vals), 8),
        "last": round(safe_last(vals), 8),
    }
