"""Feature extraction from recorded diffusion trajectory signals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .signals import (
    cosine_similarity,
    latent_updates,
    normalized_residual,
    prefix_summary,
    tensor_abs_mean,
    tensor_rms,
    tensor_std,
    to_numpy,
)

DEFAULT_SIGNAL_NAMES = [
    "latent_rms",
    "latent_std",
    "latent_abs_mean",
    "latent_volatility_rms",
    "latent_update_cosine",
    "denoiser_pred_rms",
    "denoiser_pred_std",
    "denoiser_pred_abs_mean",
    "denoiser_pred_delta_rms",
    "denoiser_pred_cosine_prev",
    "cfg_divergence_rms",
    "cfg_divergence_abs_mean",
    "cfg_divergence_relative",
    "cfg_alignment_cosine",
    "guided_minus_cond_rms",
    "denoising_consistency_residual",
    "denoising_update_pred_cosine",
]

SUMMARY_NAMES = ["mean", "max", "std", "slope", "last"]


class PrefixFeatureExtractor:
    """Extract compact prefix features from trajectory records.

    The output naming scheme is ``<signal>_<summary>``, for example
    ``latent_rms_mean`` or ``cfg_divergence_relative_slope``.
    """

    def __init__(self, prefix_steps: int = 5, signal_names: list[str] | None = None):
        if prefix_steps <= 0:
            raise ValueError("prefix_steps must be positive")
        self.prefix_steps = int(prefix_steps)
        self.signal_names = list(signal_names or DEFAULT_SIGNAL_NAMES)

    def extract(self, record: Mapping[str, Any]) -> dict[str, float]:
        signals = build_signal_sequences(record, prefix_steps=self.prefix_steps)
        features: dict[str, float] = {}
        for signal_name in self.signal_names:
            values = signals.get(signal_name)
            if values is None:
                continue
            summary = prefix_summary(values[: self.prefix_steps])
            for stat_name, stat_value in summary.items():
                features[f"{signal_name}_{stat_name}"] = stat_value
        return features


def build_signal_sequences(record: Mapping[str, Any], prefix_steps: int | None = None) -> dict[str, list[float]]:
    """Build per-step scalar signal sequences from a raw or precomputed record."""

    signals: dict[str, list[float]] = {}

    # Preserve already-computed sequences and ``*_json`` sequences.
    for name in DEFAULT_SIGNAL_NAMES:
        value = record.get(name)
        if value is None:
            value = record.get(f"{name}_json")
        if value is not None:
            parsed = _parse_sequence(value)
            if parsed is not None:
                signals[name] = parsed

    latents = list(record.get("latents") or [])
    if latents:
        limit = prefix_steps if prefix_steps is not None else len(latents)
        latents = latents[:limit]
        signals.setdefault("latent_rms", [tensor_rms(z) for z in latents])
        signals.setdefault("latent_std", [tensor_std(z) for z in latents])
        signals.setdefault("latent_abs_mean", [tensor_abs_mean(z) for z in latents])

        updates = latent_updates(latents)
        signals.setdefault("latent_volatility_rms", [tensor_rms(u) for u in updates])
        if len(updates) >= 2:
            signals.setdefault(
                "latent_update_cosine",
                [cosine_similarity(updates[i], updates[i - 1]) for i in range(1, len(updates))],
            )

    preds = list(record.get("denoiser_predictions") or record.get("guided_predictions") or [])
    if preds:
        limit = prefix_steps if prefix_steps is not None else len(preds)
        preds = preds[:limit]
        signals.setdefault("denoiser_pred_rms", [tensor_rms(p) for p in preds])
        signals.setdefault("denoiser_pred_std", [tensor_std(p) for p in preds])
        signals.setdefault("denoiser_pred_abs_mean", [tensor_abs_mean(p) for p in preds])
        deltas = [to_numpy(preds[i]) - to_numpy(preds[i - 1]) for i in range(1, len(preds))]
        signals.setdefault("denoiser_pred_delta_rms", [tensor_rms(d) for d in deltas])
        if len(preds) >= 2:
            signals.setdefault(
                "denoiser_pred_cosine_prev",
                [cosine_similarity(preds[i], preds[i - 1]) for i in range(1, len(preds))],
            )

    cond = list(record.get("conditional_predictions") or [])
    uncond = list(record.get("unconditional_predictions") or [])
    guided = list(record.get("guided_predictions") or record.get("denoiser_predictions") or [])
    if cond and uncond:
        n = min(len(cond), len(uncond), prefix_steps or len(cond))
        cond = cond[:n]
        uncond = uncond[:n]
        diffs = [to_numpy(cond[i]) - to_numpy(uncond[i]) for i in range(n)]
        signals.setdefault("cfg_divergence_rms", [tensor_rms(d) for d in diffs])
        signals.setdefault("cfg_divergence_abs_mean", [tensor_abs_mean(d) for d in diffs])
        signals.setdefault("cfg_alignment_cosine", [cosine_similarity(cond[i], uncond[i]) for i in range(n)])
        if guided:
            guided = guided[: min(len(guided), n)]
            signals.setdefault(
                "cfg_divergence_relative",
                [tensor_rms(diffs[i]) / (tensor_rms(guided[i]) + 1e-8) for i in range(len(guided))],
            )
            signals.setdefault(
                "guided_minus_cond_rms",
                [tensor_rms(to_numpy(guided[i]) - to_numpy(cond[i])) for i in range(len(guided))],
            )

    if latents and preds:
        updates = latent_updates(latents)
        n = min(len(updates), len(preds))
        if n > 0:
            signals.setdefault(
                "denoising_consistency_residual",
                [_safe_residual(updates[i], preds[i]) for i in range(n)],
            )
            signals.setdefault(
                "denoising_update_pred_cosine",
                [_safe_cosine(updates[i], preds[i]) for i in range(n)],
            )

    return signals


def load_feature_schema(path: str | Path) -> list[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "features" in data:
        data = data["features"]
    if not isinstance(data, list):
        raise ValueError("feature schema must be a list or {'features': [...]} dict")
    return [str(x) for x in data]


def save_feature_schema(features: Mapping[str, float], path: str | Path) -> None:
    Path(path).write_text(
        json.dumps({"features": sorted(features.keys())}, indent=2),
        encoding="utf-8",
    )


def align_features(features: Mapping[str, float], schema: list[str], fill_value: float = 0.0) -> np.ndarray:
    row = [features.get(name, fill_value) for name in schema]
    return np.asarray(row, dtype=np.float64).reshape(1, -1)


def _parse_sequence(value: Any) -> list[float] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, (list, tuple, np.ndarray)):
        return None
    out: list[float] = []
    for item in value:
        try:
            f = float(item)
        except (TypeError, ValueError):
            continue
        out.append(f)
    return out


def _safe_residual(delta: Any, pred: Any) -> float:
    try:
        return normalized_residual(delta, pred)
    except Exception:
        return float("nan")


def _safe_cosine(a: Any, b: Any) -> float:
    try:
        return cosine_similarity(a, b)
    except Exception:
        return float("nan")
