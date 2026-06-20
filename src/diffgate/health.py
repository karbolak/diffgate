"""Training-free health scoring."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np


# Signal-level weights. Feature-level names such as ``latent_rms_mean`` are
# mapped back to their signal name before scoring.
DEFAULT_SIGNAL_WEIGHTS: dict[str, float] = {
    # Productive movement and directional stability.
    "latent_volatility_rms": 0.30,
    "latent_update_cosine": 0.20,
    "denoiser_pred_delta_rms": 0.15,
    "denoiser_pred_cosine_prev": 0.20,
    "denoising_update_pred_cosine": 0.35,
    # Guidance terms.
    "cfg_alignment_cosine": 0.20,
    "cfg_divergence_rms": 0.10,
    "guided_minus_cond_rms": 0.05,
    # Penalties.
    "latent_rms": -0.15,
    "cfg_divergence_relative": -0.20,
    "denoising_consistency_residual": -0.40,
}

# Summary-level weights keep the mean/last/slope more important than auxiliary
# diagnostic max/std values.
SUMMARY_WEIGHTS: dict[str, float] = {
    "mean": 1.00,
    "last": 0.75,
    "slope": 0.50,
    "max": 0.25,
    "std": 0.25,
}


class TrainingFreeHealthScorer:
    """Hand-designed health score over prefix features.

    For calibrated thesis reproduction, pass a JSON file containing feature
    statistics and raw-score scaling parameters. Without those statistics, the
    scorer still works using a bounded transform, but scores should be treated
    as uncalibrated diagnostics.
    """

    def __init__(
        self,
        signal_weights: Mapping[str, float] | None = None,
        summary_weights: Mapping[str, float] | None = None,
        feature_stats: Mapping[str, Mapping[str, float]] | None = None,
        raw_score_min: float | None = None,
        raw_score_max: float | None = None,
    ) -> None:
        self.signal_weights = dict(signal_weights or DEFAULT_SIGNAL_WEIGHTS)
        self.summary_weights = dict(summary_weights or SUMMARY_WEIGHTS)
        self.feature_stats = dict(feature_stats or {})
        self.raw_score_min = raw_score_min
        self.raw_score_max = raw_score_max

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainingFreeHealthScorer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            signal_weights=data.get("signal_weights") or data.get("weights"),
            summary_weights=data.get("summary_weights"),
            feature_stats=data.get("feature_stats"),
            raw_score_min=data.get("raw_score_min"),
            raw_score_max=data.get("raw_score_max"),
        )

    def score(self, features: Mapping[str, float]) -> float:
        return self.scale_raw_score(self.raw_score(features))

    def raw_score(self, features: Mapping[str, float]) -> float:
        total = 0.0
        for feature_name, value in features.items():
            signal_name, summary_name = split_feature_name(feature_name)
            signal_weight = self.signal_weights.get(signal_name)
            if signal_weight is None:
                continue
            summary_weight = self.summary_weights.get(summary_name, 0.0)
            if summary_weight == 0.0:
                continue
            v = _finite_float(value)
            if v is None:
                continue
            total += signal_weight * summary_weight * self._standardise(feature_name, signal_name, v)
        return float(total)

    def _standardise(self, feature_name: str, signal_name: str, value: float) -> float:
        stats = self.feature_stats.get(feature_name) or self.feature_stats.get(signal_name)
        if stats:
            mean = float(stats.get("mean", 0.0))
            std = float(stats.get("std", 1.0))
            return float((value - mean) / (std + 1e-8))
        return float(np.tanh(value))

    def scale_raw_score(self, raw_score: float) -> float:
        if self.raw_score_min is not None and self.raw_score_max is not None:
            denom = self.raw_score_max - self.raw_score_min
            if abs(denom) < 1e-8:
                return 50.0
            return float(np.clip(100.0 * (raw_score - self.raw_score_min) / (denom + 1e-8), 0.0, 100.0))
        return float(100.0 / (1.0 + math.exp(-raw_score)))


def split_feature_name(feature_name: str) -> tuple[str, str]:
    for suffix in ("mean", "max", "std", "slope", "last"):
        needle = f"_{suffix}"
        if feature_name.endswith(needle):
            return feature_name[: -len(needle)], suffix
    return feature_name, "mean"


def _finite_float(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None
