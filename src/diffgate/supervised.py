"""Supervised health scorer support."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .features import align_features, load_feature_schema


class SupervisedHealthScorer:
    """Load a saved sklearn-style predictor and return a 0--100 health score."""

    def __init__(self, model: Any, feature_schema: list[str], scaler: Any | None = None):
        self.model = model
        self.feature_schema = list(feature_schema)
        self.scaler = scaler

    @classmethod
    def from_files(
        cls,
        model_path: str | Path,
        feature_schema_path: str | Path,
        scaler_path: str | Path | None = None,
    ) -> "SupervisedHealthScorer":
        try:
            import joblib
        except ImportError as exc:  # pragma: no cover
            raise ImportError("supervised mode requires joblib. Install diffgate[supervised].") from exc

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path) if scaler_path else None
        schema = load_feature_schema(feature_schema_path)
        return cls(model=model, scaler=scaler, feature_schema=schema)

    def score(self, features: Mapping[str, float]) -> float:
        x = align_features(features, self.feature_schema)
        if self.scaler is not None:
            x = self.scaler.transform(x)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(x)
            probability = float(proba[0][1] if not hasattr(proba, "shape") else proba[0, 1])
        elif hasattr(self.model, "decision_function"):
            import math

            decision = float(self.model.decision_function(x)[0])
            probability = 1.0 / (1.0 + math.exp(-decision))
        else:
            prediction = float(self.model.predict(x)[0])
            probability = max(0.0, min(1.0, prediction))
        return 100.0 * probability
