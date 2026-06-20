"""Abort policies for selective continuation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ThresholdAbortPolicy:
    """Abort if the health score is below a fixed threshold on a 0--100 scale."""

    threshold: float = 25.0

    def __post_init__(self) -> None:
        if not 0 <= self.threshold <= 100:
            raise ValueError("threshold must be between 0 and 100")

    def should_abort(self, health_score: float | None) -> bool:
        if health_score is None:
            return False
        return float(health_score) < self.threshold
