"""Result objects returned by DiffGate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationResult:
    """Output from one DiffGate generation attempt."""

    prompt: str
    seed: int | None
    aborted: bool
    health_score: float | None
    steps_used: int
    image: Any | None = None
    features: dict[str, float] = field(default_factory=dict)
    signals: dict[str, Any] | None = None
    abort_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_features: bool = True, include_signals: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "prompt": self.prompt,
            "seed": self.seed,
            "aborted": self.aborted,
            "health_score": self.health_score,
            "steps_used": self.steps_used,
            "has_image": self.image is not None,
            "abort_reason": self.abort_reason,
            "metadata": self.metadata,
        }
        if include_features:
            data["features"] = self.features
        if include_signals:
            data["signals"] = self.signals
        return data
