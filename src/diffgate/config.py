"""Configuration objects for DiffGate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

HealthMode = Literal["training_free", "supervised"]
RecordMode = Literal["latent", "rich"]


@dataclass(slots=True)
class DiffGateConfig:
    """Configuration for early trajectory assessment."""

    prefix_steps: int = 5
    num_inference_steps: int = 25
    guidance_scale: float = 7.0
    threshold: float = 25.0
    scorer: HealthMode = "training_free"
    record_mode: RecordMode = "latent"

    def __post_init__(self) -> None:
        if self.prefix_steps <= 0:
            raise ValueError("prefix_steps must be positive")
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        if self.prefix_steps > self.num_inference_steps:
            raise ValueError("prefix_steps cannot exceed num_inference_steps")
        if not 0 <= self.threshold <= 100:
            raise ValueError("threshold must be between 0 and 100")
        if self.scorer not in {"training_free", "supervised"}:
            raise ValueError("scorer must be 'training_free' or 'supervised'")
        if self.record_mode not in {"latent", "rich"}:
            raise ValueError("record_mode must be 'latent' or 'rich'")


# Backwards-compatible name used in earlier drafts.
EarlyAbortConfig = DiffGateConfig
