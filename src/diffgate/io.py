"""Input/output helpers for DiffGate runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .results import GenerationResult


def save_result(result: GenerationResult, output_dir: str | Path, image_name: str = "image.png") -> None:
    """Save a generation result to a directory.

    Files written:
    - ``result.json``: metadata and health score
    - ``features.json``: extracted prefix features
    - ``signals.json``: recorded signals, if present
    - ``image.png``: generated image, if generation completed
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if result.image is not None:
        result.image.save(out / image_name)

    (out / "result.json").write_text(
        json.dumps(result.to_dict(include_features=False, include_signals=False), indent=2, default=_json_default),
        encoding="utf-8",
    )
    (out / "features.json").write_text(
        json.dumps(result.features, indent=2, default=_json_default),
        encoding="utf-8",
    )
    if result.signals is not None:
        (out / "signals.json").write_text(
            json.dumps(result.signals, indent=2, default=_json_default),
            encoding="utf-8",
        )


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if hasattr(obj, "detach") and hasattr(obj, "cpu"):
        return obj.detach().cpu().tolist()
    return str(obj)
