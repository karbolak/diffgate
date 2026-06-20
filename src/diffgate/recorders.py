"""Trajectory recorders for DiffGate.

DiffGate provides two recording modes:

``latent``
    Uses only the standard Diffusers step-end callback and records latent-state
    behaviour. This mode is portable and useful for smoke tests.

``rich``
    Uses the callback plus a forward hook around ``pipe.transformer.forward``.
    This is the thesis-faithful SD3.5 mode and records denoiser and CFG-related
    signals when Diffusers exposes them in the transformer output.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .signals import (
    cosine_similarity,
    latent_updates,
    linear_slope,
    normalized_residual,
    safe_max,
    safe_mean,
    safe_std,
    tensor_abs_mean,
    tensor_rms,
    tensor_std,
    to_numpy,
)


class AbortGeneration(RuntimeError):
    """Internal control-flow exception used to stop generation early."""

    def __init__(self, health_score: float, steps_used: int, features: dict[str, float]):
        super().__init__(f"Generation aborted at step {steps_used} with health score {health_score:.3f}")
        self.health_score = float(health_score)
        self.steps_used = int(steps_used)
        self.features = dict(features)


@dataclass
class LatentRecorder:
    """Callback-only recorder for latent trajectory signals."""

    early_steps: int = 5
    timesteps: list[int] = field(default_factory=list)
    step_times_sec: list[float] = field(default_factory=list)
    latents: list[Any] = field(default_factory=list)
    latent_rms: list[float] = field(default_factory=list)
    latent_std: list[float] = field(default_factory=list)
    latent_abs_mean: list[float] = field(default_factory=list)
    latent_volatility_rms: list[float] = field(default_factory=list)
    latent_update_cosine: list[float] = field(default_factory=list)

    _prev_latent: Any | None = None
    _prev_delta: Any | None = None
    _last_callback_time: float | None = None

    def callback(self, step_index: int, timestep: int, latents: Any) -> None:
        """Record latents at a Diffusers step-end callback."""

        if step_index >= self.early_steps:
            return

        now = time.perf_counter()
        if self._last_callback_time is not None:
            self.step_times_sec.append(float(now - self._last_callback_time))
        self._last_callback_time = now

        lat = _detach_clone_cpu(latents)
        self.latents.append(lat)
        self.timesteps.append(int(timestep))
        self.latent_rms.append(tensor_rms(lat))
        self.latent_std.append(tensor_std(lat))
        self.latent_abs_mean.append(tensor_abs_mean(lat))

        if self._prev_latent is not None:
            delta = to_numpy(lat) - to_numpy(self._prev_latent)
            self.latent_volatility_rms.append(tensor_rms(delta))
            if self._prev_delta is not None:
                self.latent_update_cosine.append(cosine_similarity(delta, self._prev_delta))
            self._prev_delta = delta

        self._prev_latent = lat

    def to_record(self, include_raw_latents: bool = False) -> dict[str, Any]:
        record: dict[str, Any] = {
            "timesteps": list(self.timesteps),
            "step_times_sec": list(self.step_times_sec),
            "latent_rms": list(self.latent_rms),
            "latent_std": list(self.latent_std),
            "latent_abs_mean": list(self.latent_abs_mean),
            "latent_volatility_rms": list(self.latent_volatility_rms),
            "latent_update_cosine": list(self.latent_update_cosine),
        }
        if include_raw_latents:
            record["latents"] = list(self.latents)
        return record

    def to_summary_row(self) -> dict[str, Any]:
        return _summary_row_from_record(self.to_record(include_raw_latents=False))


@dataclass
class RichSD35Recorder(LatentRecorder):
    """Recorder matching the rich thesis signal-capture script for SD3.5.

    It records latent signals from the callback and denoiser/CFG signals from a
    wrapper around ``pipe.transformer.forward``.
    """

    guidance_scale: float = 7.0

    denoiser_pred_rms: list[float] = field(default_factory=list)
    denoiser_pred_std: list[float] = field(default_factory=list)
    denoiser_pred_abs_mean: list[float] = field(default_factory=list)
    denoiser_pred_delta_rms: list[float] = field(default_factory=list)
    denoiser_pred_cosine_prev: list[float] = field(default_factory=list)

    cfg_divergence_rms: list[float] = field(default_factory=list)
    cfg_divergence_abs_mean: list[float] = field(default_factory=list)
    cfg_divergence_relative: list[float] = field(default_factory=list)
    cfg_alignment_cosine: list[float] = field(default_factory=list)
    guided_minus_cond_rms: list[float] = field(default_factory=list)
    cfg_chunk_count: list[int] = field(default_factory=list)

    denoising_consistency_residual: list[float] = field(default_factory=list)
    denoising_update_pred_cosine: list[float] = field(default_factory=list)

    _hook_step_count: int = 0
    _prev_pred: Any | None = None
    _last_pred: Any | None = None

    def transformer_hook(self, output: Any) -> None:
        """Record denoiser and CFG signals from a transformer forward output."""

        if self._hook_step_count >= self.early_steps:
            self._hook_step_count += 1
            return

        try:
            pred = _extract_prediction_tensor(output)
            if pred is None:
                return

            pred_np = to_numpy(pred)
            if pred_np.ndim == 0:
                return

            chunk_count = int(pred_np.shape[0]) if pred_np.ndim >= 1 else 1
            self.cfg_chunk_count.append(chunk_count)

            pred_for_consistency = pred_np
            if chunk_count >= 2:
                uncond = pred_np[0:1]
                cond = pred_np[1:2]
                cfg_delta = cond - uncond
                guided = uncond + self.guidance_scale * cfg_delta

                self.cfg_divergence_rms.append(tensor_rms(cfg_delta))
                self.cfg_divergence_abs_mean.append(tensor_abs_mean(cfg_delta))
                self.cfg_alignment_cosine.append(cosine_similarity(uncond, cond))
                self.guided_minus_cond_rms.append(tensor_rms(guided - cond))
                self.cfg_divergence_relative.append(tensor_rms(cfg_delta) / (tensor_rms(guided) + 1e-8))

                pred_for_consistency = guided
            else:
                self.cfg_divergence_rms.append(np.nan)
                self.cfg_divergence_abs_mean.append(np.nan)
                self.cfg_alignment_cosine.append(np.nan)
                self.guided_minus_cond_rms.append(np.nan)
                self.cfg_divergence_relative.append(np.nan)

            self.denoiser_pred_rms.append(tensor_rms(pred_for_consistency))
            self.denoiser_pred_std.append(tensor_std(pred_for_consistency))
            self.denoiser_pred_abs_mean.append(tensor_abs_mean(pred_for_consistency))

            if self._prev_pred is not None:
                try:
                    self.denoiser_pred_delta_rms.append(
                        tensor_rms(pred_for_consistency - to_numpy(self._prev_pred))
                    )
                    self.denoiser_pred_cosine_prev.append(
                        cosine_similarity(pred_for_consistency, self._prev_pred)
                    )
                except Exception:
                    self.denoiser_pred_delta_rms.append(np.nan)
                    self.denoiser_pred_cosine_prev.append(np.nan)

            self._prev_pred = np.array(pred_for_consistency, copy=True)
            self._last_pred = np.array(pred_for_consistency, copy=True)
        except Exception:
            # Signal logging must not crash generation.
            pass
        finally:
            self._hook_step_count += 1

    def callback(self, step_index: int, timestep: int, latents: Any) -> None:
        if step_index >= self.early_steps:
            return

        previous_latent = self._prev_latent
        super().callback(step_index, timestep, latents)

        if previous_latent is not None and self._last_pred is not None and self._prev_latent is not None:
            try:
                delta = to_numpy(self._prev_latent) - to_numpy(previous_latent)
                self.denoising_consistency_residual.append(normalized_residual(delta, self._last_pred))
                self.denoising_update_pred_cosine.append(cosine_similarity(delta, self._last_pred))
            except Exception:
                self.denoising_consistency_residual.append(np.nan)
                self.denoising_update_pred_cosine.append(np.nan)

    def to_record(self, include_raw_latents: bool = False) -> dict[str, Any]:
        record = super().to_record(include_raw_latents=include_raw_latents)
        record.update(
            {
                "denoiser_pred_rms": list(self.denoiser_pred_rms),
                "denoiser_pred_std": list(self.denoiser_pred_std),
                "denoiser_pred_abs_mean": list(self.denoiser_pred_abs_mean),
                "denoiser_pred_delta_rms": list(self.denoiser_pred_delta_rms),
                "denoiser_pred_cosine_prev": list(self.denoiser_pred_cosine_prev),
                "cfg_divergence_rms": list(self.cfg_divergence_rms),
                "cfg_divergence_abs_mean": list(self.cfg_divergence_abs_mean),
                "cfg_divergence_relative": list(self.cfg_divergence_relative),
                "cfg_alignment_cosine": list(self.cfg_alignment_cosine),
                "guided_minus_cond_rms": list(self.guided_minus_cond_rms),
                "cfg_chunk_count": list(self.cfg_chunk_count),
                "denoising_consistency_residual": list(self.denoising_consistency_residual),
                "denoising_update_pred_cosine": list(self.denoising_update_pred_cosine),
            }
        )
        return record


def _extract_prediction_tensor(output: Any) -> Any | None:
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    if hasattr(output, "sample"):
        return output.sample
    if hasattr(output, "detach") or isinstance(output, np.ndarray):
        return output
    return None


def _detach_clone_cpu(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "to"):
        try:
            return value.detach().to("cpu").clone()
        except Exception:
            return value.detach().cpu().clone()
    return np.array(value, copy=True)


def _summary_row_from_record(record: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for name, values in record.items():
        if name in {"latents"}:
            continue
        if isinstance(values, list):
            row[f"{name}_json"] = json.dumps(_jsonable(values))
            if values and not isinstance(values[0], (list, dict, tuple)):
                row[f"{name}_mean"] = round(safe_mean(values), 8)
                row[f"{name}_max"] = round(safe_max(values), 8)
                row[f"{name}_std"] = round(safe_std(values), 8)
                row[f"{name}_slope"] = round(linear_slope(values), 8)
    return row


def _jsonable(values: list[Any]) -> list[Any]:
    out: list[Any] = []
    for value in values:
        try:
            if isinstance(value, np.ndarray):
                out.append(value.tolist())
            elif isinstance(value, (np.floating, np.integer)):
                out.append(float(value))
            else:
                out.append(value)
        except Exception:
            out.append(str(value))
    return out
