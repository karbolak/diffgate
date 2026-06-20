"""High-level DiffGate wrapper around Stable Diffusion 3.5 generation."""

from __future__ import annotations

from typing import Any

from .abort import ThresholdAbortPolicy
from .config import DiffGateConfig
from .features import PrefixFeatureExtractor
from .health import TrainingFreeHealthScorer
from .recorders import AbortGeneration, LatentRecorder, RichSD35Recorder
from .results import GenerationResult
from .supervised import SupervisedHealthScorer
from .wrapper import load_sd35_pipeline, make_torch_generator, resolve_torch_dtype


class DiffGateSD35:
    """Stable Diffusion 3.5 wrapper with early trajectory assessment.

    Parameters are intentionally close to the thesis setup, but the public API
    separates recording from scoring:

    - ``record_mode='latent'`` uses only the Diffusers callback.
    - ``record_mode='rich'`` also hooks ``pipe.transformer.forward`` to record
      denoiser and CFG signals.
    - ``scorer='training_free'`` uses a hand-designed health score.
    - ``scorer='supervised'`` uses a saved sklearn-style predictor.
    """

    def __init__(
        self,
        pipe: Any,
        scorer: TrainingFreeHealthScorer | SupervisedHealthScorer,
        policy: ThresholdAbortPolicy,
        config: DiffGateConfig | None = None,
    ) -> None:
        self.pipe = pipe
        self.scorer = scorer
        self.policy = policy
        self.config = config or DiffGateConfig(threshold=policy.threshold)
        self.extractor = PrefixFeatureExtractor(prefix_steps=self.config.prefix_steps)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "stabilityai/stable-diffusion-3.5-large",
        scorer: str = "training_free",
        mode: str | None = None,
        record_mode: str = "latent",
        threshold: float = 25.0,
        prefix_steps: int = 5,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.0,
        health_config_path: str | None = None,
        predictor_path: str | None = None,
        scaler_path: str | None = None,
        feature_schema_path: str | None = None,
        torch_dtype: str | Any | None = "auto",
        device: str | None = None,
        **kwargs: Any,
    ) -> "DiffGateSD35":
        """Load a local Diffusers SD3/SD3.5 pipeline and wrap it with DiffGate."""

        if mode is not None:
            scorer = mode  # compatibility with older README drafts

        record_mode = _normalise_record_mode(record_mode)
        scorer = _normalise_scorer(scorer)
        dtype = resolve_torch_dtype(torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
        pipe = load_sd35_pipeline(model_id=model_id, torch_dtype=dtype, device=device, **kwargs)

        config = DiffGateConfig(
            prefix_steps=prefix_steps,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            threshold=threshold,
            scorer=scorer,  # type: ignore[arg-type]
            record_mode=record_mode,  # type: ignore[arg-type]
        )

        if scorer == "supervised":
            if predictor_path is None or feature_schema_path is None:
                raise ValueError("supervised scorer requires predictor_path and feature_schema_path")
            scorer_obj: TrainingFreeHealthScorer | SupervisedHealthScorer = SupervisedHealthScorer.from_files(
                model_path=predictor_path,
                scaler_path=scaler_path,
                feature_schema_path=feature_schema_path,
            )
        else:
            scorer_obj = (
                TrainingFreeHealthScorer.from_json(health_config_path)
                if health_config_path
                else TrainingFreeHealthScorer()
            )

        return cls(pipe=pipe, scorer=scorer_obj, policy=ThresholdAbortPolicy(threshold), config=config)

    def generate(
        self,
        prompt: str,
        seed: int | None = None,
        abort: bool = True,
        return_trajectory: bool = False,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate an image and optionally abort after the configured prefix."""

        recorder = self._make_recorder()
        latest_features: dict[str, float] = {}
        latest_health: float | None = None
        original_forward = None

        generator = kwargs.pop("generator", None)
        if generator is None and seed is not None:
            generator = make_torch_generator(seed, device=getattr(self.pipe, "device", None))

        if self.config.record_mode == "rich":
            original_forward = self._attach_transformer_hook(recorder)

        def callback_on_step_end(pipe: Any, step: int, timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
            nonlocal latest_features, latest_health
            latents = callback_kwargs.get("latents")
            if latents is not None:
                recorder.callback(step, int(timestep), latents)

            steps_used = step + 1
            if steps_used == self.config.prefix_steps:
                latest_features = self.extractor.extract(recorder.to_record(include_raw_latents=False))
                latest_health = self.scorer.score(latest_features)
                if abort and self.policy.should_abort(latest_health):
                    raise AbortGeneration(latest_health, steps_used, latest_features)
            return callback_kwargs

        call_kwargs = {
            "prompt": prompt,
            "num_inference_steps": kwargs.pop("num_inference_steps", self.config.num_inference_steps),
            "guidance_scale": kwargs.pop("guidance_scale", self.config.guidance_scale),
            "generator": generator,
            "callback_on_step_end": callback_on_step_end,
            "callback_on_step_end_tensor_inputs": ["latents"],
        }
        call_kwargs.update(kwargs)

        try:
            output = self.pipe(**call_kwargs)
        except AbortGeneration as exc:
            return GenerationResult(
                prompt=prompt,
                seed=seed,
                aborted=True,
                health_score=exc.health_score,
                steps_used=exc.steps_used,
                image=None,
                features=exc.features,
                signals=recorder.to_record(include_raw_latents=False) if return_trajectory else None,
                abort_reason="health_score_below_threshold",
                metadata=self._metadata(),
            )
        finally:
            if original_forward is not None:
                self.pipe.transformer.forward = original_forward

        if not latest_features:
            latest_features = self.extractor.extract(recorder.to_record(include_raw_latents=False))
        if latest_health is None:
            latest_health = self.scorer.score(latest_features)

        image = output.images[0] if hasattr(output, "images") and output.images else None
        return GenerationResult(
            prompt=prompt,
            seed=seed,
            aborted=False,
            health_score=latest_health,
            steps_used=len(recorder.timesteps),
            image=image,
            features=latest_features,
            signals=recorder.to_record(include_raw_latents=False) if return_trajectory else None,
            metadata=self._metadata(),
        )

    def _make_recorder(self) -> LatentRecorder | RichSD35Recorder:
        if self.config.record_mode == "rich":
            return RichSD35Recorder(
                early_steps=self.config.prefix_steps,
                guidance_scale=self.config.guidance_scale,
            )
        return LatentRecorder(early_steps=self.config.prefix_steps)

    def _attach_transformer_hook(self, recorder: LatentRecorder | RichSD35Recorder) -> Any:
        if not hasattr(self.pipe, "transformer") or not hasattr(self.pipe.transformer, "forward"):
            raise RuntimeError("rich record_mode requires a pipeline with pipe.transformer.forward")
        if not isinstance(recorder, RichSD35Recorder):
            raise RuntimeError("rich record_mode requires RichSD35Recorder")
        original_forward = self.pipe.transformer.forward

        def wrapped_forward(*args: Any, **kwargs: Any) -> Any:
            output = original_forward(*args, **kwargs)
            recorder.transformer_hook(output)
            return output

        self.pipe.transformer.forward = wrapped_forward
        return original_forward

    def _metadata(self) -> dict[str, Any]:
        return {
            "threshold": self.policy.threshold,
            "prefix_steps": self.config.prefix_steps,
            "num_inference_steps": self.config.num_inference_steps,
            "guidance_scale": self.config.guidance_scale,
            "record_mode": self.config.record_mode,
            "scorer": self.config.scorer,
        }


# Backwards-compatible alias from the first package sketch.
EarlyAbortSD35 = DiffGateSD35


def _normalise_record_mode(record_mode: str) -> str:
    aliases = {
        "latent_only": "latent",
        "callback": "latent",
        "callback_only": "latent",
        "full": "rich",
        "rich_sd35": "rich",
    }
    record_mode = aliases.get(record_mode, record_mode)
    if record_mode not in {"latent", "rich"}:
        raise ValueError("record_mode must be 'latent' or 'rich'")
    return record_mode


def _normalise_scorer(scorer: str) -> str:
    aliases = {
        "training-free": "training_free",
        "trainingfree": "training_free",
        "learned": "supervised",
    }
    scorer = aliases.get(scorer, scorer)
    if scorer not in {"training_free", "supervised"}:
        raise ValueError("scorer must be 'training_free' or 'supervised'")
    return scorer
