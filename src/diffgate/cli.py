"""Command-line interface for DiffGate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .features import PrefixFeatureExtractor, build_signal_sequences
from .health import TrainingFreeHealthScorer
from .io import save_result
from .pipeline import DiffGateSD35


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="diffgate",
        description="DiffGate: early trajectory assessment and selective continuation for diffusion generation.",
    )
    subparsers = parser.add_subparsers(dest="command")

    generate = subparsers.add_parser("generate", help="Generate with DiffGate early assessment.")
    generate.add_argument("--prompt", required=True, help="Text prompt to generate.")
    generate.add_argument("--output-dir", default="outputs/diffgate_run", help="Directory for result files.")
    generate.add_argument("--model-id", default="stabilityai/stable-diffusion-3.5-large")
    generate.add_argument("--record-mode", choices=["latent", "rich"], default="latent")
    generate.add_argument("--scorer", choices=["training_free", "supervised"], default="training_free")
    generate.add_argument("--threshold", type=float, default=25.0)
    generate.add_argument("--prefix-steps", type=int, default=5)
    generate.add_argument("--num-inference-steps", type=int, default=25)
    generate.add_argument("--guidance-scale", type=float, default=7.0)
    generate.add_argument("--height", type=int, default=None)
    generate.add_argument("--width", type=int, default=None)
    generate.add_argument("--seed", type=int, default=None)
    generate.add_argument("--device", default=None, help="Example: cuda, cpu, cuda:0.")
    generate.add_argument("--torch-dtype", default="auto", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    generate.add_argument("--no-abort", action="store_true", help="Score at prefix but always continue generation.")
    generate.add_argument("--save-trajectory", action="store_true", help="Save recorded trajectory signals.")
    generate.add_argument("--health-config-path", default=None, help="JSON config for calibrated training-free score.")
    generate.add_argument("--predictor-path", default=None, help="Saved supervised predictor, e.g. joblib file.")
    generate.add_argument("--scaler-path", default=None, help="Optional saved scaler for supervised predictor.")
    generate.add_argument("--feature-schema-path", default=None, help="Feature schema JSON for supervised predictor.")
    generate.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory.")
    generate.add_argument("--local-files-only", action="store_true", help="Only use locally cached model files.")

    score = subparsers.add_parser("score-features", help="Score an existing features.json file.")
    score.add_argument("--features", required=True, help="Path to features JSON.")
    score.add_argument("--health-config-path", default=None)

    extract = subparsers.add_parser("extract-features", help="Extract features from a saved signals.json file.")
    extract.add_argument("--signals", required=True, help="Path to saved signals JSON.")
    extract.add_argument("--prefix-steps", type=int, default=5)
    extract.add_argument("--output", default=None, help="Optional output features JSON path.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        # Backwards-compatible default: allow `diffgate-generate --prompt ...`.
        if hasattr(args, "prompt"):
            args.command = "generate"
        else:
            parser.print_help()
            return

    if args.command == "generate":
        _run_generate(args)
    elif args.command == "score-features":
        _run_score_features(args)
    elif args.command == "extract-features":
        _run_extract_features(args)
    else:  # pragma: no cover
        parser.error(f"unknown command: {args.command}")


def _run_generate(args: argparse.Namespace) -> None:
    extra = {}
    if args.cache_dir:
        extra["cache_dir"] = args.cache_dir
    if args.local_files_only:
        extra["local_files_only"] = True

    gate = DiffGateSD35.from_pretrained(
        model_id=args.model_id,
        scorer=args.scorer,
        record_mode=args.record_mode,
        threshold=args.threshold,
        prefix_steps=args.prefix_steps,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        health_config_path=args.health_config_path,
        predictor_path=args.predictor_path,
        scaler_path=args.scaler_path,
        feature_schema_path=args.feature_schema_path,
        torch_dtype=args.torch_dtype,
        device=args.device,
        **extra,
    )

    call_kwargs = {}
    if args.height is not None:
        call_kwargs["height"] = args.height
    if args.width is not None:
        call_kwargs["width"] = args.width

    result = gate.generate(
        prompt=args.prompt,
        seed=args.seed,
        abort=not args.no_abort,
        return_trajectory=args.save_trajectory,
        **call_kwargs,
    )
    save_result(result, Path(args.output_dir))
    print(json.dumps(result.to_dict(include_features=False, include_signals=False), indent=2))


def _run_score_features(args: argparse.Namespace) -> None:
    features = json.loads(Path(args.features).read_text(encoding="utf-8"))
    scorer = (
        TrainingFreeHealthScorer.from_json(args.health_config_path)
        if args.health_config_path
        else TrainingFreeHealthScorer()
    )
    print(json.dumps({"health_score": scorer.score(features)}, indent=2))


def _run_extract_features(args: argparse.Namespace) -> None:
    signals = json.loads(Path(args.signals).read_text(encoding="utf-8"))
    # This call also validates that the saved signal keys are readable.
    build_signal_sequences(signals, prefix_steps=args.prefix_steps)
    features = PrefixFeatureExtractor(prefix_steps=args.prefix_steps).extract(signals)
    if args.output:
        Path(args.output).write_text(json.dumps(features, indent=2), encoding="utf-8")
    print(json.dumps(features, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
