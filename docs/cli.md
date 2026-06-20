# DiffGate CLI

The package installs two equivalent commands:

```bash
diffgate
diffgate-generate
```

The main command is:

```bash
diffgate generate [OPTIONS]
```

## Generation command

Basic form:

```bash
diffgate generate \
  --prompt "A red car on a snowy road" \
  --record-mode rich \
  --scorer training_free \
  --threshold 25 \
  --seed 12345 \
  --device cuda \
  --torch-dtype float16 \
  --output-dir outputs/example
```

This will write:

- `result.json`
- `features.json`
- `image.png`, if generation completed
- `signals.json`, if `--save-trajectory` is passed

## Recording modes

### `--record-mode latent`

Callback-only mode. Records latent state statistics through the standard Diffusers step-end callback.

Use this mode for:

- smoke tests;
- portability;
- quick debugging;
- running on pipelines where the transformer hook is unavailable.

Example:

```bash
diffgate generate \
  --prompt "A red car on a snowy road" \
  --record-mode latent \
  --scorer training_free \
  --device cuda
```

### `--record-mode rich`

Thesis-faithful SD3.5 mode. Records latents through the callback and wraps `pipe.transformer.forward` to capture denoiser and CFG-related signals.

Use this mode for:

- reproducing the thesis feature set;
- CFG divergence features;
- denoiser prediction features;
- update-prediction consistency features.

Example:

```bash
diffgate generate \
  --prompt "A red car on a snowy road" \
  --record-mode rich \
  --scorer training_free \
  --prefix-steps 5 \
  --num-inference-steps 25 \
  --guidance-scale 7.0 \
  --device cuda \
  --torch-dtype float16
```

## Scoring modes

### `--scorer training_free`

Uses a hand-designed trajectory health score.

Optional calibrated config:

```bash
diffgate generate \
  --prompt "A red car on a snowy road" \
  --scorer training_free \
  --health-config-path models/training_free_hpsv2.json
```

The config can contain feature statistics, signal weights, summary weights, and raw-score min/max values.

### `--scorer supervised`

Uses a saved sklearn-style model.

Required:

- `--predictor-path`
- `--feature-schema-path`

Optional:

- `--scaler-path`

Example:

```bash
diffgate generate \
  --prompt "A red car on a snowy road" \
  --record-mode rich \
  --scorer supervised \
  --predictor-path models/hpsv2_prefix5_logreg.joblib \
  --scaler-path models/hpsv2_prefix5_scaler.joblib \
  --feature-schema-path models/hpsv2_prefix5_schema.json \
  --threshold 25 \
  --device cuda \
  --torch-dtype float16
```

## Continue but do not abort

Use `--no-abort` to score the prefix but always continue generation:

```bash
diffgate generate \
  --prompt "A red car on a snowy road" \
  --record-mode rich \
  --no-abort \
  --save-trajectory \
  --output-dir outputs/analysis_only
```

This is useful for collecting features and validating thresholds.

## Offline / cached model use

After the model has been downloaded once, you can use:

```bash
diffgate generate \
  --prompt "A red car on a snowy road" \
  --local-files-only \
  --cache-dir /path/to/hf_cache
```

## Non-generation commands

Score an existing feature file:

```bash
diffgate score-features --features outputs/example/features.json
```

Extract features from saved signals:

```bash
diffgate extract-features \
  --signals outputs/example/signals.json \
  --prefix-steps 5 \
  --output outputs/example/features.json
```
