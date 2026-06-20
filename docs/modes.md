# DiffGate Modes

DiffGate separates **recording mode** from **scoring mode**.

## Recording mode

Recording mode determines which internal signals are captured during generation.

### `latent`

The `latent` mode uses only the standard Diffusers step-end callback.

Signals:

- latent RMS
- latent standard deviation
- latent absolute mean
- latent volatility
- latent update cosine

Advantages:

- simpler;
- more portable;
- easier to debug;
- does not depend on SD3.5 transformer internals.

Limitations:

- does not capture CFG divergence;
- does not capture denoiser prediction features;
- is not the full thesis feature set.

### `rich`

The `rich` mode uses the step-end callback and a wrapper around `pipe.transformer.forward`.

Signals:

- latent state statistics;
- latent update statistics;
- denoiser prediction statistics;
- denoiser prediction changes;
- CFG divergence;
- relative CFG divergence;
- CFG alignment;
- guided-minus-conditional RMS;
- update-prediction consistency residual;
- update-prediction cosine.

Advantages:

- closest to the thesis implementation;
- supports the full trajectory-health concept;
- records the signals needed for supervised prefix models trained on rich features.

Limitations:

- currently specific to Diffusers SD3/SD3.5-style pipelines;
- may require updates if Diffusers changes the internal transformer output format;
- requires local model execution.

## Scoring mode

Scoring mode determines how the prefix features become a health score.

### `training_free`

Uses fixed hand-designed feature weights.

This mode does not require labels or a trained model. Calibrated use should provide a JSON config containing feature statistics from a reference dataset.

### `supervised`

Uses a saved predictor, usually an sklearn logistic-regression model.

This mode requires:

- a saved model file;
- a feature schema JSON;
- optionally a saved scaler.

The supervised score is returned as `100 * P(good_generation)` when the model exposes `predict_proba`.

## Recommended combinations

For thesis-faithful testing:

```bash
--record-mode rich --scorer supervised
```

For training-free early-abort experiments:

```bash
--record-mode rich --scorer training_free
```

For laptop smoke tests:

```bash
--record-mode latent --scorer training_free --no-abort
```
