# Feature Definitions

DiffGate extracts scalar sequences from the first denoising steps and summarises each sequence with:

- mean;
- max;
- standard deviation;
- slope;
- final prefix value.

## Latent features

- `latent_rms`: latent magnitude.
- `latent_std`: latent dispersion.
- `latent_abs_mean`: mean absolute latent value.
- `latent_volatility_rms`: RMS of step-to-step latent updates.
- `latent_update_cosine`: cosine similarity between successive latent updates.

## Denoiser features

- `denoiser_pred_rms`: magnitude of the denoiser prediction.
- `denoiser_pred_std`: dispersion of the denoiser prediction.
- `denoiser_pred_abs_mean`: mean absolute denoiser prediction value.
- `denoiser_pred_delta_rms`: change in denoiser prediction across steps.
- `denoiser_pred_cosine_prev`: cosine similarity to previous denoiser prediction.

## CFG features

- `cfg_divergence_rms`: magnitude of conditional-minus-unconditional prediction.
- `cfg_divergence_abs_mean`: mean absolute CFG difference.
- `cfg_divergence_relative`: CFG divergence divided by guided prediction magnitude.
- `cfg_alignment_cosine`: cosine similarity between conditional and unconditional predictions.
- `guided_minus_cond_rms`: magnitude of guided prediction minus conditional prediction.

## Consistency features

- `denoising_consistency_residual`: residual after fitting the denoiser prediction direction to the actual latent update.
- `denoising_update_pred_cosine`: cosine similarity between actual latent update and denoiser prediction.

## Example feature names

- `latent_rms_mean`
- `latent_rms_slope`
- `cfg_divergence_relative_mean`
- `denoising_consistency_residual_last`
