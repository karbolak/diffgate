# Feature Definitions

DiffGate extracts trajectory statistics from the early denoising process.

Current feature families include:

## Latent Movement

Measures the magnitude of latent updates between denoising steps.

## CFG Divergence

Measures disagreement between conditional and unconditional denoiser predictions.

## Relative CFG Divergence

CFG divergence normalized by latent magnitude.

## CFG Alignment

Cosine similarity between conditional and unconditional predictions.

## Denoiser Prediction Change

Measures variation in predicted clean-image estimates across steps.

## Update Consistency

Agreement between predicted and observed latent update directions.

For each trajectory signal, DiffGate computes summary statistics over a configurable prefix:

- mean
- slope
- final value

These compact summaries form the input to the health-scoring system.