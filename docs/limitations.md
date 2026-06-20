# Limitations

DiffGate is a research prototype.

The current implementation is based on experiments conducted with:

- Stable Diffusion 3.5 Large
- 25 denoising steps
- CFG scale 7.0
- 1024x1024 resolution

Performance may not transfer to:

- other diffusion models,
- other schedulers,
- other guidance scales,
- other prompt distributions.

The supervised predictor was trained on automatically scored images and should not be interpreted as a replacement for human evaluation.

DiffGate should not be considered a production-quality quality-control system.