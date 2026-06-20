# Limitations

DiffGate is a research prototype.

The current implementation is designed around experiments with:

- Stable Diffusion 3.5 Large;
- 25 denoising steps;
- CFG scale 7.0;
- 1024x1024 resolution;
- prefix length 5.

Performance may not transfer to:

- other diffusion models;
- other schedulers;
- other guidance scales;
- other prompt distributions;
- other image resolutions.

The training-free score is heuristic. The supervised score depends on the training data and quality metric used to train the predictor.

DiffGate should not be considered a production quality-control system.
