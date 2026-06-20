# Thesis Reproducibility

DiffGate accompanies the Bachelor thesis:

**Early Signals Predict Quality in Image Generation**

Author: Kajetan Karbowski  
Supervisor: Ivo De Jong  
University of Groningen, BSc Artificial Intelligence

The thesis experiments used:

- Stable Diffusion 3.5 Large;
- 7,320 prompts;
- 5 seeds per prompt;
- 36,600 generated images;
- 25 denoising steps;
- CFG scale 7.0;
- prefix length 5.

Quality was evaluated with:

- CLIP;
- HPSv2;
- ImageReward.

DiffGate implements the trajectory recording and early gating mechanism used by the thesis. The saved supervised predictors and calibrated training-free statistics are not included by default; place them in `models/` and pass them through the CLI options described in `docs/cli.md`.
