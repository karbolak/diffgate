# Thesis Reproducibility

This repository accompanies the Bachelor thesis:

"Early Signals Predict Quality in Diffusion Image Generation"

University of Groningen
BSc Artificial Intelligence

Author:
Kajetan Karbowski

Supervisor:
Ivo De Jong

The experiments used:

- Stable Diffusion 3.5 Large
- 7,320 prompts
- 5 seeds per prompt
- 36,600 generated images
- Prefix length: 5 denoising steps

Quality was evaluated using:

- CLIP Score
- HPSv2
- ImageReward

The repository provides an implementation of the trajectory-based selective continuation framework described in the thesis.