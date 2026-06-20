# DiffGate Concept

DiffGate evaluates diffusion trajectories before image generation is complete.

The underlying idea is that useful information about final image quality appears early in the denoising process.

Instead of waiting for a trajectory to finish, DiffGate extracts trajectory signals during the first few denoising steps and computes a quality estimate.

Depending on the selected policy, the trajectory can then:

- continue normally,
- be logged for analysis,
- be aborted early.

DiffGate supports both a training-free trajectory health score and a supervised quality predictor.