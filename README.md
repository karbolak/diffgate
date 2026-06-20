# DiffGate

**Early trajectory assessment and selective continuation for diffusion image generation.**

DiffGate is an open-source research prototype that estimates image quality from the early stages of a diffusion trajectory and supports selective continuation through early-abort decisions.

The project accompanies a Bachelor thesis investigating whether information contained in the first few denoising steps of Stable Diffusion 3.5 Large can predict final image quality.

---

## Motivation

Diffusion models generate images through an iterative denoising process.

Image quality is typically unknown until generation is complete, meaning that compute and user waiting time can be spent on trajectories that ultimately produce low-quality outputs.

DiffGate explores whether generator-internal signals observed during the first few denoising steps can be used to:

- estimate final image quality,
- identify unpromising trajectories,
- support early-abort decisions,
- reduce unnecessary computation.

---

## Features

- Training-free trajectory health score
- Supervised prefix-quality predictor
- Early-abort decisions after a configurable denoising prefix
- Stable Diffusion 3.5 Large integration
- Trajectory logging and feature extraction
- Reproducible research implementation

---

## Installation

```bash
pip install diffgate
```

Or install from source:

```bash
git clone https://github.com/karbolak/diffgate.git
cd DiffGate
pip install -e .
```

---

## Example

```python
from diffgate import EarlyAbortSD35

pipe = EarlyAbortSD35.from_pretrained(
    mode="training_free",
    threshold=25,
)

result = pipe.generate(
    prompt="A red car on a snowy road",
    seed=12345,
)

print(result.health_score)
print(result.aborted)
```

---

## Status

DiffGate is a research prototype.

The package is intended as a reproducibility artifact accompanying academic work. It should not be considered a fully validated production-quality filtering system.

---

## Citation

If you use DiffGate, please cite:

```bibtex
@software{karbowski2026diffgate,
  author = {Karbowski, Kajetan},
  title = {DiffGate},
  year = {2026},
  url = {https://github.com/karbolak/diffgate}
}
```

---

## License

MIT License.
