# DiffGate

**Early trajectory assessment and selective continuation for diffusion image generation.**

DiffGate is a research prototype that wraps Stable Diffusion 3.5 Large generation, records early denoising trajectory signals, computes a trajectory health score, and optionally aborts generation after a short prefix.

It supports two recording modes:

- `latent`: callback-only latent trajectory logging;
- `rich`: thesis-faithful SD3.5 logging with latent, denoiser, classifier-free guidance, and consistency signals.

It supports two scoring modes:

- `training_free`: a hand-designed trajectory health score;
- `supervised`: a saved sklearn-style prefix-quality predictor.

DiffGate is released as a research and reproducibility artifact for a Bachelor thesis on early quality prediction in text-to-image diffusion. It is not a production-quality filtering system.

---

## Installation

For local development and unit tests:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

For real Stable Diffusion 3.5 generation:

```bash
pip install -e ".[sd35,supervised]"
```

The SD3.5 model weights are loaded locally through Hugging Face Diffusers. This is not an image-generation API call. A Hugging Face token may be required the first time if the model is gated.

---

## Python quickstart

```python
from diffgate import DiffGateSD35

gate = DiffGateSD35.from_pretrained(
    record_mode="rich",
    scorer="training_free",
    threshold=25,
    prefix_steps=5,
    device="cuda",
    torch_dtype="float16",
)

result = gate.generate(
    prompt="A red car on a snowy road",
    seed=12345,
    abort=True,
    return_trajectory=True,
)

print(result.health_score)
print(result.aborted)
```

---

## CLI quickstart

Training-free rich SD3.5 mode:

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

Latent-only mode:

```bash
diffgate generate \
  --prompt "A red car on a snowy road" \
  --record-mode latent \
  --scorer training_free \
  --threshold 25 \
  --device cuda
```

Score saved features without running generation:

```bash
diffgate score-features --features outputs/example/features.json
```

Extract features from saved signals:

```bash
diffgate extract-features \
  --signals outputs/example/signals.json \
  --prefix-steps 5 \
  --output outputs/example/features_from_signals.json
```

See `docs/cli.md` and `docs/modes.md` for details.

---

## Status

DiffGate is a research prototype. The default training-free score is usable as a diagnostic, but calibrated reproduction should use saved feature statistics from the thesis dataset. The supervised mode requires exported model artifacts.

---

## Citation

```bibtex
@software{karbowski2026diffgate,
  author = {Karbowski, Kajetan},
  title = {DiffGate},
  year = {2026},
  url = {https://github.com/karbolak/DiffGate},
  note = {Research prototype for early trajectory assessment in diffusion image generation}
}
```

---

## License

MIT License.
