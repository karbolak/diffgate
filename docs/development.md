# Development

## Local laptop development

Your laptop can run unit tests and non-generation commands:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

This does not require Stable Diffusion 3.5 Large.

## GPU / cluster testing

Real SD3.5 generation should be run on a CUDA GPU:

```bash
pip install -e ".[sd35,supervised]"
```

Then:

```bash
diffgate generate \
  --prompt "A red car on a snowy road" \
  --record-mode rich \
  --scorer training_free \
  --threshold 25 \
  --device cuda \
  --torch-dtype float16 \
  --save-trajectory \
  --output-dir outputs/smoke_test
```

## Testing strategy

The test suite avoids loading large diffusion models. It tests:

- numerical signal helpers;
- feature extraction;
- training-free scoring;
- supervised scorer wrapper with a dummy model;
- recorders using NumPy arrays;
- result saving.

Slow integration tests with SD3.5 should be added separately and marked with `pytest.mark.slow`.
