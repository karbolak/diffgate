# Hugging Face Access

DiffGate does not use an image-generation API.

It loads Stable Diffusion 3.5 Large locally through Hugging Face Diffusers:

```python
StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large")
```

If the model is gated, you may need to authenticate once:

```bash
huggingface-cli login
```

For cluster use, set a cache directory:

```bash
export HF_HOME=/scratch/$USER/huggingface
export HF_HUB_DISABLE_XET=1
```

Then run DiffGate with:

```bash
diffgate generate \
  --prompt "A red car on a snowy road" \
  --record-mode rich \
  --device cuda \
  --torch-dtype float16
```

After the model is cached, `--local-files-only` can be used to avoid downloads.
