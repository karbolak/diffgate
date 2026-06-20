"""Diffusers loading helpers for DiffGate."""

from __future__ import annotations

from typing import Any


def resolve_torch_dtype(dtype: str | None) -> Any | None:
    """Resolve a CLI dtype string to a torch dtype without importing torch at package import."""

    if dtype is None or dtype == "auto":
        return None
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("torch is required when specifying torch_dtype") from exc

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"unknown torch dtype: {dtype}")
    return mapping[dtype]


def load_sd35_pipeline(
    model_id: str = "stabilityai/stable-diffusion-3.5-large",
    torch_dtype: Any | None = None,
    device: str | None = None,
    **kwargs: Any,
) -> Any:
    """Load Stable Diffusion 3/3.5 through Diffusers.

    This is local model loading, not an image-generation API call. A Hugging
    Face token may be needed to download gated model weights the first time.
    """

    try:
        from diffusers import StableDiffusion3Pipeline
    except ImportError as exc:  # pragma: no cover
        raise ImportError("SD3.5 generation requires diffusers. Install diffgate[sd35].") from exc

    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch_dtype, **kwargs)
    if device is not None:
        pipe = pipe.to(device)
    return pipe


def make_torch_generator(seed: int | None, device: Any | None = None) -> Any | None:
    if seed is None:
        return None
    try:
        import torch
    except ImportError:  # pragma: no cover
        return None
    try:
        return torch.Generator(device=device).manual_seed(int(seed))
    except Exception:
        return torch.Generator().manual_seed(int(seed))
