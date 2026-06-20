import math

import numpy as np

from diffgate.signals import (
    cosine_similarity,
    latent_updates,
    linear_slope,
    normalized_residual,
    prefix_summary,
    tensor_abs_mean,
    tensor_rms,
    tensor_std,
)


def test_tensor_summaries():
    x = np.array([1.0, -2.0, 3.0])
    assert math.isclose(tensor_rms(x), math.sqrt((1 + 4 + 9) / 3))
    assert math.isclose(tensor_abs_mean(x), 2.0)
    assert tensor_std(x) > 0


def test_cosine_similarity():
    assert cosine_similarity(np.array([1, 0]), np.array([1, 0])) > 0.999
    assert abs(cosine_similarity(np.array([1, 0]), np.array([0, 1]))) < 1e-6


def test_latent_updates():
    latents = [np.array([0, 0]), np.array([1, 2]), np.array([2, 2])]
    updates = latent_updates(latents)
    assert len(updates) == 2
    assert np.allclose(updates[0], [1, 2])
    assert np.allclose(updates[1], [1, 0])


def test_normalized_residual_perfect_alignment():
    delta = np.array([2.0, 0.0])
    pred = np.array([1.0, 0.0])
    assert normalized_residual(delta, pred) < 1e-6


def test_linear_slope_and_prefix_summary():
    assert math.isclose(linear_slope([1, 2, 3]), 1.0)
    summary = prefix_summary([1, 2, 3])
    assert summary["mean"] == 2.0
    assert summary["last"] == 3.0
