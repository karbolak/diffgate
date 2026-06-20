import numpy as np

from diffgate.recorders import LatentRecorder, RichSD35Recorder


def test_latent_recorder_callback():
    rec = LatentRecorder(early_steps=3)
    rec.callback(0, 10, np.ones((1, 2, 2)))
    rec.callback(1, 9, np.ones((1, 2, 2)) * 2)
    record = rec.to_record()
    assert len(record["latent_rms"]) == 2
    assert len(record["latent_volatility_rms"]) == 1


def test_rich_recorder_transformer_hook_cfg():
    rec = RichSD35Recorder(early_steps=2, guidance_scale=7.0)
    # batch dimension 2: uncond and cond
    pred = np.stack([np.ones((2, 2)), np.ones((2, 2)) * 2], axis=0)
    rec.transformer_hook(pred)
    record = rec.to_record()
    assert record["cfg_chunk_count"] == [2]
    assert record["cfg_divergence_rms"][0] > 0
    assert record["denoiser_pred_rms"][0] > 0
