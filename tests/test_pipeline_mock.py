import numpy as np

from diffgate.abort import ThresholdAbortPolicy
from diffgate.config import DiffGateConfig
from diffgate.health import TrainingFreeHealthScorer
from diffgate.pipeline import DiffGateSD35


class DummyOutput:
    def __init__(self):
        self.images = ["image"]


class DummyTransformer:
    def forward(self, *args, **kwargs):
        return np.stack([np.ones((2, 2)), np.ones((2, 2)) * 2], axis=0)


class DummyPipe:
    device = "cpu"

    def __init__(self):
        self.transformer = DummyTransformer()

    def __call__(self, **kwargs):
        callback = kwargs["callback_on_step_end"]
        for step in range(kwargs["num_inference_steps"]):
            # Simulate transformer call before callback, as Diffusers does during denoising.
            self.transformer.forward()
            callback(self, step, 10 - step, {"latents": np.ones((1, 2, 2)) * (step + 1)})
        return DummyOutput()


def test_pipeline_mock_latent_no_abort():
    gate = DiffGateSD35(
        pipe=DummyPipe(),
        scorer=TrainingFreeHealthScorer(),
        policy=ThresholdAbortPolicy(0),
        config=DiffGateConfig(prefix_steps=2, num_inference_steps=3, record_mode="latent"),
    )
    result = gate.generate("prompt", abort=True, return_trajectory=True)
    assert not result.aborted
    assert result.image == "image"
    assert result.signals is not None


def test_pipeline_mock_rich_records_cfg():
    gate = DiffGateSD35(
        pipe=DummyPipe(),
        scorer=TrainingFreeHealthScorer(),
        policy=ThresholdAbortPolicy(0),
        config=DiffGateConfig(prefix_steps=2, num_inference_steps=3, record_mode="rich"),
    )
    result = gate.generate("prompt", abort=False, return_trajectory=True)
    assert not result.aborted
    assert result.signals is not None
    assert result.signals["cfg_chunk_count"]
