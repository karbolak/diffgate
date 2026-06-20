from pathlib import Path

from diffgate.abort import ThresholdAbortPolicy
from diffgate.config import DiffGateConfig
from diffgate.io import save_result
from diffgate.results import GenerationResult


def test_abort_policy():
    policy = ThresholdAbortPolicy(25)
    assert policy.should_abort(10)
    assert not policy.should_abort(30)
    assert not policy.should_abort(None)


def test_config_validation():
    DiffGateConfig(prefix_steps=5, num_inference_steps=25, record_mode="rich")


def test_save_result(tmp_path: Path):
    result = GenerationResult(
        prompt="test",
        seed=1,
        aborted=True,
        health_score=10.0,
        steps_used=5,
        features={"a": 1.0},
        signals={"x": [1, 2, 3]},
    )
    save_result(result, tmp_path)
    assert (tmp_path / "result.json").exists()
    assert (tmp_path / "features.json").exists()
    assert (tmp_path / "signals.json").exists()
