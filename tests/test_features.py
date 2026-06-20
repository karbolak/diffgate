import json

import numpy as np

from diffgate.features import PrefixFeatureExtractor, align_features, build_signal_sequences, load_feature_schema, save_feature_schema


def test_extract_from_latents():
    record = {
        "latents": [
            np.zeros((2, 2)),
            np.ones((2, 2)),
            np.ones((2, 2)) * 2,
        ]
    }
    features = PrefixFeatureExtractor(prefix_steps=3).extract(record)
    assert "latent_rms_mean" in features
    assert "latent_volatility_rms_mean" in features


def test_extract_from_sequences_json():
    record = {"latent_rms_json": json.dumps([1.0, 2.0, 3.0])}
    signals = build_signal_sequences(record, prefix_steps=3)
    assert signals["latent_rms"] == [1.0, 2.0, 3.0]
    features = PrefixFeatureExtractor(prefix_steps=3).extract(record)
    assert features["latent_rms_mean"] == 2.0


def test_feature_schema_roundtrip(tmp_path):
    features = {"a": 1.0, "b": 2.0}
    path = tmp_path / "schema.json"
    save_feature_schema(features, path)
    schema = load_feature_schema(path)
    assert schema == ["a", "b"]
    arr = align_features({"a": 1.0}, schema, fill_value=-1)
    assert arr.shape == (1, 2)
    assert arr[0, 1] == -1
