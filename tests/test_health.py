from diffgate.health import TrainingFreeHealthScorer, split_feature_name


def test_split_feature_name():
    assert split_feature_name("latent_rms_mean") == ("latent_rms", "mean")
    assert split_feature_name("cfg_divergence_relative_slope") == ("cfg_divergence_relative", "slope")


def test_training_free_score_range():
    scorer = TrainingFreeHealthScorer()
    score = scorer.score(
        {
            "latent_volatility_rms_mean": 0.5,
            "denoising_update_pred_cosine_mean": 0.8,
            "denoising_consistency_residual_mean": 0.2,
        }
    )
    assert 0 <= score <= 100


def test_training_free_json(tmp_path):
    path = tmp_path / "score.json"
    path.write_text('{"signal_weights": {"latent_rms": -1.0}, "raw_score_min": -1, "raw_score_max": 1}')
    scorer = TrainingFreeHealthScorer.from_json(path)
    assert 0 <= scorer.score({"latent_rms_mean": 0.5}) <= 100
