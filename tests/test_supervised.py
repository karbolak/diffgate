from diffgate.supervised import SupervisedHealthScorer


class DummyModel:
    def predict_proba(self, x):
        return [[0.2, 0.8]]


class DummyScaler:
    def transform(self, x):
        return x


def test_supervised_score():
    scorer = SupervisedHealthScorer(DummyModel(), feature_schema=["a", "b"], scaler=DummyScaler())
    score = scorer.score({"a": 1.0, "b": 2.0})
    assert score == 80.0
