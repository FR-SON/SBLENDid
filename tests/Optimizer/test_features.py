from src.Optimizer.features import compute_features
from src.DBHandler import DBHandler


class FakeDB:
    """Minimal stand-in: real tokenization, injected frequency table."""
    def __init__(self, freqs):
        self._freqs = freqs
    def clean_value_collection(self, values):
        return DBHandler.clean_value_collection(values)
    def get_token_frequencies(self, tokens):
        toks = self.clean_value_collection(set(tokens))
        return {t: self._freqs.get(t, 1) for t in toks}


def test_single_column_features():
    db = FakeDB({"apple": 10, "pear": 5})
    cols = [["apple", "pear", "apple"]]
    feats = compute_features(cols, db)
    assert feats[0] == 2
    assert feats[1] == 15.0
    assert feats[2] == 1


def test_two_column_geomean():
    db = FakeDB({"a": 4, "b": 9})
    cols = [["a", "a"], ["b", "b"]]
    feats = compute_features(cols, db)
    assert feats[2] == 2
    assert feats[1] == 6.0


def test_predict_runtime_uses_compute_features(monkeypatch):
    from src.Operators.Seekers.SingleColumnOverlap import SingleColumnOverlap
    from src.DBHandler import DBHandler

    monkeypatch.setattr(DBHandler, "USE_ML_OPTIMIZER", False)
    seeker = SingleColumnOverlap(["apple", "pear"], k=10)

    db = FakeDB({"apple": 10, "pear": 5})
    captured = {}
    seeker.model = type("M", (), {"predict": lambda self, X: [captured.setdefault("X", X[0]) or 0.0]})()
    seeker._cached_predicted_runtime = None

    cols = [["apple", "pear"]]
    seeker._predict_runtime(cols, db)
    assert captured["X"] == compute_features(cols, db)
