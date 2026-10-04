import pandas as pd
from src.Benchmark.recipes.correlation import _key_to_target, _spearman_abs


def test_key_to_target_aggregates():
    df = pd.DataFrame({"k": ["a", "a", "b"], "t": ["1", "3", "10"]})
    s = _key_to_target(df, "k", "t")
    assert s["a"] == 2.0 and s["b"] == 10.0


def test_spearman_abs_perfect_monotonic():
    left = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0})
    right = pd.Series({"a": 10.0, "b": 20.0, "c": 30.0})
    assert abs(_spearman_abs(left, right) - 1.0) < 1e-9
