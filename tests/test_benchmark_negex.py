import pandas as pd
from src.Benchmark.recipes.negative_example import _row_set, _is_superset


def test_row_set_and_superset():
    q = pd.DataFrame({"a": ["x", "y"], "b": ["1", "2"]})
    c_full = pd.DataFrame({"a": ["x", "y", "z"], "b": ["1", "2", "3"]})
    c_part = pd.DataFrame({"a": ["x"], "b": ["1"]})
    qs = _row_set(q, "a", "b")
    assert _is_superset(_row_set(c_full, "a", "b"), qs) is True
    assert _is_superset(_row_set(c_part, "a", "b"), qs) is False
