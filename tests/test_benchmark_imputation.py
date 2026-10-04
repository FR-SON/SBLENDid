import pandas as pd
from src.Benchmark.recipes.imputation import (
    _fd_confidence, _reconstruct, _select_key_value)


def test_fd_confidence():
    func = pd.DataFrame({"k": ["a", "a", "b"], "v": ["1", "1", "2"]})
    assert _fd_confidence(func, "k", "v") == 1.0
    broken = pd.DataFrame({"k": ["a", "a", "b"], "v": ["1", "2", "3"]})
    assert abs(_fd_confidence(broken, "k", "v") - 2 / 3) < 1e-9


def test_select_picks_discriminative_key_value(tmp_path):
    df = pd.DataFrame({
        "id":    [f"E{i:03d}" for i in range(20)],
        "city":  ["paris", "lyon", "nice", "metz"] * 5,
        "flag":  ["1", "2"] * 10,
        "const": ["x"] * 20,
    })
    df.to_csv(tmp_path / "t.csv", index=False)
    kv = _select_key_value(tmp_path / "t.csv")
    assert kv is not None
    assert kv[0] == "id"
    assert kv[1] == "city"


def test_select_rejects_table_without_key(tmp_path):
    df = pd.DataFrame({"a": ["x", "y"] * 10, "b": ["1", "2", "3", "4"] * 5})
    df.to_csv(tmp_path / "t.csv", index=False)
    assert _select_key_value(tmp_path / "t.csv") is None


def test_reconstruct_masks_second_column(tmp_path):
    df = pd.DataFrame({"k": [str(i) for i in range(8)], "v": [str(i * 10) for i in range(8)]})
    df.to_csv(tmp_path / "t.csv", index=False)
    examples, queries = _reconstruct(tmp_path / "t.csv", "k", "v")
    assert list(examples.columns) == ["k", "v"]
    assert len(examples) == 5
    assert len(queries) == 3
