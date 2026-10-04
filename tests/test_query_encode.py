import numpy as np
import pandas as pd
import pytest

from src.Semantic import query_encode as qe


class _Enc:
    na_cell = "nan"
    dim = 3

    def __init__(self):
        self.calls = []

    def encode_one(self, ref, *, cells=None):
        self.calls.append(("one", ref.global_id, cells))
        return np.full(3, float(len(cells)), dtype=np.float32)


class _BatchEnc(_Enc):
    def encode_batch(self, refs, *, cells_per_ref=None):
        self.calls.append(("batch", [r.global_id for r in refs], cells_per_ref))
        return {r.global_id: np.full(3, float(r.global_id), dtype=np.float32) for r in refs}


def test_cells_for_substitutes_na_cell():
    enc = _Enc()
    s = pd.Series(["a", None, 1.5])
    assert qe.cells_for(enc, s) == ["a", "nan", "1.5"]
    enc.na_cell = ""
    assert qe.cells_for(enc, s) == ["a", "", "1.5"]


def test_empty_columns_detects_nan_and_whitespace():
    df = pd.DataFrame({"nan_only": [None, None], "blank": ["", "  "],
                       "mixed": ["", "x"], "num": [1, 2]})
    assert qe.empty_columns(df, list(df.columns)) == ["nan_only", "blank"]


def test_query_refs_shapes():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    (r,) = qe.query_refs("q.csv", df, ["b"])
    assert (r.global_id, r.table_id, r.col_idx, r.col_name, r.n_rows, r.source_path) == \
        (0, "q.csv", 1, "b", 3, "q.csv")


def test_encode_query_columns_per_column_without_batch():
    enc = _Enc()
    df = pd.DataFrame({"a": ["1", "2"], "b": ["x", None]})
    out = qe.encode_query_columns(enc, "q.csv", df, ["a", "b"])
    assert set(out) == {"a", "b"}
    assert [c[0] for c in enc.calls] == ["one", "one"]
    assert enc.calls[1][2] == ["x", "nan"]
    assert out["a"].dtype == np.float32


def test_encode_query_columns_prefers_batch():
    enc = _BatchEnc()
    df = pd.DataFrame({"a": ["1", "2"], "b": ["x", "y"]})
    out = qe.encode_query_columns(enc, "q.csv", df, ["a", "b"])
    assert len(enc.calls) == 1 and enc.calls[0][0] == "batch"
    assert enc.calls[0][2] == {0: ["1", "2"], 1: ["x", "y"]}
    assert np.array_equal(out["b"], np.full(3, 1.0, dtype=np.float32))


def test_encode_query_columns_empty_cols():
    assert qe.encode_query_columns(_Enc(), "q.csv", pd.DataFrame({"a": [1]}), []) == {}


def test_encode_query_columns_integer_labels_keyed_by_str():
    df = pd.DataFrame([["x", "y"], ["z", "w"]])            # columns 0, 1
    out = qe.encode_query_columns(_Enc(), "q.csv", df, [0, 1])
    assert set(out) == {"0", "1"}


def test_encode_query_columns_rejects_duplicate_labels():
    df = pd.DataFrame([["x", "y"]], columns=["id", "id"])
    with pytest.raises(ValueError, match="duplicate column"):
        qe.encode_query_columns(_Enc(), "q.csv", df, ["id"])
    with pytest.raises(ValueError, match="duplicate column"):
        qe.empty_columns(df, ["id"])
