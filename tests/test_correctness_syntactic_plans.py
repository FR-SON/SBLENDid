import pandas as pd

from src.Benchmark import coltype
from src.Benchmark.correctness import syntactic as SY


def test_infer_col_numeric():
    assert coltype.infer_col(["1", "1", "1", "2", "2", "2"])["col_type"] == "numeric"


def test_infer_col_id():
    assert coltype.infer_col(["1", "2", "3", "4", "5", "6"])["col_type"] == "id"


def test_infer_col_categorical():
    assert coltype.infer_col(["a", "b"] * 10)["col_type"] == "categorical"


def test_infer_col_text():
    cells = [f"free form sentence number {i}" for i in range(20)]
    assert coltype.infer_col(cells)["col_type"] == "text"


def test_infer_col_empty():
    assert coltype.infer_col(["", "  ", "nan"])["col_type"] == "empty"


def test_classify_columns_maps_names():
    df = pd.DataFrame({"cat": ["a", "b"] * 10,
                       "val": ["1", "1", "1", "2", "2", "2"] * 3 + ["1", "2"]})
    types = coltype.classify_columns(df)
    assert types["cat"] == "categorical"
    assert types["val"] == "numeric"


def test_select_inputs_sc_defaults_to_col0():
    df = pd.DataFrame({"a": ["x", "y"], "b": ["1", "2"]})
    inp = SY.select_inputs("sc", df, None)
    assert inp.kind == "sc" and inp.roles["col"] == "a"


def test_select_inputs_sc_honours_qcol():
    df = pd.DataFrame({"a": ["x"], "b": ["y"]})
    assert SY.select_inputs("sc", df, "b").roles["col"] == "b"


def test_select_inputs_c_picks_categorical_key_numeric_target():
    df = pd.DataFrame({"cat": ["a", "b"] * 10,
                       "val": [str(i % 3) for i in range(20)]})
    inp = SY.select_inputs("c", df, None)
    assert inp.kind == "c"
    assert inp.roles == {"key": "cat", "target": "val"}


def test_select_inputs_c_skips_when_no_numeric():
    df = pd.DataFrame({"cat": ["a", "b"] * 10,
                       "txt": [f"long free text value {i}" for i in range(20)]})
    assert SY.select_inputs("c", df, None) is None


def test_select_inputs_mc_join_first_plus_nonnumeric_excludes_numeric():
    df = pd.DataFrame({"j": ["k1", "k2"] * 10,
                       "other": ["p", "q"] * 10,
                       "num": [str(i % 3) for i in range(20)]})
    inp = SY.select_inputs("mc", df, "j")
    assert inp.kind == "mc"
    assert inp.roles["cols"][0] == "j"
    assert "other" in inp.roles["cols"]
    assert "num" not in inp.roles["cols"]


def test_select_inputs_mc_skips_single_usable_col():
    df = pd.DataFrame({"j": ["k1", "k2"] * 10,
                       "num": [str(i % 3) for i in range(20)]})
    assert SY.select_inputs("mc", df, "j") is None


def test_derive_with_order_uses_full_order():
    full_order = [5, 4, 3, 2, 1]
    assert SY.derive(set(full_order), full_order, [1, 2, 3], 2) == [3, 2]


def test_derive_without_order_uses_semantic_order():
    assert SY.derive({1, 2, 3}, None, [3, 1, 2, 9], 2) == [3, 1]
