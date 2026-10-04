import pandas as pd
from src.Optimizer import sampling as smp


def _make_lake(tmp_path):
    d = tmp_path / "csvs"
    d.mkdir()
    pd.DataFrame({
        "city": ["a", "b", "c", "a"],
        "pop": ["10", "20", "30", "40"],
    }).to_csv(d / "t0.csv", index=False)
    pd.DataFrame({
        "only": ["x", "y"],
    }).to_csv(d / "t1.csv", index=False)
    return d


def test_sc_candidates_cover_all_columns(tmp_path):
    d = _make_lake(tmp_path)
    cands = smp.candidates(d, "SC")
    pairs = {(c.csv, tuple(c.cols)) for c in cands}
    assert ("t0.csv", ("city",)) in pairs
    assert ("t0.csv", ("pop",)) in pairs
    assert ("t1.csv", ("only",)) in pairs
    assert all(c.cardinality >= 1 for c in cands)


def test_c_candidates_need_categorical_and_numeric(tmp_path):
    d = _make_lake(tmp_path)
    cands = smp.candidates(d, "C")
    assert all(c.csv == "t0.csv" for c in cands)
    assert all(len(c.cols) == 2 for c in cands)


def test_c_candidates_skip_high_cardinality_categoricals(tmp_path):
    d = tmp_path / "csvs"
    d.mkdir()
    pd.DataFrame({
        "key": [f"id{i}" for i in range(20)],
        "val": [str(i) for i in range(20)],
    }).to_csv(d / "t.csv", index=False)
    assert smp.candidates(d, "C", max_cat_card=1000) != []
    assert smp.candidates(d, "C", max_cat_card=5) == []


def test_mc_candidates_need_two_plus_columns(tmp_path):
    d = _make_lake(tmp_path)
    cands = smp.candidates(d, "MC")
    assert all(c.csv == "t0.csv" for c in cands)
    assert all(len(c.cols) >= 2 for c in cands)


def test_stratified_sample_is_deterministic(tmp_path):
    d = _make_lake(tmp_path)
    cands = smp.candidates(d, "SC")
    a = smp.stratified_sample(cands, n=2, seed=1)
    b = smp.stratified_sample(cands, n=2, seed=1)
    assert [(c.csv, tuple(c.cols)) for c in a] == [(c.csv, tuple(c.cols)) for c in b]


def test_save_load_roundtrip(tmp_path):
    d = _make_lake(tmp_path)
    specs = smp.candidates(d, "SC")
    p = tmp_path / "specs.jsonl"
    smp.save_specs(specs, p)
    assert smp.load_specs(p) == specs


def test_load_columns_returns_value_lists(tmp_path):
    d = _make_lake(tmp_path)
    spec = smp.QuerySpec("SC", "t0.csv", ["city"], 3)
    cols = smp.load_columns(d, spec)
    assert cols == [["a", "b", "c", "a"]]
