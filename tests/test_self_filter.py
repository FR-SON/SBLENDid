import pandas as pd

from src.Benchmark.self_filter import (
    drop_self_gt,
    rollup_without_self,
    without_self,
    without_self_scored,
)


def test_drop_self_gt_removes_self_rows():
    gt = pd.DataFrame(
        {
            "query_table": ["a.csv", "a.csv", "b.csv"],
            "candidate_table": ["a.csv", "c.csv", "d.csv"],
        }
    )
    out = drop_self_gt(gt)
    assert list(zip(out["query_table"], out["candidate_table"])) == [
        ("a.csv", "c.csv"),
        ("b.csv", "d.csv"),
    ]


def test_drop_self_gt_noop_when_no_self():
    gt = pd.DataFrame(
        {"query_table": ["a", "b"], "candidate_table": ["c", "d"]}
    )
    assert len(drop_self_gt(gt)) == 2


def test_without_self_bare_and_tuple():
    assert without_self(["a", "b", "a"], "a") == ["b"]
    assert without_self([("a", "x"), ("b", "y"), ("a", "z")], "a") == [("b", "y")]


def test_without_self_scored_keeps_alignment():
    items = ["a", "b", "a", "c"]
    scores = [0.9, 0.8, 0.7, 0.6]
    it, sc = without_self_scored(items, scores, "a")
    assert it == ["b", "c"]
    assert sc == [0.8, 0.6]


def test_rollup_without_self_excludes_self_and_frees_slot():
    per_col = [(["q", "t1", "t2"], [1.0, 0.9, 0.8])]
    rolled = rollup_without_self(per_col, "q", k=2)
    tables = [t for t, _ in rolled]
    assert "q" not in tables
    assert tables == ["t1", "t2"]
