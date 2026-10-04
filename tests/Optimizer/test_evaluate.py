import pandas as pd
from src.Optimizer import evaluate as ev


def test_sample_plan_specs_mc_splits_columns(tmp_path):
    d = tmp_path / "csvs"
    d.mkdir()
    pd.DataFrame({"a": ["1"], "b": ["2"], "c": ["3"], "d": ["4"]}).to_csv(d / "t.csv", index=False)
    pd.DataFrame({"x": ["1"], "y": ["2"]}).to_csv(d / "few.csv", index=False)
    specs = ev.sample_plan_specs(d, seeker_type="MC", n=50)
    assert len(specs) == 1
    ps = specs[0]
    assert ps.seeker_type == "MC" and ps.csv == "t.csv"
    assert ps.col_groups == [["a", "b"], ["c", "d"]]


def test_aggregate_intersection():
    assert sorted(ev.aggregate("Intersection", [{1, 2, 3}, {2, 3, 4}], k=10)) == [2, 3]


def test_aggregate_union_respects_k():
    out = ev.aggregate("Union", [{1, 2}, {3, 4}], k=3)
    assert len(out) == 3
    assert set(out) <= {1, 2, 3, 4}


def test_aggregate_difference():
    assert sorted(ev.aggregate("Difference", [{1, 2, 3}, {2}], k=10)) == [1, 3]


def test_aggregate_counter_orders_by_frequency():
    out = ev.aggregate("Counter", [{1, 2}, {2, 3}, {2, 4}], k=2)
    assert out[0] == 2


def test_summarize_groups_by_seeker_type(tmp_path):
    d = tmp_path / "eval"
    d.mkdir()
    pd.DataFrame([
        {"seeker_type": "SC", "t_naive": 1.0, "t_rule": 0.9, "t_ml": 1.2, "t_infer": 0.3, "t_query_ml": 0.9},
        {"seeker_type": "SC", "t_naive": 1.0, "t_rule": 0.9, "t_ml": 1.0, "t_infer": 0.2, "t_query_ml": 0.8},
        {"seeker_type": "C", "t_naive": 5.0, "t_rule": 4.0, "t_ml": 3.5, "t_infer": 0.1, "t_query_ml": 3.4},
    ]).to_csv(d / "optimizer_eval.csv", index=False)
    g = ev.summarize(d)
    assert g.loc["SC", "t_ml"] == 2.2 and g.loc["SC", "plans"] == 2
    assert bool(g.loc["C", "ml<=naive"]) is True
    assert bool(g.loc["SC", "ml<=naive"]) is False
    assert bool(g.loc["C", "ml_order_vs_rule"]) is True
