import pytest

from src.Semantic.aggregators.max_pool import MaxPoolAggregator


def test_requires_no_ctx():
    assert MaxPoolAggregator.requires_ctx is False
    assert MaxPoolAggregator.name == "max_pool"


def test_sign_convention_stored_values_are_negated_similarities():
    hits = [(["better", "worse"], [-0.9, -0.2])]
    ranked = MaxPoolAggregator().aggregate(hits, k=10)
    assert [t for t, _ in ranked] == ["better", "worse"]
    assert ranked[0][1] == pytest.approx(0.9)
    assert ranked[1][1] == pytest.approx(0.2)


def test_best_column_wins_even_when_ranked_low():
    hits = [
        (["x1", "x2", "x3", "deep"], [-0.99, -0.98, -0.97, -0.95]),
        (["popular", "y1"], [-0.5, -0.4]),
    ]
    ranked = MaxPoolAggregator().aggregate(hits, k=10)
    order = [t for t, _ in ranked]
    assert order.index("deep") < order.index("popular")
    assert dict(ranked)["deep"] == pytest.approx(0.95)


def test_k_truncates_output():
    hits = [(["a", "b", "c"], [-0.9, -0.8, -0.7])]
    ranked = MaxPoolAggregator().aggregate(hits, k=2)
    assert [t for t, _ in ranked] == ["a", "b"]


def test_ties_break_by_table_id():
    hits = [(["zeta", "alpha"], [-0.5, -0.5]), (["mid"], [-0.5])]
    ranked = MaxPoolAggregator().aggregate(hits, k=10)
    assert [t for t, _ in ranked] == ["alpha", "mid", "zeta"]


def test_empty_input():
    assert MaxPoolAggregator().aggregate([], k=10) == []
    assert MaxPoolAggregator().aggregate([([], [])], k=10) == []
