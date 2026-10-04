import numpy as np
import pytest

from src.Semantic.aggregators.base import AggregationContext
from src.Semantic.aggregators.munkres import MunkresAggregator


def _ctx(qmat, tables: dict[str, np.ndarray]):
    return AggregationContext(
        qmat=np.asarray(qmat, dtype=np.float32),
        candidate_vectors_fn=lambda t: np.asarray(
            tables.get(t, np.empty((0, qmat.shape[1]))), dtype=np.float32),
        query_table_id="q.csv",
    )


def test_requires_ctx():
    assert MunkresAggregator.requires_ctx is True
    with pytest.raises(ValueError, match="ctx"):
        MunkresAggregator(threshold=0.5).aggregate([(["a"], [0.1])], k=5)


def test_assignment_score_and_ranking():
    q = np.eye(2)
    tables = {"A": np.eye(2), "B": np.array([[1.0, 0.0]])}
    hits = [(["A", "B"], [0.0, 0.1]), (["A"], [0.0])]
    agg = MunkresAggregator(threshold=0.5)
    ranked = agg.aggregate(hits, k=10, ctx=_ctx(q, tables))
    assert ranked[0] == ("A", pytest.approx(2.0))
    assert ranked[1] == ("B", pytest.approx(1.0))


def test_threshold_masks_all_pairs():
    q = np.eye(2)
    tables = {"A": np.array([[0.6, 0.8]])}
    hits = [(["A"], [0.0])]
    agg = MunkresAggregator(threshold=0.9)
    assert agg.aggregate(hits, k=10, ctx=_ctx(q, tables)) == []


def test_one_to_one_assignment():
    q = np.eye(2)
    a = np.array([[0.9, 0.1], [0.8, 0.6]])
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    hits = [(["A"], [0.0])]
    agg = MunkresAggregator(threshold=0.05)
    ranked = agg.aggregate(hits, k=10, ctx=_ctx(q, {"A": a}))
    cos = np.eye(2, dtype=np.float32) @ a.T.astype(np.float32)
    expected = max(cos[0, 0] + cos[1, 1], cos[0, 1] + cos[1, 0])
    assert ranked[0][1] == pytest.approx(float(expected), abs=1e-5)


def test_candidate_first_seen_order_and_k_cut():
    q = np.eye(2)
    tables = {t: np.eye(2) for t in ("A", "B", "C")}
    hits = [(["C", "A"], [0.0, 0.0]), (["B"], [0.0])]
    agg = MunkresAggregator(threshold=0.5)
    ranked = agg.aggregate(hits, k=2, ctx=_ctx(q, tables))
    assert len(ranked) == 2
    assert [t for t, _ in ranked] == ["C", "A"]


def test_empty_candidate_matrix_skipped():
    q = np.eye(2)
    hits = [(["ghost"], [0.0])]
    agg = MunkresAggregator(threshold=0.1)
    assert agg.aggregate(hits, k=5, ctx=_ctx(q, {})) == []
