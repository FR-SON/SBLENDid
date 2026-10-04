from src.Benchmark.metrics import ndcg_at_k


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == 1.0


def test_ndcg_empty_relevant_is_zero():
    assert ndcg_at_k(["a", "b"], set(), 3) == 0.0


def test_ndcg_no_retrieved_is_zero():
    assert ndcg_at_k([], {"a"}, 3) == 0.0


def test_ndcg_is_rank_sensitive():
    top = ndcg_at_k(["a", "x", "y"], {"a"}, 3)
    bottom = ndcg_at_k(["x", "y", "a"], {"a"}, 3)
    assert top > bottom


def test_ndcg_partial():
    import math
    expected = 1.0 / (1.0 + 1.0 / math.log2(3))
    assert abs(ndcg_at_k(["a", "x"], {"a", "b"}, 2) - expected) < 1e-9
