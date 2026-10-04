from src.Benchmark import metrics as m

def test_ranking_metrics():
    retrieved = ["a", "x", "b"]
    relevant = {"a", "b", "c"}
    assert m.precision_at_k(retrieved, relevant) == 2 / 3
    assert m.recall_at_k(retrieved, relevant) == 2 / 3
    assert abs(m.average_precision(retrieved, relevant) - (1.0 + 2/3) / 3) < 1e-9
    assert m.reciprocal_rank(retrieved, relevant) == 1.0

def test_empty_and_f1_and_hit_rate():
    assert m.precision_at_k([], {"a"}) == 0.0
    assert m.recall_at_k(["x"], set()) == 0.0
    assert m.f1(0.5, 0.5) == 0.5
    assert m.f1(0.0, 0.0) == 0.0
    assert m.hit_rate(["q", "a"], {"a"}) == 1.0
    assert m.hit_rate(["q"], {"a"}) == 0.0
