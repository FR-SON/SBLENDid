from src.Optimizer.semantic_cost import crossover, optimal_play_mean, assign_cost_integers


def test_crossover_picks_last_gids_below_hnsw():
    curve = [(2, 0.5), (8, 1.0), (32, 3.0), (128, 9.0)]
    assert crossover(curve, hnsw_runtime=2.0) == 8


def test_crossover_zero_when_exact_never_faster():
    assert crossover([(2, 5.0), (8, 9.0)], hnsw_runtime=1.0) == 0


def test_optimal_play_mean_caps_at_hnsw():
    curve = [(2, 0.5), (8, 4.0)]
    assert optimal_play_mean(curve, hnsw_runtime=2.0) == 1.25


def test_assign_cost_integers_normalizes_to_cheapest():
    out = assign_cost_integers({"KW": 0.013, "SC": 0.015, "C": 0.26, "MC": 0.39,
                                "SU": 0.05, "SJ": 0.04})
    assert out["KW"] == 1
    assert out["SJ"] == round(0.04 / 0.013)
    assert out["SU"] == round(0.05 / 0.013)
    assert out["C"] == round(0.26 / 0.013)
    assert out["MC"] == round(0.39 / 0.013)
    assert out["KW"] <= out["SC"] < out["SJ"] < out["SU"] < out["C"] < out["MC"]
    assert out["C"] >= 15 * out["KW"]


def test_assign_cost_integers_ties_on_rounding_and_handles_empty():
    out = assign_cost_integers({"A": 0.01, "B": 0.012, "C": 0.10})
    assert out["A"] == 1 and out["B"] == 1 and out["C"] == 10
    assert assign_cost_integers({}) == {}
