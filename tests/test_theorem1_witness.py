from src.Benchmark.correctness.theorem1.witness import floor_stats, witness_flags


def test_witness_stable_and_differs():
    f = witness_flags([[1, 2], [1, 2]], [[3], [3]])
    assert f["both_stable"] and f["witness"]
    assert f["witness_added"] == 2 and f["witness_dropped"] == 1
    assert not f["witness_bno_empty"]


def test_witness_bno_empty():
    f = witness_flags([[1]], [[]])
    assert f["witness"] and f["witness_bno_empty"]
    assert f["witness_added"] == 1 and f["witness_dropped"] == 0


def test_unstable_is_not_a_witness():
    f = witness_flags([[1], [2]], [[3], [3]])
    assert not f["witness"] and not f["stable_opt"] and f["stable_bno"]
    assert f["witness_added"] is None


def test_stable_equal_not_witness():
    f = witness_flags([[1, 2]], [[2, 1]])
    assert f["both_stable"] and not f["witness"]


def test_floor_hand_computed():
    rows = [{"opt_runs": [[1, 2], [1, 2]], "bno_runs": [[1, 2], [2, 1]]}]
    fs = floor_stats(rows)
    s, l = fs["set"], fs["list"]
    assert (s["p_null_opt"], s["p_null_bno"], s["p_eff"]) == (0.0, 0.0, 0.0)
    assert s["excess_vs_opt"] == s["excess_vs_max"] == 0.0
    assert s["unstable_bno"] == 0
    assert (l["p_null_opt"], l["p_null_bno"], l["p_eff"]) == (0.0, 1.0, 0.5)
    assert l["excess_vs_opt"] == 0.5
    assert l["excess_vs_max"] == -0.5
    assert l["unstable_bno"] == 1
    assert s["raw_rate_not_evidence"] is True
