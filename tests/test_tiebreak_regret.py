"""Regret of the tiebreak model: a misordered pair costs only the gap between its legs."""

from __future__ import annotations

from src.Benchmark.tiebreak.bench import classify_pair, summarize


def _rows(pairs):
    """pairs: (a_gt, b_gt, a_ncols, b_ncols, a_pred, b_pred) -> summarize rows."""
    out = []
    for a_gt, b_gt, a_nc, b_nc, a_pr, b_pr in pairs:
        a = {"gt_ms": a_gt, "n_cols": a_nc, "pred_ms": a_pr}
        b = {"gt_ms": b_gt, "n_cols": b_nc, "pred_ms": b_pr}
        out.append({**classify_pair(a, b), "a_gt_ms": a_gt, "b_gt_ms": b_gt})
    return out


def _sum(rows):
    return summarize(rows, "ds", queries=[], configs=[], skipped={}, k=10,
                     k_coarse_grid=[60], repeats=1, seed=0)


def test_regret_charges_only_the_gap_of_a_misordered_pair():
    s = _sum(_rows([(1.0, 9.0, 1, 2, 9.0, 1.0)]))
    assert s["regret"]["new"]["regret_ms"] == 8.0
    assert s["regret"]["new"]["regret_vs_oracle_pct"] == 800.0


def test_being_wrong_only_on_near_ties_is_almost_free():
    """A less accurate model can still be cheaper when its errors land on near-ties."""
    pairs = []
    for _ in range(10):
        pairs.append((1.00, 1.01, 1, 2, 1.01, 1.00))
    for _ in range(10):
        pairs.append((1.0, 50.0, 1, 2, 1.0, 50.0))
    s = _sum(_rows(pairs))
    r = s["regret"]["new"]
    assert s["new_correct"] == 10
    assert r["regret_ms"] < 0.2
    assert r["mean_gap_when_wrong_ms"] < s["median_gap_ms"]


def test_ties_are_charged_half_the_gap():
    s = _sum(_rows([(1.0, 5.0, 3, 3, 1.0, 5.0)]))
    assert s["placeholder_ties"] == 1
    assert s["regret"]["placeholder"]["regret_ms"] == 2.0
    assert s["regret"]["new"]["regret_ms"] == 0.0


def test_regressions_are_counted_separately_from_corrections():
    rows = _rows([
        (1.0, 9.0, 2, 1, 1.0, 9.0),
        (1.0, 9.0, 1, 2, 9.0, 1.0),
    ])
    s = _sum(rows)
    assert s["decisions_corrected"] == 1
    assert s["decisions_regressed"] == 1


def test_coin_flip_baseline_is_half_of_every_gap():
    """Without a chance reference the two arms are only comparable to each other."""
    s = _sum(_rows([(1.0, 5.0, 1, 2, 1.0, 5.0), (2.0, 4.0, 1, 2, 2.0, 4.0)]))
    assert s["coin_flip_regret_ms"] == 3.0
    assert s["regret"]["new"]["regret_ms"] == 0.0


def test_pair_sampling_is_uniform_and_bounded():
    """Sampling must stay unbiased over unordered pairs without materialising them all."""
    from collections import Counter
    from src.Benchmark.tiebreak.bench import _all_pairs, _sample_pairs

    assert len(list(_all_pairs(50))) == 50 * 49 // 2
    got = _sample_pairs(50, 400, seed=0)
    assert len(got) == 400 and len(set(got)) == 400
    assert all(i < j < 50 for i, j in got)
    c = Counter(x for pair in got for x in pair)
    assert max(c.values()) < 3 * min(c.values())


def test_sampling_is_deterministic_for_a_seed():
    from src.Benchmark.tiebreak.bench import _sample_pairs
    assert _sample_pairs(80, 200, seed=7) == _sample_pairs(80, 200, seed=7)
    assert _sample_pairs(80, 200, seed=7) != _sample_pairs(80, 200, seed=8)


def test_summary_records_which_op_was_ordered():
    """SU and SJ legs are different decisions, so results.json must record the op."""
    s = _sum(_rows([(1.0, 9.0, 1, 2, 1.0, 9.0)]))
    assert s["op"] == "SU"
    s2 = summarize(_rows([(1.0, 9.0, 1, 2, 1.0, 9.0)]), "ds", queries=[], configs=[],
                   skipped={}, k=10, k_coarse_grid=[60], repeats=1, seed=0, op="SJ")
    assert s2["op"] == "SJ"


def test_op_is_in_the_run_dir_name_so_sj_cannot_clobber_su():
    from src.Benchmark.cli import build_parser
    from src.Benchmark.runspec import Field, render_slug as slug
    fields = (Field("op", default="SU"), Field("k", prefix="k", default=10))
    su = slug(fields, {"op": "SU", "k": 25})
    sj = slug(fields, {"op": "SJ", "k": 25})
    assert su != sj and "SJ" in sj
    assert "SU" not in su
    assert build_parser().parse_args(
        ["tiebreak-bench", "--dataset", "L", "--op", "SJ"]).op == "SJ"
