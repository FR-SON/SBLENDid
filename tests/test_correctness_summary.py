from src.Benchmark.correctness.sweep import _summarize


def _r(arm, *, point_kind="operating", axis=None, axis_val=None, n_exact=5, n_nopush=4,
       retention_vs_exact=1.0, retention_vs_nopush=1.0, added_vs_exact=0,
       additions_vs_nopush=0, ids=(), runtime_ms=0.0, real_losses=0, real_gains=0,
       query="a"):
    return dict(arm=arm, point_kind=point_kind, axis=axis, axis_val=axis_val,
                exact_source=False, k_coarse_used=None, n_exact=n_exact, n_nopush=n_nopush,
                retention_vs_exact=retention_vs_exact, retention_vs_nopush=retention_vs_nopush,
                added_vs_exact=added_vs_exact, additions_vs_nopush=additions_vs_nopush,
                rbo_vs_exact=retention_vs_exact, n_retrieved=len(ids), ids=list(ids),
                runtime_ms=runtime_ms, real_losses=real_losses, real_gains=real_gains,
                query=query)


def _rows():
    return [
        _r("R-NOPUSH", retention_vs_exact=0.85, ids=[1, 2, 3]),
        _r("R-EXACT", retention_vs_exact=1.0, ids=[1, 2, 3, 4, 5]),
        _r("OPT", retention_vs_exact=0.8, retention_vs_nopush=0.9, added_vs_exact=2,
           ids=[1, 2], runtime_ms=10.0, real_losses=2, real_gains=1),
        _r("REV", retention_vs_exact=1.0, ids=[1, 2, 3], runtime_ms=5.0),
        _r("GA", retention_vs_exact=0.95, ids=[1, 2, 3, 4], runtime_ms=15.0),
        _r("GB", retention_vs_exact=1.0, ids=[1, 2, 3, 4, 5], runtime_ms=4.0),
        _r("RECOVERY", point_kind="recovery", axis="width", axis_val=10, retention_vs_exact=0.4),
        _r("RECOVERY", point_kind="recovery", axis="width", axis_val=100, retention_vs_exact=0.99),
        _r("RECOVERY", point_kind="recovery", axis="k_coarse", axis_val=50, retention_vs_exact=0.7),
        _r("RECOVERY", point_kind="recovery", axis="ef", axis_val=16, retention_vs_exact=0.6),
        _r("RECOVERY", point_kind="recovery", axis="width", axis_val=10, n_exact=0,
           retention_vs_exact=1.0, query="z"),
    ]


def _summary():
    return _summarize(_rows(), ["a", "z"], dataset="d", plan="su_sc", k=10,
                      operating={"k_coarse": 500, "ef": 64})


def test_a_order():
    a = _summary()["answers"]["a_order"]
    assert a["retention_opt_vs_exact"] == 0.8
    assert a["retention_rev_vs_exact"] == 1.0
    assert a["opt_vs_rev_jaccard_dist"] == 0.3333
    assert a["queries_changed_frac"] == 1.0
    assert a["n_queries"] == 1


def test_b_harm():
    b = _summary()["answers"]["b_harm"]
    assert b["opt_dropped_frac_vs_exact"] == 0.2
    assert b["lead_gt_losses_distinct"] == 2
    assert b["lead_gt_gains_distinct"] == 1
    assert b["attribution"]["approx_only_dropped_frac"] == 0.15
    assert b["attribution"]["pushdown_added_dropped_frac"] == 0.1


def test_c_recovery():
    c = _summary()["answers"]["c_recovery"]
    assert c["by_result_width_exact_source"] == {"10": 0.4, "100": 0.99}
    assert c["by_k_coarse"] == {"50": 0.7}
    assert c["by_ef_search"] == {"16": 0.6}
    assert c["width_for_retention_0.99_exact_source"] == 100
    assert c["ga_retention_vs_exact"] == 0.95
    assert c["gb_retention_vs_exact"] == 1.0


def test_d_time():
    d = _summary()["answers"]["d_time"]
    assert d["runtime_ms_median"] == {"OPT": 10.0, "REV": 5.0, "GA": 15.0, "GB": 4.0}
    assert d["speedup_opt_over_rev"] == 0.5
    assert d["guard_a_cost_frac"] == 0.5


def test_coverage_excludes_degenerate():
    s = _summary()
    assert s["coverage"]["queries_vs_exact"] == 1
    assert s["operating_point"]["k_coarse"] == 500
