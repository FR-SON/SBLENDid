from src.Benchmark.judit import metrics as M


def test_per_query_metrics_basic():
    r = M.per_query_metrics(["a", "b", "c"], {"a", "c"}, ks=(1, 2, 3))
    assert r["precision_at_1"] == 1.0
    assert r["recall_at_2"] == 0.5
    assert r["precision_at_3"] == 2 / 3


def test_assemble_union_matches_inline():
    rows = [M.per_query_metrics(["a", "b"], {"a"}, ks=(1, 2)),
            M.per_query_metrics(["c"], {"c", "d"}, ks=(1, 2))]
    out = M.assemble_union_metrics(
        rows, gt_counts=[1, 2], ks=(1, 2),
        k_coarse={1: 1000, 2: 1000}, k_vote={1: 150, 2: 150},
        ef_used={1: None, 2: None}, depths_derived=False,
        n_saturated=0, n_specs_total=2, n_queries_input=2, n_gt_rows_input=3,
        n_gt_rows_kept=3, n_dropped_q=0, n_dropped_c=0, wall_clock_seconds=0.0)
    assert out["task"] == "union"
    assert out["precision_at_k"]["1"] == (1.0 + 1.0) / 2
    assert out["precision_at_k_corrected"] is not None
    assert set(out) >= {"recall_at_k_ceiling", "recall_at_k_util_pct", "gt_density"}


def test_liftus_map_hand_computed():
    """ranked=[a,x,b,y,c], relevant={a,b,c}, k=5 -> MAP@5 = 49/75."""
    got = M._liftus_map_at_k(["a", "x", "b", "y", "c"], {"a", "b", "c"}, 5)
    assert got == 49 / 75
    assert round(got, 10) == 0.6533333333


def test_liftus_map_equals_ap_when_topk_all_relevant():
    ranked = ["a", "b", "c"]
    relevant = {"a", "b", "c"}
    assert M._liftus_map_at_k(ranked, relevant, 3) == 1.0
    assert M._ap_at_k(ranked, relevant, 3) == 1.0
    assert M._liftus_map_at_k(ranked, relevant, 3) == M._ap_at_k(ranked, relevant, 3)


def test_liftus_map_exceeds_ap_when_precision_drops():
    ranked = ["a", "x", "b", "y", "c"]
    relevant = {"a", "b", "c", "d", "e", "f"}
    liftus = M._liftus_map_at_k(ranked, relevant, 5)
    ap = M._ap_at_k(ranked, relevant, 5)
    assert liftus > ap
    assert liftus == 49 / 75
    assert ap == 34 / 75


def test_liftus_map_below_ap_when_relevant_set_smaller_than_k():
    ranked = ["a", "x", "b", "y", "c"]
    relevant = {"a", "b", "c"}
    liftus = M._liftus_map_at_k(ranked, relevant, 5)
    ap = M._ap_at_k(ranked, relevant, 5)
    assert liftus < ap
    assert liftus == 49 / 75
    assert ap == 34 / 45


def test_liftus_map_empty_relevant_is_zero():
    assert M._liftus_map_at_k(["a", "b"], set(), 5) == 0.0
    assert M._ap_at_k(["a", "b"], set(), 5) == 0.0


def test_liftus_map_ranked_shorter_than_k_keeps_k_as_divisor():
    got = M._liftus_map_at_k(["a"], {"a", "b", "c", "d", "e"}, 5)
    assert got == 137 / 300
    assert got != 1.0


def test_liftus_map_k_one():
    assert M._liftus_map_at_k(["a", "b"], {"a"}, 1) == 1.0
    assert M._liftus_map_at_k(["b", "a"], {"a"}, 1) == 0.0
    assert M._liftus_map_at_k([], {"a"}, 1) == 0.0


def test_ap_at_k_regression_unchanged():
    assert M._ap_at_k(["a", "x", "b", "y", "c"], {"a", "b", "c"}, 5) == 34 / 45
    assert M._ap_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == 1.0
    assert M._ap_at_k(["x", "y"], {"a"}, 2) == 0.0
    assert M._ap_at_k(["x", "a"], {"a"}, 2) == 0.5
    assert M._ap_at_k(["a", "b"], set(), 2) == 0.0


def test_k_range_carries_the_liftus_paper_depths():
    """K_RANGE covers the LIFTus Fig. 4 depths, ascending and duplicate-free, with max 150."""
    assert 60 in M.K_RANGE
    assert {2, 4, 6, 8, 10}.issubset(M.K_RANGE)
    assert {12, 24, 36, 48, 60}.issubset(M.K_RANGE)
    assert set(range(1, 11)).issubset(M.K_RANGE)
    assert list(M.K_RANGE) == sorted(M.K_RANGE)
    assert len(set(M.K_RANGE)) == len(M.K_RANGE)
    assert max(M.K_RANGE) == 150


def test_k_range_never_drops_a_previously_reported_depth():
    assert {1, 3, 5, 10, 15, 20, 25, 50, 60, 75, 100, 130, 150}.issubset(M.K_RANGE)


def test_per_query_metrics_emits_liftus_map_alongside_map():
    r = M.per_query_metrics(["a", "x", "b", "y", "c"], {"a", "b", "c"}, ks=(5,))
    assert r["map_at_5"] == 34 / 45
    assert r["liftus_map_at_5"] == 49 / 75


def test_assemble_union_emits_liftus_map_block():
    rows = [M.per_query_metrics(["a", "b"], {"a"}, ks=(1, 2)),
            M.per_query_metrics(["c"], {"c", "d"}, ks=(1, 2))]
    out = M.assemble_union_metrics(
        rows, gt_counts=[1, 2], ks=(1, 2),
        k_coarse={1: 1000, 2: 1000}, k_vote={1: 150, 2: 150},
        ef_used={1: None, 2: None}, depths_derived=False,
        n_saturated=0, n_specs_total=2, n_queries_input=2, n_gt_rows_input=3,
        n_gt_rows_kept=3, n_dropped_q=0, n_dropped_c=0, wall_clock_seconds=0.0)
    assert out["liftus_map_at_k"] == {"1": 1.0, "2": 0.75}
    assert out["liftus_map_at_k_corrected"] is not None
    assert "liftus_map_at_k_util_pct" not in out


def test_assemble_union_tolerates_rows_without_the_new_key():
    rows = [{f"{m}_at_{k}": 0.0
             for m in ("precision", "recall", "ndcg", "map",
                       "precision_lb", "recall_lb")
             for k in (1, 2)}]
    out = M.assemble_union_metrics(
        rows, gt_counts=[1], ks=(1, 2),
        k_coarse={1: 1000, 2: 1000}, k_vote={1: 150, 2: 150},
        ef_used={1: None, 2: None}, depths_derived=False,
        n_saturated=0, n_specs_total=1, n_queries_input=1, n_gt_rows_input=1,
        n_gt_rows_kept=1, n_dropped_q=0, n_dropped_c=0, wall_clock_seconds=0.0)
    assert out["liftus_map_at_k"] is None
    assert out["liftus_map_at_k_corrected"] is None
