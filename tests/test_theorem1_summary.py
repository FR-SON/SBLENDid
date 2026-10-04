from src.Benchmark.correctness.theorem1 import summary as SM


def _row(query, shape, opt, bno, **kw):
    base = dict(query=query, shape=shape, k=10, repeats=2,
                opt_runs=[opt, opt], bno_runs=[bno, bno],
                stable_opt=True, stable_bno=True, both_stable=True,
                witness=set(opt) != set(bno),
                witness_bno_empty=bool(set(opt) != set(bno) and not bno),
                witness_added=len(set(opt) - set(bno)) if set(opt) != set(bno) else None,
                witness_dropped=len(set(bno) - set(opt)) if set(opt) != set(bno) else None,
                n_full_a=None, n_full_b=None, n_I=None,
                shape_source="src", operators_upstream_faithful=True,
                optimizer_reachable=True, arm3_only=False,
                bno_construction_used=["slice", "slice"], wall_s=0.1,
                prov={"leg_costs": {"a": 4, "b": 4}})
    base.update(kw)
    return base


def test_summary_blocks_and_labels():
    rows = [_row("q1", "sc_sc_flat", [1, 2], [1, 2]),
            _row("q2", "sc_sc_flat", [1, 2], [3]),
            _row("q3", "sc_kw", [5], [], ab=[5], ba=[6],
                 a1_set_differs=True, a1_list_differs=True,
                 optimizer_reachable=False)]
    s = SM.build_summary(rows, dataset="d", k=10, repeats=2,
                         shape_names=["sc_sc_flat", "sc_kw"],
                         skipped={"missing_csv": 0, "too_few_usable_columns": 0,
                                  "error": 0},
                         queries_total=3, arm_set={"a3", "a1"})
    w = s["primary_evidence"]["witnesses_by_shape"]
    assert w["sc_sc_flat"]["witnesses"] == 1
    assert w["sc_sc_flat"]["both_stable"] == 2
    assert w["sc_kw"]["witness_bno_empty"] == 1
    fl = s["supporting_evidence_floor_choice_dependent"]["floor_by_shape"]
    assert fl["sc_sc_flat"]["set"]["raw_rate_not_evidence"] is True
    assert s["arm1"]["aggregate_set_differs"] == 0
    assert s["arm1"]["by_shape"]["sc_kw"]["optimizer_reachable"] is False
    assert "never emits" in s["arm1"]["by_shape"]["sc_kw"]["caveat"]
    assert s["cohort"]["cohort_complete"] is True
    assert "gt_overlay" in s["omitted"]
    SM.print_summary(s)


def test_flatten_rows_scalars_only():
    flat = SM.flatten_rows([_row("q1", "sc_sc_flat", [1, 2], [3])])
    a3 = [r for r in flat if r["arm"] == "a3"]
    assert len(a3) == 2
    assert {"query", "shape", "arm", "repeat", "n_opt", "n_bno",
            "set_differs", "list_differs", "added", "dropped",
            "bno_empty"} <= set(a3[0])
    assert all(not isinstance(v, (list, dict, set)) for r in flat for v in r.values())


def test_arm3_split_by_bno_empty():
    rows = [_row("q1", "sc_sc_flat", [1], []), _row("q2", "sc_sc_flat", [1], [2])]
    s = SM.build_summary(rows, dataset="d", k=10, repeats=2,
                         shape_names=["sc_sc_flat"],
                         skipped={"error": 0}, queries_total=2, arm_set={"a3"})
    a3 = s["arm3"]["by_shape"]["sc_sc_flat"]
    assert a3["bno_empty"]["n"] == 1 and a3["bno_nonempty"]["n"] == 1
    assert a3["note_n_I"].startswith("n_I unavailable")
