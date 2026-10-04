import csv
import json

import pandas as pd
import pytest

from src.Optimizer import export_tables as xt


def _costs(base, lake, profile, *, order, med, counts=None, sampled=None,
           censored=None, backend="faiss", ef=60, kc=60):
    d = base / lake / "optimizer" / profile
    d.mkdir(parents=True, exist_ok=True)
    doc = {"costs": {}, "report": {
        "median_s": med, "order": order,
        "cost_int": {t: max(1, round(med[t] / min(med.values()))) for t in med},
        "counts": counts or {t: 10 for t in order},
        "sampled": sampled or {}, "censored": censored,
        "basis": {t: "unfiltered" for t in order}},
        "metadata": {"backend": backend, "ef_default": ef, "kc_default": kc}}
    (d / "costs.json").write_text(json.dumps(doc))


def _measure(base, lake, profile, seeker, runtimes, outcomes=None):
    d = base / lake / profile / "measure"
    d.mkdir(parents=True, exist_ok=True)
    cols = {"seeker_type": [seeker] * len(runtimes),
            "csv": [f"t{i}.csv" for i in range(len(runtimes))],
            "cols": ["x"] * len(runtimes), "runtime_s": runtimes}
    if outcomes is not None:
        cols["outcome"] = outcomes
    pd.DataFrame(cols).to_csv(d / f"single_seeker_runtimes_{seeker}.csv", index=False)


def _read(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def test_complete_flags_a_truncated_median():
    rep = {"sampled": {"MC": 50}, "counts": {"MC": 9}, "censored": None, "errored": None}
    assert xt._complete(rep, "MC") is False


def test_complete_accepts_a_fully_observed_legacy_cell():
    rep = {"sampled": {"SC": 500}, "counts": {"SC": 500}, "censored": None, "errored": None}
    assert xt._complete(rep, "SC") is True


def test_complete_counts_censored_as_accounted():
    rep = {"sampled": {"MC": 200}, "counts": {"MC": 134}, "censored": {"MC": 66},
           "errored": {"MC": 0}}
    assert xt._complete(rep, "MC") is True


def test_complete_true_for_seekers_with_no_sample_file():
    assert xt._complete({"sampled": {}, "counts": {"SU": 600}}, "SU") is True


def test_cost_ordering_carries_rank_and_provenance(tmp_path):
    _costs(tmp_path, "lk", xt.DUCKDB_PROFILE, order=["SJ", "SC"],
           med={"SJ": 0.001, "SC": 0.1}, sampled={"SC": 500},
           counts={"SJ": 3000, "SC": 500}, censored={"SC": 0})
    rows = xt.cost_ordering_rows(tmp_path, ["lk"], [xt.DUCKDB_PROFILE])
    assert [r["seeker"] for r in rows] == ["SJ", "SC"]
    assert [r["rank"] for r in rows] == [1, 2]
    assert rows[1]["median_ms"] == 100.0
    assert rows[0]["ef_search"] == 60 and rows[0]["k_coarse"] == 60


def test_cross_backend_still_reports_completeness_per_side(tmp_path):
    _costs(tmp_path, "lk", xt.DUCKDB_PROFILE, order=["MC"], med={"MC": 1.0},
           sampled={"MC": 200}, counts={"MC": 134}, censored={"MC": 66})
    _costs(tmp_path, "lk", xt.PGVECTOR_PROFILE, order=["MC"], med={"MC": 0.3},
           sampled={"MC": 50}, counts={"MC": 9}, censored=None, backend="pgvector")
    row = xt.cross_backend_rows(tmp_path, ["lk"])[0]
    assert row["ratio_b_over_a"] == 0.3
    assert row["complete_a"] is True and row["complete_b"] is False
    assert row["comparison"] == "unknown"


def test_cross_backend_skips_a_seeker_missing_on_one_side(tmp_path):
    _costs(tmp_path, "lk", xt.DUCKDB_PROFILE, order=["SJ", "MC"],
           med={"SJ": 0.001, "MC": 1.0})
    _costs(tmp_path, "lk", xt.PGVECTOR_PROFILE, order=["SJ"], med={"SJ": 0.005},
           backend="pgvector")
    assert [r["seeker"] for r in xt.cross_backend_rows(tmp_path, ["lk"])] == ["SJ"]


def test_censoring_reports_both_medians(tmp_path):
    _measure(tmp_path, "lk", xt.DUCKDB_PROFILE, "MC", [0.01, 0.02, 0.03, None, None],
             ["ok", "ok", "ok", "censored", "censored"])
    row = xt.censoring_rows(tmp_path, ["lk"], [xt.DUCKDB_PROFILE])[0]
    assert row["n_finite"] == 3 and row["n_censored"] == 2
    assert row["median_survivor_s"] == 0.02
    assert row["median_corrected_s"] == 0.03
    assert row["correction_ratio"] == 1.5


def test_censoring_leaves_corrected_blank_past_the_cliff(tmp_path):
    _measure(tmp_path, "lk", xt.DUCKDB_PROFILE, "MC", [0.01, None, None, None],
             ["ok", "censored", "censored", "censored"])
    row = xt.censoring_rows(tmp_path, ["lk"], [xt.DUCKDB_PROFILE])[0]
    assert row["median_corrected_s"] is None and row["correction_ratio"] is None


def test_export_writes_every_table(tmp_path):
    _costs(tmp_path / "ds", "lk", xt.DUCKDB_PROFILE, order=["SJ"], med={"SJ": 0.001})
    _measure(tmp_path / "rn", "lk", xt.DUCKDB_PROFILE, "SC", [0.1, 0.2])
    out = tmp_path / "out"
    written = xt.export(out, datasets_root=tmp_path / "ds", runs_root=tmp_path / "rn",
                        lakes=["lk"], profiles=[xt.DUCKDB_PROFILE])
    assert set(written) == {"cost_ordering.csv", "cross_backend_ratio.csv",
                            "censoring.csv", "depth_sweep.csv", "dispersion.csv",
                            "semantic_dispersion.csv"}
    assert _read(out / "cost_ordering.csv")[0]["seeker"] == "SJ"
    assert _read(out / "censoring.csv")[0]["schema"] == "legacy"


def _surface(base, lake, profile, op, cells):
    d = base / lake / profile / "semantic"
    d.mkdir(parents=True, exist_ok=True)
    rows = [{"seeker": op, "efSearch": ef, "k_coarse": kc, "total_ms": ms}
            for (ef, kc), ms in cells.items()]
    pd.DataFrame(rows).to_csv(d / f"hnsw_surface_{op}.csv", index=False)


def test_depth_sweep_varies_semantic_and_holds_token_constant(tmp_path):
    ds, rn = tmp_path / "ds", tmp_path / "rn"
    _costs(ds, "lk", xt.DUCKDB_PROFILE, order=["SJ", "SC"],
           med={"SJ": 0.001, "SC": 0.5})
    _surface(rn, "lk", xt.DUCKDB_PROFILE, "SJ", {(30, 30): 1.0, (60, 60): 4.0})
    rows = xt.depth_sweep_rows(ds, rn, ["lk"], [xt.DUCKDB_PROFILE])
    sj = {r["k_coarse"]: r["median_ms"] for r in rows if r["seeker"] == "SJ"}
    sc = {r["k_coarse"]: r["median_ms"] for r in rows if r["seeker"] == "SC"}
    assert sj == {30: 1.0, 60: 4.0}
    assert sc == {30: 500.0, 60: 500.0}
    assert all(r["depth_varying"] == (r["seeker"] == "SJ") for r in rows)


def test_depth_sweep_cost_int_is_recomputed_per_cell(tmp_path):
    ds, rn = tmp_path / "ds", tmp_path / "rn"
    _costs(ds, "lk", xt.DUCKDB_PROFILE, order=["SJ", "SC"],
           med={"SJ": 0.001, "SC": 0.5})
    _surface(rn, "lk", xt.DUCKDB_PROFILE, "SJ", {(30, 30): 1.0, (60, 60): 4.0})
    by = {(r["k_coarse"], r["seeker"]): r["cost_int"]
          for r in xt.depth_sweep_rows(ds, rn, ["lk"], [xt.DUCKDB_PROFILE])}
    assert by[(30, "SJ")] == 1 and by[(30, "SC")] == 500
    assert by[(60, "SJ")] == 1 and by[(60, "SC")] == 125


def test_depth_sweep_marks_unreachable_faiss_cells(tmp_path):
    ds, rn = tmp_path / "ds", tmp_path / "rn"
    _costs(ds, "lk", xt.DUCKDB_PROFILE, order=["SJ"], med={"SJ": 0.001})
    _surface(rn, "lk", xt.DUCKDB_PROFILE, "SJ", {(60, 60): 1.0, (64, 500): 9.0})
    faiss = {(r["ef_search"], r["k_coarse"]): r["reachable"]
             for r in xt.depth_sweep_rows(ds, rn, ["lk"], [xt.DUCKDB_PROFILE])}
    assert faiss[(60, 60)] is True and faiss[(64, 500)] is False

    _costs(ds, "lk", xt.PGVECTOR_PROFILE, order=["SJ"], med={"SJ": 0.001},
           backend="pgvector")
    _surface(rn, "lk", xt.PGVECTOR_PROFILE, "SJ", {(64, 30): 2.0, (64, 310): 3.0})
    pg = {(r["ef_search"], r["k_coarse"]): r["reachable"]
          for r in xt.depth_sweep_rows(ds, rn, ["lk"], [xt.PGVECTOR_PROFILE])}
    assert all(pg.values())


def test_depth_sweep_emits_the_curve_without_costs_json(tmp_path):
    ds, rn = tmp_path / "ds", tmp_path / "rn"
    _surface(rn, "lk", xt.DUCKDB_PROFILE, "SU", {(30, 30): 1.5, (60, 60): 2.5})
    rows = xt.depth_sweep_rows(ds, rn, ["lk"], [xt.DUCKDB_PROFILE])
    assert [r["median_ms"] for r in rows] == [1.5, 2.5]
    assert all(r["cost_int"] is None and r["rank"] is None for r in rows)
    assert all(r["backend"] == "faiss" for r in rows)


def test_censored_quantile_agrees_with_the_median_helper():
    from src.Optimizer.report import censored_median, censored_quantile
    for xs, cens in ((list(range(1, 36)), 0), (list(range(1, 36)), 15),
                     ([1, 2, 3, 4], 0), ([1, 2, 3], 0)):
        assert censored_quantile(xs, cens, 0.5) == censored_median(xs, cens)


def test_censored_quantile_blanks_a_truncated_tail():
    from src.Optimizer.report import censored_quantile
    xs, cens = list(range(1, 36)), 15
    assert censored_quantile(xs, cens, 0.5) is not None
    assert censored_quantile(xs, cens, 0.95) is None
    assert censored_quantile(xs, cens, 0.99) is None


def test_dispersion_thins_out_exactly_where_the_cap_cut(tmp_path):
    _measure(tmp_path, "lk", xt.DUCKDB_PROFILE, "MC",
             [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, None, None],
             ["ok"] * 8 + ["censored", "censored"])
    row = xt.dispersion_rows(tmp_path, ["lk"], [xt.DUCKDB_PROFILE])[0]
    assert row["censored_frac"] == 0.2
    assert row["p50_ms"] is not None and row["p75_ms"] is not None
    assert row["p90_ms"] is None and row["p95_ms"] is None and row["p99_ms"] is None
    assert row["max_identifiable_q"] == 0.75
    assert row["max_finite_ms"] == 800.0


def test_dispersion_reports_every_quantile_when_nothing_was_capped(tmp_path):
    _measure(tmp_path, "lk", xt.DUCKDB_PROFILE, "SC", [i / 100 for i in range(1, 101)],
             ["ok"] * 100)
    row = xt.dispersion_rows(tmp_path, ["lk"], [xt.DUCKDB_PROFILE])[0]
    assert row["max_identifiable_q"] == 0.99
    assert all(row[f"p{q}_ms"] is not None for q in (50, 75, 90, 95, 99))
    assert row["spread_p95_over_p50"] == pytest.approx(row["p95_ms"] / row["p50_ms"])


def _samples(base, lake, seeker, n):
    d = base / lake
    d.mkdir(parents=True, exist_ok=True)
    (d / f"samples_{seeker}.jsonl").write_text(
        "".join('{"seeker_type": "%s", "csv": "t%d.csv"}\n' % (seeker, i)
                for i in range(n)))


def _measure_named(base, lake, profile, seeker, names, runtimes):
    d = base / lake / profile / "measure"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"seeker_type": [seeker] * len(names), "csv": names,
                  "cols": ["x"] * len(names), "runtime_s": runtimes}).to_csv(
        d / f"single_seeker_runtimes_{seeker}.csv", index=False)


def _two_profiles(ds, lake, seeker, med_a, med_b):
    for prof, m in ((xt.DUCKDB_PROFILE, med_a), (xt.PGVECTOR_PROFILE, med_b)):
        _costs(ds, lake, prof, order=[seeker], med={seeker: m},
               backend="faiss" if "faiss" in prof else "pgvector")


def test_ratio_is_disjoint_when_the_draws_differ(tmp_path):
    ds, rn = tmp_path / "ds", tmp_path / "rn"
    _two_profiles(ds, "lk", "SC", 0.1, 0.5)
    _measure_named(rn, "lk", xt.DUCKDB_PROFILE, "SC", ["a", "b", "c"], [0.1, 0.2, 0.3])
    _measure_named(rn, "lk", xt.PGVECTOR_PROFILE, "SC", ["c", "d"], [0.5, 0.6])
    row = xt.cross_backend_rows(ds, ["lk"], runs_root=rn)[0]
    assert row["comparison"] == "disjoint"
    assert row["specs_nested"] is False and row["n_shared_specs"] == 1
    assert row["comparable"] is False


def test_ratio_is_nested_when_one_side_is_a_prefix(tmp_path):
    ds, rn = tmp_path / "ds", tmp_path / "rn"
    _two_profiles(ds, "lk", "SC", 0.1, 0.5)
    _measure_named(rn, "lk", xt.DUCKDB_PROFILE, "SC", ["a", "b", "c"], [0.1, 0.2, 0.3])
    _measure_named(rn, "lk", xt.PGVECTOR_PROFILE, "SC", ["a", "b"], [0.5, 0.6])
    row = xt.cross_backend_rows(ds, ["lk"], runs_root=rn)[0]
    assert row["comparison"] == "nested"
    assert row["same_specs"] is False and row["specs_nested"] is True
    assert row["n_specs_a"] == 3 and row["n_specs_b"] == 2
    assert row["comparable"] is True


def test_ratio_stays_citable_when_the_specs_match(tmp_path):
    ds, rn = tmp_path / "ds", tmp_path / "rn"
    for prof in (xt.DUCKDB_PROFILE, xt.PGVECTOR_PROFILE):
        _costs(ds, "lk", prof, order=["SC"], med={"SC": 0.1},
               sampled={"SC": 2}, counts={"SC": 2},
               backend="faiss" if "faiss" in prof else "pgvector")
        _measure(rn, "lk", prof, "SC", [0.1, 0.2])
    row = xt.cross_backend_rows(ds, ["lk"], runs_root=rn)[0]
    assert row["same_specs"] is True and row["comparison"] == "paired"
    assert row["comparable"] is True


def test_semantic_rows_have_no_spec_comparison(tmp_path):
    ds, rn = tmp_path / "ds", tmp_path / "rn"
    for prof in (xt.DUCKDB_PROFILE, xt.PGVECTOR_PROFILE):
        _costs(ds, "lk", prof, order=["SU"], med={"SU": 0.01},
               backend="faiss" if "faiss" in prof else "pgvector")
    row = xt.cross_backend_rows(ds, ["lk"], runs_root=rn)[0]
    assert row["same_specs"] is None and row["comparison"] == "unknown"
    assert row["comparable"] is True


def test_censoring_keeps_the_median_on_a_short_run_but_flags_the_size(tmp_path):
    _samples(tmp_path, "lk", "MC", 200)
    _measure(tmp_path, "lk", xt.DUCKDB_PROFILE, "MC", [0.1, 0.2, 0.3], ["ok"] * 3)
    row = xt.censoring_rows(tmp_path, ["lk"], [xt.DUCKDB_PROFILE])[0]
    assert row["n_offered"] == 200 and row["complete"] is False and row["n_total"] == 3
    assert row["median_survivor_s"] == 0.2 and row["median_corrected_s"] == 0.2


def test_dispersion_suppresses_a_quantile_the_sample_cannot_support(tmp_path):
    _samples(tmp_path, "lk", "MC", 200)
    _measure(tmp_path, "lk", xt.DUCKDB_PROFILE, "MC",
             [i / 100 for i in range(1, 10)], ["ok"] * 9)
    row = xt.dispersion_rows(tmp_path, ["lk"], [xt.DUCKDB_PROFILE])[0]
    assert row["n_total"] == 9 and row["complete"] is False
    assert row["p50_ms"] is not None and row["p75_ms"] is not None
    assert row["p90_ms"] is None and row["p95_ms"] is None and row["p99_ms"] is None
    assert row["max_identifiable_q"] == 0.75


def test_dispersion_support_rule_scales_with_sample_size(tmp_path):
    _measure(tmp_path, "lk", xt.DUCKDB_PROFILE, "SC",
             [i / 1000 for i in range(1, 101)], ["ok"] * 100)
    row = xt.dispersion_rows(tmp_path, ["lk"], [xt.DUCKDB_PROFILE])[0]
    assert row["max_identifiable_q"] == 0.99


def _query_surface(base, lake, profile, op, cells, *, repeats=1, col=""):
    d = base / lake / profile / "semantic"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for (ef, kc), per_query in cells.items():
        for i, ms in enumerate(per_query):
            for rep in range(repeats):
                rows.append({"seeker": op, "repeat": rep, "efSearch": ef,
                             "k_coarse": kc, "table_id": f"t{i}",
                             "col_name": f"{col}{i}" if col else "",
                             "n_cols": 1, "total_ms": ms * (1 + 0.01 * rep),
                             "plan_label": "faiss"})
    pd.DataFrame(rows).to_csv(d / f"hnsw_surface_{op}.csv", index=False)


def test_semantic_dispersion_collapses_replicates_before_quantiles(tmp_path):
    _query_surface(tmp_path, "lk", xt.DUCKDB_PROFILE, "SU",
                   {(60, 60): [1.0, 9.0]}, repeats=3)
    row = xt.semantic_dispersion_rows(tmp_path, ["lk"], [xt.DUCKDB_PROFILE])[0]
    assert row["n_queries"] == 2 and row["n_repeats"] == 3
    assert row["p50_ms"] == pytest.approx(5.05)
    assert row["median_within_query_cv_pct"] == pytest.approx(0.99, abs=0.01)


def test_semantic_dispersion_has_no_censoring_to_thin_it(tmp_path):
    _query_surface(tmp_path, "lk", xt.DUCKDB_PROFILE, "SU",
                   {(60, 60): [float(i) for i in range(1, 201)]})
    row = xt.semantic_dispersion_rows(tmp_path, ["lk"], [xt.DUCKDB_PROFILE])[0]
    assert all(row[f"p{q}_ms"] is not None for q in (50, 75, 90, 95, 99))
    assert row["max_ms"] == 200.0
    assert row["spread_p95_over_p50"] == pytest.approx(190.05 / 100.5)


def test_semantic_dispersion_suppresses_an_unsupported_quantile(tmp_path):
    _query_surface(tmp_path, "lk", xt.DUCKDB_PROFILE, "SU",
                   {(60, 60): [float(i) for i in range(1, 41)]})
    row = xt.semantic_dispersion_rows(tmp_path, ["lk"], [xt.DUCKDB_PROFILE])[0]
    assert row["p95_ms"] is not None and row["p99_ms"] is None


def test_semantic_dispersion_keys_queries_by_column_not_just_table(tmp_path):
    d = tmp_path / "lk" / xt.DUCKDB_PROFILE / "semantic"
    d.mkdir(parents=True)
    pd.DataFrame([{"seeker": "SJ", "repeat": 0, "efSearch": 60, "k_coarse": 60,
                   "table_id": "t0", "col_name": c, "n_cols": 1, "total_ms": ms,
                   "plan_label": "faiss"}
                  for c, ms in (("a", 1.0), ("b", 5.0), ("c", 9.0))]
                 ).to_csv(d / "hnsw_surface_SJ.csv", index=False)
    row = xt.semantic_dispersion_rows(tmp_path, ["lk"], [xt.DUCKDB_PROFILE])[0]
    assert row["n_queries"] == 3 and row["p50_ms"] == 5.0


def test_semantic_dispersion_marks_unreachable_faiss_cells(tmp_path):
    _query_surface(tmp_path, "lk", xt.DUCKDB_PROFILE, "SU",
                   {(60, 60): [1.0, 2.0], (64, 500): [3.0, 4.0]})
    rows = xt.semantic_dispersion_rows(tmp_path, ["lk"], [xt.DUCKDB_PROFILE])
    reach = {(r["ef_search"], r["k_coarse"]): r["reachable"] for r in rows}
    assert reach == {(60, 60): True, (64, 500): False}


def test_semantic_dispersion_skips_a_surface_without_per_query_rows(tmp_path):
    _surface(tmp_path, "lk", xt.DUCKDB_PROFILE, "SJ", {(30, 30): 1.0})
    assert xt.semantic_dispersion_rows(tmp_path, ["lk"], [xt.DUCKDB_PROFILE]) == []
