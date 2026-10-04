import pandas as pd

from src.Optimizer.report import (_serialize_report, build_cost_report,
                                  censored_median, format_report)


def _measure_csvs(base):
    (base / "measure").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"seeker_type": ["SC", "SC"], "csv": ["a", "b"],
                  "cols": ["x", "y"], "runtime_s": [0.01, 0.02]}).to_csv(
        base / "measure" / "single_seeker_runtimes_SC.csv", index=False)
    pd.DataFrame({"seeker_type": ["C", "C"], "csv": ["a", "b"],
                  "cols": ["x", "y"], "runtime_s": [1.0, 2.0]}).to_csv(
        base / "measure" / "single_seeker_runtimes_C.csv", index=False)


def _hnsw_csv(base, *, faiss=True):
    (base / "semantic").mkdir(parents=True, exist_ok=True)
    nan = float("nan")
    pd.DataFrame({"seeker": ["SU", "SU"], "efSearch": [64, 64], "k_coarse": [500, 500],
                  "table_id": ["t", "u"], "col_name": ["", ""], "n_cols": [5, 5],
                  "faiss_ms": [0.5, 0.5] if faiss else [nan, nan],
                  "total_ms": [3.0, 3.0],
                  "overhead_ms": [2.5, 2.5] if faiss else [nan, nan]}).to_csv(
        base / "semantic" / "hnsw_surface_SU.csv", index=False)


def _exact_curve(base):
    pd.DataFrame({"seeker": ["SU", "SU"], "table_id": ["t", "u"], "col_name": ["", ""],
                  "n_query_cols": [5, 5], "m_tables": [1, 9], "n_gids": [2, 100],
                  "exact_ms": [0.5, 5.0]}).to_csv(
        base / "semantic" / "exact_curve_SU.csv", index=False)


def _filtered_curve(base):
    pd.DataFrame({"seeker": ["SU", "SU"], "table_id": ["t", "u"], "col_name": ["", ""],
                  "n_query_cols": [5, 5], "m_tables": [1, 9], "n_gids": [7, 2049],
                  "exact_ms": [0.4, 2.2]}).to_csv(
        base / "semantic" / "filtered_curve_SU.csv", index=False)


def _write_fixture(base):
    _measure_csvs(base)
    _hnsw_csv(base, faiss=True)
    _exact_curve(base)


def test_build_cost_report_ranks_and_assigns_costs(tmp_path):
    _write_fixture(tmp_path)
    rep = build_cost_report(tmp_path, backend="faiss")
    assert rep["order"] == ["SU", "SC", "C"]
    assert rep["cost_int"]["SU"] < rep["cost_int"]["SC"] < rep["cost_int"]["C"]
    assert rep["counts"]["SC"] == 2 and rep["counts"]["C"] == 2
    su = rep["semantic"]["SU"]
    assert su["exact_threshold_gids"] == 2
    assert su["faiss_ms"] == 0.5 and su["overhead_ms"] == 2.5
    assert "SU" in rep["sensitivity"]
    assert int(rep["crossover_surface"]["SU"].loc[64, 500]) == 2
    assert rep["backend"] == "faiss"


def test_faiss_default_backend_unchanged(tmp_path):
    _write_fixture(tmp_path)
    assert build_cost_report(tmp_path)["semantic"]["SU"]["exact_threshold_gids"] == 2


def test_faiss_ignores_stale_filtered_curve(tmp_path):
    _write_fixture(tmp_path)
    _filtered_curve(tmp_path)
    su = build_cost_report(tmp_path, backend="faiss")["semantic"]["SU"]
    assert su["exact_threshold_gids"] == 2
    assert "filtered_ms_at_min" not in su


def test_format_report_is_a_string(tmp_path):
    _write_fixture(tmp_path)
    text = format_report(build_cost_report(tmp_path, backend="faiss"))
    assert "cost() ordering" in text
    assert "exact_threshold" in text
    assert "SU=" in text
    assert "HNSW sensitivity" in text
    assert "recommended exact_threshold (crossover |gids|) per efSearch" in text


def test_pgvector_reports_filtered_block_not_exact_threshold(tmp_path):
    _measure_csvs(tmp_path)
    _hnsw_csv(tmp_path, faiss=False)
    _filtered_curve(tmp_path)
    rep = build_cost_report(tmp_path, backend="pgvector")
    su = rep["semantic"]["SU"]
    assert su["filtered_gids_min"] == 7 and su["filtered_gids_max"] == 2049
    assert su["filtered_ms_at_min"] == 0.4 and su["filtered_ms_at_max"] == 2.2
    assert "exact_threshold_gids" not in su
    assert rep["crossover_surface"] == {}
    assert rep["backend"] == "pgvector"
    text = format_report(rep)
    assert "filtered 0.400 ms @|gids|=7" in text
    assert "planner-owned" in text
    assert "faiss n/a" in text and "faiss nan" not in text
    assert "recommended exact_threshold" not in text


def test_pgvector_ignores_stale_exact_curve(tmp_path):
    _measure_csvs(tmp_path)
    _hnsw_csv(tmp_path, faiss=False)
    _filtered_curve(tmp_path)
    _exact_curve(tmp_path)
    su = build_cost_report(tmp_path, backend="pgvector")["semantic"]["SU"]
    assert su["filtered_ms_at_min"] == 0.4
    assert "exact_threshold_gids" not in su


def test_build_cost_report_tolerates_missing_semantic(tmp_path):
    (tmp_path / "measure").mkdir(parents=True)
    pd.DataFrame({"seeker_type": ["SC"], "csv": ["a"], "cols": ["x"],
                  "runtime_s": [0.01]}).to_csv(
        tmp_path / "measure" / "single_seeker_runtimes_SC.csv", index=False)
    rep = build_cost_report(tmp_path, backend="pgvector")
    assert rep["order"] == ["SC"]
    assert rep["semantic"] == {}
    assert rep["crossover_surface"] == {}


def _measure_csv(base, stype, runtimes):
    d = base / "measure"
    d.mkdir(parents=True, exist_ok=True)
    import pandas as _pd
    _pd.DataFrame([{"seeker_type": stype, "csv": f"t{i}.csv", "cols": "a",
                    "runtime_s": rt} for i, rt in enumerate(runtimes)]
                  ).to_csv(d / f"single_seeker_runtimes_{stype}.csv", index=False)


def _samples(base, stype, n):
    base.mkdir(parents=True, exist_ok=True)
    (base / f"samples_{stype}.jsonl").write_text(
        "".join('{"csv": "t%d.csv", "cols": ["a"]}\n' % i for i in range(n)))


def test_report_records_how_many_queries_were_offered(tmp_path):
    _measure_csv(tmp_path, "MC", [1.0, 2.0, 3.0])
    _samples(tmp_path, "MC", 15)
    rep = build_cost_report(tmp_path)
    assert rep["counts"]["MC"] == 3
    assert rep["sampled"]["MC"] == 15
    assert "12 / 15" in format_report(rep)


def test_offered_count_is_absent_when_the_samples_file_is_gone(tmp_path):
    _measure_csv(tmp_path, "SC", [0.5, 0.6])
    rep = build_cost_report(tmp_path)
    assert rep["counts"]["SC"] == 2
    assert "SC" not in rep["sampled"]
    assert "     -" in format_report(rep)


def test_offered_count_survives_serialization(tmp_path):
    _measure_csv(tmp_path, "Keyword", [0.1, 0.2])
    _samples(tmp_path, "Keyword", 4)
    ser = _serialize_report(build_cost_report(tmp_path))
    assert ser["sampled"] == {"KW": 4}


def test_report_says_why_an_operator_is_absent(tmp_path):
    _measure_csv(tmp_path, "SC", [0.5])
    (tmp_path / "semantic").mkdir()
    pd.DataFrame([{"seeker": "SU", "efSearch": 64, "k_coarse": 500, "table_id": "t",
                   "col_name": "", "n_cols": 2, "faiss_ms": 1.0, "total_ms": 8.0,
                   "overhead_ms": 1.0}]).to_csv(
        tmp_path / "semantic" / "hnsw_surface_SU.csv", index=False)
    rep = build_cost_report(tmp_path, ef_default=60, kc_default=60)
    txt = format_report(rep)
    assert "no cell at ef=60/kc=60" in rep["skipped_ops"]["SU"]
    assert "ef=[64]" in rep["skipped_ops"]["SU"]
    assert "no hnsw_surface_SJ.csv" in rep["skipped_ops"]["SJ"]
    assert "SU ABSENT" in txt and "rebases" in txt


def test_report_reads_split_measure_and_semantic_dirs(tmp_path):
    """measure/ and semantic/ dirs resolve independently (one may be profile-keyed, one legacy)."""
    keyed, legacy = tmp_path / "duckdb-faiss-single", tmp_path
    _measure_csv(keyed, "SC", [0.4, 0.5])
    _samples(legacy, "SC", 10)
    rep = build_cost_report(tmp_path, measure_dir=keyed / "measure",
                            semantic_dir=legacy / "semantic",
                            samples_dir=legacy)
    assert rep["counts"]["SC"] == 2
    assert rep["sampled"]["SC"] == 10


def _outcome_csv(base, stype, runtimes, outcomes):
    (base / "measure").mkdir(parents=True, exist_ok=True)
    n = len(outcomes)
    pd.DataFrame({"seeker_type": [stype] * n, "csv": [f"c{i}" for i in range(n)],
                  "cols": ["x"] * n, "runtime_s": runtimes,
                  "outcome": outcomes}).to_csv(
        base / "measure" / f"single_seeker_runtimes_{stype}.csv", index=False)


def test_censored_median_indexes_the_full_sample():
    assert censored_median(list(range(1, 36)), 15) == 25.5
    assert censored_median(list(range(1, 36)), 0) == 18.0


def test_censored_median_is_none_once_half_is_censored():
    assert censored_median([1, 2, 3], 3) is None
    assert censored_median([1, 2], 5) is None
    assert censored_median([], 0) is None


def test_censored_rows_raise_the_median(tmp_path):
    _hnsw_csv(tmp_path, faiss=True)
    _exact_curve(tmp_path)
    _outcome_csv(tmp_path, "SC", [0.01, 0.02, 0.03, "", ""],
                 ["ok", "ok", "ok", "censored", "censored"])
    rep = build_cost_report(tmp_path, backend="faiss")
    assert rep["median_s"]["SC"] == 0.03
    assert rep["counts"]["SC"] == 3
    assert rep["censored"]["SC"] == 2
    assert rep["errored"]["SC"] == 0


def test_errors_leave_the_population(tmp_path):
    _hnsw_csv(tmp_path, faiss=True)
    _exact_curve(tmp_path)
    _outcome_csv(tmp_path, "SC", [0.01, 0.02, 0.03, ""], ["ok", "ok", "ok", "error"])
    rep = build_cost_report(tmp_path, backend="faiss")
    assert rep["median_s"]["SC"] == 0.02
    assert rep["errored"]["SC"] == 1 and rep["censored"]["SC"] == 0


def test_majority_censored_drops_the_type_with_a_reason(tmp_path):
    _hnsw_csv(tmp_path, faiss=True)
    _exact_curve(tmp_path)
    _outcome_csv(tmp_path, "SC", [0.01, "", "", ""],
                 ["ok", "censored", "censored", "censored"])
    rep = build_cost_report(tmp_path, backend="faiss")
    assert "SC" not in rep["median_s"]
    assert "wall cap" in rep["skipped_ops"]["SC"]
    assert "SC" not in format_report(rep).split("cost() ordering")[1].split("\n")[0]


def test_legacy_csv_without_outcome_reads_as_fully_observed(tmp_path):
    _write_fixture(tmp_path)
    rep = build_cost_report(tmp_path, backend="faiss")
    assert rep["median_s"]["SC"] == 0.015
    assert rep["censored"]["SC"] is None and rep["errored"]["SC"] is None
    assert "n/a" in format_report(rep)
