from pathlib import Path
import pytest
from src.Benchmark.correctness.sweep import run_sweep
from src.Benchmark.db import open_dataset_db
from src.Benchmark.datasource import load_union_queries

_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (_ROOT / "datasets/santos/blend.duckdb").exists(),
                    reason="santos dataset not present")
def test_sweep_smoke_santos():
    db = open_dataset_db("santos")
    try:
        queries = [(t, None) for t in load_union_queries("santos")[:2]]
        rows, summary = run_sweep("santos", plan="su_sc", combine="intersection",
                                  k=10, widths=[10, 100], k_coarses=[100, 500],
                                  efs=[16, 64], ga_width=200, repeats=1,
                                  queries=queries, db=db)
    finally:
        db.close()
    assert rows, "no rows produced"
    arms = {r["arm"] for r in rows}
    assert {"OPT", "REV", "GA", "GB", "R-NOPUSH"} <= arms
    if any(r["n_exact"] > 0 for r in rows):
        assert "RECOVERY" in arms
    for r in rows:
        assert {"query", "arm", "point_kind", "axis", "ids", "retention_vs_exact"} <= set(r)
    ans = summary["answers"]
    assert {"a_order", "b_harm", "c_recovery", "d_time"} <= set(ans)
    assert summary["coverage"]["queries_total"] == 2
    by_w = ans["c_recovery"]["by_result_width_exact_source"]
    assert isinstance(by_w, dict)
    if by_w:
        vals = [by_w[w] for w in sorted(by_w, key=int)]
        assert vals == sorted(vals), "width recovery retention not monotone non-decreasing"
