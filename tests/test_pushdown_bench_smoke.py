import pytest

from src.Benchmark.datasource import load_union_queries
from src.Benchmark.db import open_dataset_db
from src.Benchmark.pushdown.bench import run_pushdown_bench


def test_pushdown_bench_smoke():
    try:
        db = open_dataset_db("santos")
    except Exception as e:
        pytest.skip(f"no live Postgres/santos: {e}")
    try:
        try:
            queries = load_union_queries("santos")[:2]
        except FileNotFoundError as e:
            pytest.skip(f"no santos dataset on disk: {e}")
        rows, summary = run_pushdown_bench(
            "santos", k=10, selectivities=[0.05], comps=[0, 5, 10],
            modes=["prefilter", "postfilter"], exact_threshold=0, k_coarse=60,
            repeats=1, queries=queries, db=db, seed=0,
        )
    finally:
        db.close()
    assert rows
    assert summary["max_leakage"] == 0
    for r in rows:
        assert r["recall"] is None or 0.0 <= r["recall"] <= 1.0
