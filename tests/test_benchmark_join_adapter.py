from pathlib import Path
import pytest
from src.Benchmark.adapters.join import run_join

_ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.skipif(not (_ROOT / "datasets/opendata-split-12/blend.duckdb").exists(),
                    reason="opendata-split-12 not present")
def test_run_join_opendata_smoke():
    rows, summary = run_join("opendata-split-12", k=10)
    assert summary["evaluated"] > 0
    assert summary["effective_n"] <= summary["total_queries"]
    assert all("query_column" in r for r in rows)
