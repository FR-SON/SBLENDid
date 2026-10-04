from pathlib import Path
import pytest
from src.Benchmark.adapters.union import run_union

_ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.skipif(not (_ROOT / "datasets/santos/blend.duckdb").exists(),
                    reason="santos dataset not present")
def test_run_union_santos_smoke():
    rows, summary = run_union("santos", k=10)
    assert summary["evaluated"] > 0
    assert summary["effective_n"] == summary["evaluated"]
    assert 0.0 <= summary["mean_p@k"] <= 1.0
    assert all("runtime_ms" in r for r in rows)
