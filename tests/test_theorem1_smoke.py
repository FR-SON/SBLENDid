import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (_ROOT / "datasets/santos/blend.duckdb").exists(),
                    reason="santos dataset not present")
def test_theorem1_smoke_santos(tmp_path):
    from src.Benchmark.cli import main
    rc = main(["correctness-theorem1", "--dataset", "santos",
               "--shapes", "sc_sc_flat", "--k", "10", "--repeats", "2",
               "--limit", "2", "--out", str(tmp_path / "run")])
    assert rc == 0
    rows = [json.loads(l) for l in
            (tmp_path / "run" / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    for r in rows:
        assert {"query", "shape", "opt_runs", "bno_runs", "witness",
                "both_stable", "bno_construction_used", "shape_source",
                "wall_s"} <= set(r)
        assert len(r["opt_runs"]) == 2 and len(r["bno_runs"]) == 2
    assert (tmp_path / "run" / "witnesses.json").is_file()
    assert (tmp_path / "run" / "audit.json").is_file()
    assert (tmp_path / "run" / "manifest.json").is_file()
    m = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert m["config"]["fidelity"]["pushdown_guard"] == "none"
    assert "config_snapshot" in m["config"]
