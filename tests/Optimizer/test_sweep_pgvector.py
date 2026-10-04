"""PG-gated: semantic sweep on the pgvector backend."""
import shutil
from pathlib import Path

import pandas as pd
import pytest

from src.Semantic.config import SemanticConfig, SemanticOp, OperatorConfig
from src.Semantic.retrieve import reset_semantic_cache
from src.Optimizer import semantic_sweep as sw

_ROOT = Path(__file__).resolve().parent.parent.parent
_FIX_CSVS = _ROOT / "tests" / "fixtures" / "semantic" / "csvs"
_REGISTRY = (_ROOT / "datasets" / "pytest_pg" / "semantic"
             / "liftus" / "demo" / "index" / "registry.parquet")


@pytest.mark.skipif(not _REGISTRY.exists(),
                    reason="no built pytest_pg semantic index")
def test_pgvector_hnsw_surface_records_nan_faiss_ms(pg_conn, tmp_path):
    from scripts.load_semantic_index_pg import load_semantic_index_pg
    reset_semantic_cache()
    load_semantic_index_pg("pytest_pg", "liftus", "demo")
    cfg = SemanticConfig.load(
        overrides={"dataset": "pytest_pg", "vector_backend": "pgvector"},
        operators={SemanticOp.SU: OperatorConfig(approach="liftus", index_name="demo")},
    )
    csv_dst = cfg.dataset.dir() / "csvs"
    pre = {p.name for p in csv_dst.glob("*.csv")} if csv_dst.exists() else None
    csv_dst.mkdir(parents=True, exist_ok=True)
    added = []
    for p in _FIX_CSVS.glob("*.csv"):
        if not (csv_dst / p.name).exists():
            shutil.copy(p, csv_dst / p.name)
            added.append(p.name)
    try:
        qs = sw.sample_su_queries(cfg, n=1, seed=0)
        out = sw.run_hnsw_surface(cfg, "SU", qs, ef_grid=[16], kc_grid=[10],
                                  out_csv=tmp_path / "h.csv")
        df = pd.read_csv(out)
        assert df["faiss_ms"].isna().all()
        assert df["total_ms"].notna().all() and (df["total_ms"] >= 0).all()
    finally:
        for name in added:
            (csv_dst / name).unlink(missing_ok=True)
        if pre is None and not any(csv_dst.iterdir()):
            csv_dst.rmdir()
