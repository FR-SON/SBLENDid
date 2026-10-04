import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DIM = 8


def _reset_operator_db():
    # Close Operator.DB first: evicting the module alone does not release the DuckDB file lock.
    mod = sys.modules.get("src.Operators.OperatorBase")
    if mod is not None:
        mod.Operator.DB.close()
    from tests.conftest import drop_blend_modules
    drop_blend_modules()


def fake_encode(strings):
    out = []
    for s in strings:
        seed = int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "little")
        v = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
        out.append(v / np.linalg.norm(v))
    return np.stack(out)


@pytest.fixture(params=["single", "two_table"])
def sho_dataset(request, monkeypatch, tmp_path):
    from scripts.create_blend_csv_index import ingest_csv_dir
    from scripts.create_sho_index import materialize_duckdb, run_hash, run_sample

    _reset_operator_db()
    repo = Path(__file__).resolve().parent.parent
    ds = repo / "datasets" / "_sho_e2e_test"
    if ds.exists():
        shutil.rmtree(ds)
    (ds / "csvs").mkdir(parents=True)
    (ds / "csvs" / "alpha.csv").write_text("name\nred apple\ngreen pear\nred apple\n")
    (ds / "csvs" / "beta.csv").write_text("name\nred apple\nyellow plum\n")
    (ds / "csvs" / "gamma.csv").write_text("name\nzebra\nlion\n")
    ingest_csv_dir(ds / "csvs", ds / "blend.duckdb",
                   sidecar_path=ds / "blend_index_basenames.parquet",
                   workers=1, layout=request.param)
    sho_dir = ds / "semantic" / "simhash" / "default"
    run_sample(ds / "csvs", sho_dir, ds / "blend_index_basenames.parquet",
               row_cap=1000, seed=0, skip_numeric=True)
    run_hash(sho_dir, bits=8, seed=0, encode_fn=fake_encode)
    materialize_duckdb(ds / "blend.duckdb", sho_dir / "codes.parquet")

    cfg = tmp_path / "config.ini"
    cfg.write_text(
        f"[Dataset]\nname = _sho_e2e_test\nroot = {repo / 'datasets'}\n\n"
        "[Database]\ndbms = duckdb\ndb_filename = blend.duckdb\nindex_table = blend_index\n\n"
        "[Semantic.SHO]\napproach = simhash\nindex_name = default\n"
    )
    monkeypatch.setenv("BLEND_CONFIG", str(cfg))
    from tests.conftest import drop_blend_modules
    drop_blend_modules()
    from src.Semantic.sho_artifacts import ShoHandle
    ShoHandle._CACHE.clear()
    yield ds
    shutil.rmtree(ds, ignore_errors=True)
    _reset_operator_db()


def test_sho_seeker_retrieves_overlapping_tables(sho_dataset):
    from src.Benchmark.db import open_dataset_db
    from src.Operators import Seekers
    from src.Plan import Plan

    db = open_dataset_db("_sho_e2e_test")
    try:
        from src.Benchmark.db import bind_plan
        plan = Plan()
        plan.add("sho", Seekers.SHO(["red apple", "green pear"], k=10))
        bind_plan(plan, db)
        ids = plan.run()
        i2b = dict(pd.read_parquet(sho_dataset / "blend_index_basenames.parquet")
                   [["table_int_id", "basename"]].itertuples(index=False))
        names = [i2b[int(t)] for t in ids]
        assert names[0] == "alpha.csv"
        assert "beta.csv" in names
        assert "gamma.csv" not in names
        sho2 = Seekers.SHO(["red apple"], k=10)
        sho2.DB = db
        assert sho2.run(" AND TableId IN (999999) ") == []
    finally:
        db.close()


def test_sho_receives_intersection_pushdown(sho_dataset):
    from src.Benchmark.correctness.arms import run_arm
    from src.Benchmark.db import open_dataset_db
    from src.Operators import Seekers

    db = open_dataset_db("_sho_e2e_test")
    try:
        routed = []
        orig = db.clean_query

        def spy(q):
            r = orig(q)
            routed.append(r)
            return r

        db.clean_query = spy
        legs = [
            Seekers.SC(["red apple"], k=10),
            Seekers.SHO(["red apple", "green pear"], k=10, order_cost=99),
        ]
        result = run_arm(legs, "intersection", "cost", db, 10, guard="none")
        db.clean_query = orig
        sho_sqls = [q for q in routed if "blend_index_code" in q]
        assert sho_sqls, routed
        assert any("TableId IN" in q for q in sho_sqls)
        assert isinstance(result, list) and result
    finally:
        db.close()
