import configparser
import importlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.create_blend_csv_index import ingest_csv_dir
from src.Semantic.config import OperatorConfig, SemanticConfig, SemanticOp
from src.Semantic.index_build import build_semantic_index
from src.Semantic.retrieve import reset_semantic_cache
from tests.conftest import drop_blend_modules


FIXTURE = Path(__file__).parent / "fixtures" / "semantic"


@pytest.fixture()
def blend_env(tmp_path, monkeypatch):
    root = tmp_path / "data"
    dataset_dir = root / "test_ds"
    dataset_dir.mkdir(parents=True)

    db_path = dataset_dir / "blend.duckdb"
    sidecar_path, n_tables = ingest_csv_dir(FIXTURE / "csvs", db_path)
    assert n_tables == 3
    assert sidecar_path == dataset_dir / "blend_index_basenames.parquet"

    approach_dir = dataset_dir / "semantic" / "liftus" / "demo"
    (approach_dir / "ckpt").mkdir(parents=True)
    shutil.copytree(FIXTURE / "ckpt", approach_dir / "ckpt", dirs_exist_ok=True)
    shutil.copytree(FIXTURE / "aspects", approach_dir / "aspects")
    semantic_cfg = SemanticConfig.load(
        path=tmp_path / "missing.ini",
        overrides={
            "dataset": "test_ds",
            "dataset_root": str(root),
        },
        operators={SemanticOp.SU: OperatorConfig("liftus", "demo")},
    )
    reset_semantic_cache()
    build_semantic_index(FIXTURE / "csvs", "liftus", "demo", semantic_cfg)

    cp = configparser.ConfigParser()
    cp["Dataset"] = {
        "name": "test_ds",
        "root": str(root),
    }
    cp["Database"] = {
        "dbms": "duckdb",
        "db_filename": "blend.duckdb",
        "index_table": "blend_index",
    }
    cp["Semantic"] = {
        "sidecar_filename": "blend_index_basenames.parquet",
        "faiss_quant": "flat",
        "faiss_k_coarse": "20",
        "device": "cpu",
    }
    cp["Semantic.SU"] = {
        "approach": "liftus",
        "index_name": "demo",
    }

    ini = tmp_path / "config.ini"
    with open(ini, "w") as f:
        cp.write(f)
    monkeypatch.setenv("BLEND_CONFIG", str(ini))
    drop_blend_modules()
    try:
        yield
    finally:
        drop_blend_modules()


def test_semantic_union_seeker_returns_blend_int_ids(blend_env):
    Plan = importlib.import_module("src.Plan").Plan
    Seekers = importlib.import_module("src.Operators.Seekers")

    df = pd.read_csv(
        FIXTURE / "csvs" / "SG_CSV0000000000000007.csv",
        dtype=str, keep_default_na=False,
    )
    df.attrs["table_id"] = "SG_CSV0000000000000007.csv"

    plan = Plan()
    plan.add("semu", Seekers.SU(df, k=2))
    ids = plan.run()

    assert isinstance(ids, list)
    assert 0 < len(ids) <= 2
    assert all(isinstance(t, int) for t in ids)
    assert set(ids).issubset({0, 1, 2})


def test_semantic_union_seeker_logical_cost(blend_env):
    Seekers = importlib.import_module("src.Operators.Seekers")
    df = pd.read_csv(
        FIXTURE / "csvs" / "SG_CSV0000000000000007.csv",
        dtype=str, keep_default_na=False,
    )
    df.attrs["table_id"] = "SG_CSV0000000000000007.csv"
    assert Seekers.SU(df, k=2).cost() == 2


def test_index_handle_exposes_table_col_to_gid(blend_env):
    importlib.import_module("src.Plan")
    from src.Semantic.config import SemanticConfig, SemanticOp
    from src.Semantic.retrieve import IndexHandle, reset_semantic_cache

    cfg = SemanticConfig.load()
    reset_semantic_cache()
    handle = IndexHandle.open(cfg, "liftus", cfg.operator(SemanticOp.SU).index_name)
    assert isinstance(handle.table_col_to_gid, dict)
    assert handle.table_col_to_gid
    keys = list(handle.table_col_to_gid.keys())
    table_id, col_name = keys[0]
    gid = handle.table_col_to_gid[(table_id, col_name)]
    assert isinstance(gid, int) and gid >= 0


def test_semantic_union_seeker_tiebreak_ml_cost(blend_env):
    Seekers = importlib.import_module("src.Operators.Seekers")
    qtid = "SG_CSV0000000000000007.csv"
    df = pd.read_csv(FIXTURE / "csvs" / qtid, dtype=str, keep_default_na=False)
    df.attrs["table_id"] = qtid
    multi = Seekers.SU(df, k=2)
    single = Seekers.SU(df[[df.columns[0]]], k=2, query_table_id=qtid)

    assert multi.ml_cost(None) == 1.0
    assert single.ml_cost(None) == 1.0


def test_semantic_union_seeker_unindexed_table_encodes(blend_env, capsys):
    Plan = importlib.import_module("src.Plan").Plan
    Seekers = importlib.import_module("src.Operators.Seekers")
    from src.Semantic.config import SemanticConfig
    from src.Semantic.retrieve import IndexHandle

    calls: list[list[str]] = []

    class _Fake:
        na_cell = ""
        dim = 128

        def encode_batch(self, refs, *, cells_per_ref=None):
            calls.append([r.col_name for r in refs])
            return {r.global_id: np.random.default_rng(r.global_id)
                    .standard_normal(128).astype(np.float32) for r in refs}

    handle = IndexHandle.open(SemanticConfig.load(), "liftus", "demo")
    handle.__dict__["encoder"] = _Fake()

    df = pd.DataFrame({"city": ["Berlin", "Paris"], "pop": ["3", "2"]})
    df.attrs["table_id"] = "external.csv"
    plan = Plan()
    plan.add("su", Seekers.SU(df, k=2, config_overrides={"query_encode": "auto"}))
    ids = plan.run()
    assert 0 < len(ids) <= 2 and set(ids) <= {0, 1, 2}
    assert calls == [["city", "pop"]]
    assert "[SEMANTIC][QUERY-ENCODE]" in capsys.readouterr().err


def test_semantic_union_seeker_liftus_unindexed_raises_not_implemented(blend_env):
    Plan = importlib.import_module("src.Plan").Plan
    Seekers = importlib.import_module("src.Operators.Seekers")
    df = pd.DataFrame({"city": ["Berlin"]})
    df.attrs["table_id"] = "external.csv"
    plan = Plan()
    plan.add("su", Seekers.SU(df, k=2, config_overrides={"query_encode": "auto"}))
    with pytest.raises(NotImplementedError, match="liftus"):
        plan.run()
