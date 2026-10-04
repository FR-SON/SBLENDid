"""Stored snoopy vectors match a fresh encode_one, so the query path can use lookup."""
from __future__ import annotations
import importlib

import numpy as np
import pandas as pd


def test_stored_vector_matches_fresh_encode(sj_env):
    """handle.vectors[gid] must equal encoder.encode_one(ref) within fp32 BLAS tolerance."""
    SemanticConfig = importlib.import_module("src.Semantic.config").SemanticConfig
    IndexHandle = importlib.import_module("src.Semantic.retrieve").IndexHandle
    load_registry = importlib.import_module("src.Semantic.registry").load_registry

    cfg = SemanticConfig.load()
    handle = IndexHandle.open(cfg, "snoopy", "default")
    assert handle.vectors is not None, "fixture must produce embeddings.fp32.npy"

    refs = list(load_registry(
        cfg.index_dir("snoopy", "default") / "registry.parquet"
    ))
    assert refs, "fixture registry is empty"

    mismatches: list[tuple[int, float]] = []
    for ref in refs:
        stored = np.asarray(handle.vectors[ref.global_id], dtype=np.float32)
        fresh = handle.encoder.encode_one(ref)
        if not np.allclose(stored, fresh, rtol=1e-5, atol=1e-6):
            mismatches.append((ref.global_id, float(np.abs(stored - fresh).max())))

    assert not mismatches, (
        f"{len(mismatches)} of {len(refs)} columns differ between stored "
        f"and recompute; first: gid={mismatches[0][0]} max-abs-diff="
        f"{mismatches[0][1]:.3e}. Determinism assumption violated; do not "
        "proceed with lookup substitution until the source of drift is found."
    )


def test_handle_exposes_table_col_to_gid(sj_env):
    """One table_col_to_gid entry per registry row with a non-null col_name."""
    SemanticConfig = importlib.import_module("src.Semantic.config").SemanticConfig
    IndexHandle = importlib.import_module("src.Semantic.retrieve").IndexHandle
    load_registry = importlib.import_module("src.Semantic.registry").load_registry

    cfg = SemanticConfig.load()
    handle = IndexHandle.open(cfg, "snoopy", "default")
    refs = list(load_registry(cfg.index_dir("snoopy", "default") / "registry.parquet"))

    expected = {
        (r.table_id, r.col_name): r.global_id
        for r in refs if r.col_name is not None
    }
    assert handle.table_col_to_gid == expected


def test_sj_query_path_skips_encode_one_for_indexed_column(sj_env, monkeypatch):
    """retrieve.search must use handle.vectors[gid] and never call encode_one for an indexed column."""
    Plan = importlib.import_module("src.Plan").Plan
    Seekers = importlib.import_module("src.Operators.Seekers")
    from src.Semantic.encoders.snoopy import ScorpionAdapter

    def _boom(self, ref):
        raise AssertionError(
            "encode_one called for indexed column — lookup branch not taken"
        )

    monkeypatch.setattr(ScorpionAdapter, "encode_one", _boom)

    FIXTURE = importlib.import_module(
        "tests.test_semantic_join_seeker"
    ).FIXTURE
    df = pd.read_csv(
        FIXTURE / "csvs" / "t0000003.csv",
        dtype=str, keep_default_na=False,
    )
    df.attrs["table_id"] = "t0000003.csv"
    plan = Plan()
    plan.add("sj", Seekers.SJ(df, k=2))
    ids = plan.run()
    assert 0 < len(ids) <= 2
    assert all(isinstance(t, int) for t in ids)


def test_semantic_join_seeker_logical_cost(sj_env):
    Seekers = importlib.import_module("src.Operators.Seekers")
    FIXTURE = importlib.import_module("tests.test_semantic_join_seeker").FIXTURE
    df = pd.read_csv(
        FIXTURE / "csvs" / "t0000003.csv",
        dtype=str, keep_default_na=False,
    )
    df.attrs["table_id"] = "t0000003.csv"
    assert Seekers.SJ(df, k=2).cost() == 1


def test_semantic_join_seeker_tiebreak_ml_cost(sj_env):
    Seekers = importlib.import_module("src.Operators.Seekers")
    FIXTURE = importlib.import_module("tests.test_semantic_join_seeker").FIXTURE
    df = pd.read_csv(
        FIXTURE / "csvs" / "t0000003.csv",
        dtype=str, keep_default_na=False,
    )
    df.attrs["table_id"] = "t0000003.csv"
    sj = Seekers.SJ(df, k=2)

    assert sj.ml_cost(None) == 1.0
