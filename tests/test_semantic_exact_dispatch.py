"""Exact-search dispatch tests (faiss_or_exact_search + SU exact_threshold)."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Import Seekers first: importing union_base directly is a circular import.
from src.Operators import Seekers  # noqa: F401

from src.Operators.Combiners.Intersection import Intersection
from src.Operators.Seekers.SeekerBase import Seeker

from src.Semantic.config import SemanticConfig
from src.Semantic.index_build import build_semantic_index
from src.Semantic.retrieve import (
    IndexHandle, faiss_or_exact_search, reset_semantic_cache,
)
from src.Semantic.seekers.union_base import SemanticUnionSeekerBase


FIXTURE = Path(__file__).parent / "fixtures" / "semantic"


def _basenames() -> list[str]:
    return sorted(p.name for p in (FIXTURE / "csvs").glob("*.csv"))


def _write_basenames(p: Path) -> None:
    pd.DataFrame(
        [(i, b) for i, b in enumerate(_basenames())],
        columns=["table_int_id", "basename"],
    ).astype({"table_int_id": "int64", "basename": "string"}).to_parquet(p, index=False)


@pytest.fixture()
def cfg(tmp_path):
    root = tmp_path / "data"
    dataset_dir = root / "test_ds"
    approach_dir = dataset_dir / "semantic" / "liftus" / "demo"
    (approach_dir / "ckpt").mkdir(parents=True)
    shutil.copytree(FIXTURE / "ckpt", approach_dir / "ckpt", dirs_exist_ok=True)
    shutil.copytree(FIXTURE / "aspects", approach_dir / "aspects")
    _write_basenames(dataset_dir / "blend_index_basenames.parquet")

    cfg = SemanticConfig.load(
        path=tmp_path / "missing.ini",
        overrides={
            "dataset": "test_ds",
            "dataset_root": str(root),
            "faiss_quant": "flat",
        },
    )
    reset_semantic_cache()
    build_semantic_index(FIXTURE / "csvs", "liftus", "demo", cfg)
    return cfg


def _query_df() -> pd.DataFrame:
    df = pd.read_csv(FIXTURE / "csvs" / "SG_CSV0000000000000007.csv",
                     dtype=str, keep_default_na=False)
    df.attrs["table_id"] = "SG_CSV0000000000000007.csv"
    return df


class _FakeDB:
    dbms = "duckdb"


def test_dispatch_uses_faiss_when_exact_threshold_none(cfg):
    """exact_threshold=None must never take the exact path even with a tight filter."""
    handle = IndexHandle.open(cfg, "liftus", "demo")
    from src.Semantic.registry import ColumnRef
    from src.Semantic.indexers.base import l2_normalize_rows
    ref = ColumnRef(global_id=-1, table_id="SG_CSV0000000000000007.csv",
                    col_idx=0, col_name="Hospital ID", n_rows=1,
                    source_path="SG_CSV0000000000000007.csv")
    vec = handle.encoder.encode_one(ref)
    q = l2_normalize_rows(vec[None, :].astype(np.float32))

    scores_faiss, gids_faiss = handle.faiss_index.search(q, k=5)
    scores_d, gids_d = faiss_or_exact_search(
        handle, q, k=5,
        table_filter={"SG_CSV0000000000000008.csv"},
        exact_threshold=None,
    )
    assert np.array_equal(gids_d, gids_faiss)
    assert np.allclose(scores_d, scores_faiss)


def test_dispatch_uses_exact_when_filter_within_threshold(cfg):
    """Tight filter + threshold ≥ |filter| → exact path; only allowed gids appear."""
    handle = IndexHandle.open(cfg, "liftus", "demo")
    from src.Semantic.registry import ColumnRef
    from src.Semantic.indexers.base import l2_normalize_rows
    ref = ColumnRef(global_id=-1, table_id="SG_CSV0000000000000007.csv",
                    col_idx=0, col_name="Hospital ID", n_rows=1,
                    source_path="SG_CSV0000000000000007.csv")
    vec = handle.encoder.encode_one(ref)
    q = l2_normalize_rows(vec[None, :].astype(np.float32))

    target = "SG_CSV0000000000000008.csv"
    allowed_gids = set(handle.table_to_gids[target])
    scores, gids = faiss_or_exact_search(
        handle, q, k=5,
        table_filter={target},
        exact_threshold=10,
    )
    real = [int(g) for g in gids[0].tolist() if g >= 0]
    assert real, "exact path returned no real gids; filter had matches"
    assert set(real).issubset(allowed_gids), (
        f"exact path leaked gids outside the filter: {set(real) - allowed_gids}"
    )


def test_open_raises_when_vectors_missing(cfg, tmp_path, monkeypatch):
    """Old indices without embeddings.fp32.npy → clear FileNotFoundError at open."""
    IndexHandle.open(cfg, "liftus", "demo")
    npy_path = cfg.index_dir("liftus", "demo") / "embeddings.fp32.npy"
    backup = npy_path.read_bytes()
    npy_path.unlink()
    reset_semantic_cache()
    try:
        with pytest.raises(FileNotFoundError, match="embeddings.fp32.npy"):
            IndexHandle.open(cfg, "liftus", "demo")
    finally:
        npy_path.write_bytes(backup)
        reset_semantic_cache()


def _su(cfg, *, k: int = 3, exact_threshold: int | None = None) -> SemanticUnionSeekerBase:
    return SemanticUnionSeekerBase(
        _query_df(), k=k, index_name="demo",
        exact_threshold=exact_threshold,
        config_overrides={
            "dataset": cfg.dataset.name,
            "dataset_root": str(cfg.dataset.root),
            "vector_backend": "faiss",
        },
    )


def test_su_exact_path_returns_only_allowlist_tables(cfg):
    """SU with exact_threshold large enough to trigger: returned ids ⊆ filter."""
    su = _su(cfg, k=3, exact_threshold=100)
    su.create_sql_query(_FakeDB(), additionals=" AND TableId IN (1, 2) ")
    _, ids = su._cached
    assert set(ids).issubset({1, 2})
    assert len(ids) > 0


def test_su_wide_filter_falls_back_to_hnsw_path(cfg):
    """Filter wider than threshold → HNSW + post-rollup, same result as no threshold."""
    hnsw_only = _su(cfg, k=3, exact_threshold=None)
    hnsw_only.create_sql_query(_FakeDB(), additionals=" AND TableId IN (0, 1, 2) ")
    _, hnsw_ids = hnsw_only._cached

    triggers_above = _su(cfg, k=3, exact_threshold=1)
    triggers_above.create_sql_query(_FakeDB(), additionals=" AND TableId IN (0, 1, 2) ")
    _, fallback_ids = triggers_above._cached

    assert hnsw_ids == fallback_ids, (
        "wide-filter dispatch should match the threshold=None path exactly"
    )


class _StubSeeker(Seeker):
    HAS_ML_COST_MODEL = False

    def __init__(self, *, k: int, fixed_cost: int, fixed_result: list[int] | None = None):
        super().__init__(k)
        self._fixed_cost = fixed_cost
        self._fixed_result = list(fixed_result or [])
        self.received_additionals: str | None = None

    def cost(self) -> int:
        return self._fixed_cost

    def ml_cost(self, db) -> float:
        return 1.0

    def create_sql_query(self, db, additionals: str = "") -> str:
        self.received_additionals = additionals
        return "SELECT 0 AS TableId WHERE 1=0"

    def run(self, additionals: str = ""):
        self.received_additionals = additionals
        return list(self._fixed_result)


class _TestableSU(SemanticUnionSeekerBase):
    def run(self, additionals: str = ""):
        self.create_sql_query(self.DB, additionals=additionals)
        _, ids = self._cached
        return list(ids[: self.k])


def test_should_use_exact_thresholds_on_gids():
    from src.Semantic.retrieve import should_use_exact
    assert should_use_exact(50, 100) is True
    assert should_use_exact(100, 100) is True
    assert should_use_exact(150, 100) is False
    assert should_use_exact(None, 100) is False
    assert should_use_exact(50, None) is False


def test_gids_dispatch_differs_from_table_count(cfg):
    """The threshold counts gids (columns), not tables."""
    handle = IndexHandle.open(cfg, "liftus", "demo")
    target = "SG_CSV0000000000000008.csv"
    n_cols = len(handle.table_to_gids[target])
    assert n_cols >= 2, "fixture table needs >=2 cols for this test"
    from src.Semantic.retrieve import _count_gids, should_use_exact
    assert _count_gids(handle, {target}) == n_cols
    assert should_use_exact(_count_gids(handle, {target}), n_cols - 1) is False
    assert should_use_exact(_count_gids(handle, {target}), n_cols) is True


def test_handle_applies_ef_search_from_cfg(cfg):
    import faiss
    from dataclasses import replace
    from src.Semantic.retrieve import IndexHandle, reset_semantic_cache
    reset_semantic_cache()
    cfg2 = replace(cfg, faiss_hnsw_ef_search=200)
    handle = IndexHandle.open(cfg2, "liftus", "demo")
    raw = handle.faiss_index.raw_index
    inner = faiss.downcast_index(raw.index) if hasattr(raw, "index") else raw
    assert inner.hnsw.efSearch == 200
    reset_semantic_cache()


def test_intersection_su_mid_chain_with_exact_threshold(cfg):
    """With a tight pushdown SU takes the exact path and K survives downstream."""
    cheap = _StubSeeker(k=3, fixed_cost=1, fixed_result=[1, 2])
    su = _TestableSU(
        _query_df(), k=3, index_name="demo",
        exact_threshold=100,
        config_overrides={
            "dataset": cfg.dataset.name,
            "dataset_root": str(cfg.dataset.root),
            "vector_backend": "faiss",
        },
    )
    expensive = _StubSeeker(k=3, fixed_cost=100)

    inter = Intersection(k=3)
    inter.set_inputs([su, cheap, expensive])
    inter.create_sql_query(inter.DB, additionals="")

    assert expensive.received_additionals is not None
    from src.Semantic.seekers.union_base import _parse_pushdown_ids
    forwarded = _parse_pushdown_ids(expensive.received_additionals)
    assert forwarded is not None and forwarded.issubset({1, 2})
    assert len(forwarded) > 0
