"""Tests for semantic_assist bench plan."""
from __future__ import annotations

import pytest

from src import paths

_DS = "opendata-split-11"


def test_plan_is_registered():
    from src.Benchmark.plans import PLANS
    assert "semantic_assist" in PLANS
    spec = PLANS["semantic_assist"]
    assert spec.point and callable(spec.run)


@pytest.mark.skipif(
    not (paths.datasets_root() / _DS / "blend_index_basenames.parquet").exists(),
    reason="opendata-split-11 not on disk")
def test_smoke_join(tmp_path):
    from src.Benchmark.db import open_dataset_db
    from src.Benchmark.plans import PLANS
    from src.Benchmark.plans.registry import PlanContext

    try:
        db = open_dataset_db(_DS)
    except Exception as e:
        pytest.skip(f"no live DB for {_DS}: {e}")

    try:
        ctx = PlanContext(
            dataset=_DS, db=db, task="join", k=10, seed=0, out_dir=tmp_path,
            git_sha="test", config_snapshot={}, limit=2)
        result = PLANS["semantic_assist"].run(ctx)
    finally:
        db.close()

    assert [a.name for a in result.arms] == [
        "seeker_only", "keyword_only", "union_assist", "precision_assist", "both"]
    assert "k_sweep" in result.meta
    for k in ["5", "10", "20", "50"]:
        assert k in result.meta["k_sweep"], f"k={k} missing from k_sweep"
        entry = result.meta["k_sweep"][k]
        assert "overall" in entry and "oracle_subset" in entry
    assert result.meta.get("rank_metrics_note") is not None
