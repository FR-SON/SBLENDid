"""Tests for semantic_reach_complement bench plan."""
from __future__ import annotations

import pytest


def _rrf():
    from src.Benchmark.plans.semantic_reach_complement import fuse_rrf
    return fuse_rrf


def test_fuse_rrf_equal_weights_interleaves():
    fuse_rrf = _rrf()
    leg0 = ["a", "b", "c"]
    leg1 = ["a", "d", "e"]
    result = fuse_rrf([leg0, leg1], [1.0, 1.0], kc=60)
    assert result[0] == "a"
    assert set(result) == {"a", "b", "c", "d", "e"}


def test_fuse_rrf_w_1_0_is_leg0():
    fuse_rrf = _rrf()
    leg0 = ["x", "y", "z"]
    leg1 = ["p", "q", "r"]
    result = fuse_rrf([leg0, leg1], [1.0, 0.0], kc=60)
    assert result == ["x", "y", "z", "p", "q", "r"]


def test_fuse_rrf_w_0_1_is_leg1():
    fuse_rrf = _rrf()
    leg0 = ["x", "y", "z"]
    leg1 = ["p", "q", "r"]
    result = fuse_rrf([leg0, leg1], [0.0, 1.0], kc=60)
    assert result == ["p", "q", "r", "x", "y", "z"]


def test_fuse_rrf_high_in_both_wins():
    fuse_rrf = _rrf()
    leg0 = ["a", "z"]
    leg1 = ["a", "w"]
    result = fuse_rrf([leg0, leg1], [1.0, 1.0], kc=60)
    assert result[0] == "a"


def test_fuse_rrf_cut_respected():
    fuse_rrf = _rrf()
    leg0 = ["a", "b", "c", "d"]
    leg1 = ["e", "f", "g", "h"]
    result = fuse_rrf([leg0, leg1], [1.0, 1.0], kc=60, cut=3)
    assert len(result) == 3


def test_fuse_rrf_absent_from_leg_contributes_zero():
    fuse_rrf = _rrf()
    leg0 = ["solo"]
    leg1 = ["other"]
    result = fuse_rrf([leg0, leg1], [1.0, 1.0], kc=60)
    assert "solo" in result
    assert "other" in result


def test_fuse_rrf_empty_legs():
    fuse_rrf = _rrf()
    result = fuse_rrf([[], []], [1.0, 1.0])
    assert result == []


def test_order_preserving_dedup():
    from src.Benchmark.plans.semantic_reach_complement import _dedup_ordered
    lst = ["a", "b", "a", "c", "b", "d"]
    assert _dedup_ordered(lst) == ["a", "b", "c", "d"]


def test_order_preserving_dedup_empty():
    from src.Benchmark.plans.semantic_reach_complement import _dedup_ordered
    assert _dedup_ordered([]) == []


def test_plan_is_registered():
    from src.Benchmark.plans import PLANS
    assert "semantic_reach_complement" in PLANS
    spec = PLANS["semantic_reach_complement"]
    assert spec.point and callable(spec.run)


_DS = "opendata-split-11"


def _dataset_on_disk():
    from src import paths
    return (paths.datasets_root() / _DS / "blend_index_basenames.parquet").exists()


@pytest.mark.skipif(not _dataset_on_disk(), reason="opendata-split-11 not on disk")
@pytest.mark.parametrize("task", ["join"])
def test_smoke_runs_three_arms(task, tmp_path):
    from src.Benchmark.db import open_dataset_db
    from src.Benchmark.plans import PLANS
    from src.Benchmark.plans.registry import PlanContext
    try:
        db = open_dataset_db(_DS)
    except Exception as e:
        pytest.skip(f"no live DB for {_DS}: {e}")
    try:
        ctx = PlanContext(
            dataset=_DS, db=db, task=task, k=10, seed=0, out_dir=tmp_path,
            git_sha="test", config_snapshot={}, limit=1)
        result = PLANS["semantic_reach_complement"].run(ctx)
    finally:
        db.close()
    assert result.task == task
    arm_names = [a.name for a in result.arms]
    assert "semantic" in arm_names
    assert "syntactic" in arm_names
    assert "hybrid" in arm_names
    assert "strata" in result.meta
    assert "weight_sweep" in result.meta
