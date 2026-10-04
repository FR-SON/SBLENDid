
import pytest

from src import paths

_DS = "opendata-split-11"


def test_plan_is_registered():
    from src.Benchmark.plans import PLANS
    assert "semantic_oracle_repair" in PLANS
    spec = PLANS["semantic_oracle_repair"]
    assert spec.point and callable(spec.run)


@pytest.mark.skipif(
    not (paths.datasets_root() / _DS / "blend_index_basenames.parquet").exists(),
    reason="opendata-split-11 not on disk")
@pytest.mark.parametrize("task", ["union", "join"])
def test_smoke_runs_four_arms(task, tmp_path):
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
            git_sha="test", config_snapshot={}, limit=2)
        result = PLANS["semantic_oracle_repair"].run(ctx)
    finally:
        db.close()
    assert result.task == task
    assert [a.name for a in result.arms] == [
        "seeker_only", "keyword_only", "recall_repair", "precision_repair", "both"]
    for a in result.arms:
        assert 0.0 <= a.summary["mean_recall"] <= 1.0
        assert 0.0 <= a.summary["mean_precision"] <= 1.0
