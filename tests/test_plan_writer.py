import json

from src.Benchmark.plans.registry import ArmResult, PlanResult
from src.Benchmark.plans.writer import write_plan_results


def _result():
    arm = ArmResult(
        name="seeker_only",
        rows=[{"query": "q.csv", "arm": "seeker_only", "precision": 0.4,
               "recall": 0.5, "ndcg": 0.6, "ap": 0.3, "rr": 0.5,
               "n_relevant": 4, "n_retrieved": 10, "hits": 2,
               "runtime_ms": 12.0}],
        summary={"evaluated": 1, "mean_precision": 0.4, "mean_recall": 0.5,
                 "mean_ndcg": 0.6, "MAP": 0.3, "MRR": 0.5,
                 "mean_runtime_ms": 12.0},
    )
    return PlanResult(
        plan="semantic_oracle_repair", point="the point", dataset="ds",
        task="union", arms=[arm],
        deltas={"recall_repair_minus_seeker_only": {"mean_recall": 0.1}},
        meta={"git_sha": "abc1234", "seed": 0, "k": 10, "limit": None,
              "evaluated": 1, "oracle_entries": 0, "config_snapshot": {}},
    )


def test_writer_creates_json_csv_md(tmp_path):
    json_path = write_plan_results(_result(), k=10, out_dir=tmp_path)
    assert json_path.exists() and json_path.suffix == ".json"
    assert list(tmp_path.glob("*.csv"))
    assert list(tmp_path.glob("*.md"))
    data = json.loads(json_path.read_text())
    assert data["plan"] == "semantic_oracle_repair"
    assert data["meta"]["git_sha"] == "abc1234"
    assert data["arms"][0]["summary"]["mean_recall"] == 0.5
