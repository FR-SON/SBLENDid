"""GT rows whose candidate_table equals the query_table are dropped from the relevant set."""

from pathlib import Path

import pandas as pd

from src.Semantic.config import OperatorConfig, SemanticConfig, SemanticOp
from src.Benchmark.judit.loading import load_join_qgt, load_union_qgt

PREFIX = "px"
_ROOT = Path(__file__).resolve().parents[1]

_GT_LOADER_FILES = [
    "src/Benchmark/judit/loading.py",
    "scripts/bench_sc_seeker_metrics.py",
]


def test_every_gt_read_is_self_dropped():
    """Each `pd.read_csv(... _ground_truth.csv)` must have a `drop_self_gt(gt)` within a few lines."""
    for rel in _GT_LOADER_FILES:
        lines = (_ROOT / rel).read_text().splitlines()
        reads = [i for i, ln in enumerate(lines)
                 if "_ground_truth.csv" in ln and "read_csv" in ln]
        assert reads, f"{rel}: no GT read found (test list stale?)"
        for i in reads:
            window = "\n".join(lines[i:i + 4])
            assert "drop_self_gt(gt)" in window, (
                f"{rel}:{i + 1} reads GT without a nearby drop_self_gt(gt)")


def _cfg(tmp_path, name="ds"):
    root = tmp_path / "data"
    cfg = SemanticConfig.load(
        path=tmp_path / "missing.ini",
        overrides={"dataset": name, "dataset_root": str(root)},
        operators={SemanticOp.SU: OperatorConfig("liftus", "demo"),
                   SemanticOp.SJ: OperatorConfig("liftus", "demo")},
    )
    return cfg, root / name


def _write(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def test_load_union_qgt_drops_self_candidate(tmp_path):
    cfg, base = _cfg(tmp_path)
    _write(base / "query" / f"{PREFIX}_union_query.csv",
           pd.DataFrame({"query_table": ["a.csv"]}))
    _write(base / "groundtruth" / f"{PREFIX}_union_ground_truth.csv",
           pd.DataFrame({"query_table": ["a.csv", "a.csv"],
                         "candidate_table": ["a.csv", "c.csv"]}))
    _write(base / "csvs" / "a.csv", pd.DataFrame(columns=["col0", "col1"]))
    reg = {"a.csv": {"col0", "col1"}, "c.csv": {"col0"}}

    qgt = load_union_qgt(cfg, PREFIX, reg)
    assert qgt.relevant_by_q["a.csv"] == {"c.csv"}


def test_load_join_qgt_drops_self_candidate(tmp_path):
    cfg, base = _cfg(tmp_path)
    _write(base / "query" / f"{PREFIX}_join_query.csv",
           pd.DataFrame({"query_table": ["a.csv"], "query_column": ["col0"]}))
    _write(base / "groundtruth" / f"{PREFIX}_join_ground_truth.csv",
           pd.DataFrame({"query_table": ["a.csv", "a.csv"],
                         "candidate_table": ["z.csv", "a.csv"],
                         "query_column": ["col0", "col0"],
                         "candidate_column": ["col0", "col1"]}))
    reg = {"a.csv": {"col0", "col1"}, "z.csv": {"col0"}}

    qgt = load_join_qgt(cfg, PREFIX, reg)
    assert qgt.relevant_by_q[("a.csv", "col0")] == {"z.csv"}
    assert qgt.relevant_cols_by_q[("a.csv", "col0")] == {("z.csv", "col0")}
