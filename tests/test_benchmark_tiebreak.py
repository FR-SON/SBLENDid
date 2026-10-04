"""Pure decision-logic tests for the SU∩SU tiebreak demonstration."""
import shutil
from pathlib import Path

import pandas as pd
import pytest

from src.Semantic.config import SemanticConfig, SemanticOp, OperatorConfig
from src.Semantic.index_build import build_semantic_index
from src.Semantic.retrieve import reset_semantic_cache

from src.Benchmark.tiebreak.bench import _arm_overrides, classify_pair, summarize

_FIXTURE = Path(__file__).parent / "fixtures" / "semantic"


@pytest.fixture()
def sample_cfg(tmp_path):
    root = tmp_path / "data"
    ad = root / "ds" / "semantic" / "liftus" / "default"
    (ad / "ckpt").mkdir(parents=True)
    shutil.copytree(_FIXTURE / "ckpt", ad / "ckpt", dirs_exist_ok=True)
    shutil.copytree(_FIXTURE / "aspects", ad / "aspects")
    names = sorted(p.name for p in (_FIXTURE / "csvs").glob("*.csv"))
    pd.DataFrame(
        [(i, b) for i, b in enumerate(names)], columns=["table_int_id", "basename"],
    ).astype({"table_int_id": "int64", "basename": "string"}).to_parquet(
        root / "ds" / "blend_index_basenames.parquet", index=False)
    cfg = SemanticConfig.load(
        overrides={"dataset": "ds", "dataset_root": str(root),
                   "faiss_quant": "flat", "vector_backend": "faiss"},
        operators={SemanticOp.SU: OperatorConfig("liftus", "default")})
    reset_semantic_cache()
    build_semantic_index(_FIXTURE / "csvs", "liftus", "default", cfg)
    return cfg


def _cfg(n_cols, k_coarse, gt_ms, pred_ms):
    return {"n_cols": n_cols, "k_coarse": k_coarse, "gt_ms": gt_ms, "pred_ms": pred_ms}


def test_resolve_query_items_sample_mode_uses_index(sample_cfg):
    from src.Benchmark.tiebreak.bench import _iter_query_items
    skipped = {"missing_csv": 0, "unusable_query": 0}
    items = list(_iter_query_items(sample_cfg, "ds", [], 2, 0, "SU", skipped))
    assert 1 <= len(items) <= 2
    for label, df in items:
        assert df.attrs["table_id"] == label
        assert df.shape[0] == 1 and df.shape[1] >= 1
    assert skipped == {"missing_csv": 0, "unusable_query": 0}


def test_cheaper_seeker_runs_first_is_order_A():
    r = classify_pair(_cfg(2, 100, 3.0, 3.0), _cfg(8, 100, 9.0, 9.0))
    assert r["gt"] == "A" and r["placeholder"] == "A" and r["new"] == "A"


def test_placeholder_ties_on_equal_ncols_new_corrects():
    r = classify_pair(_cfg(3, 100, 5.0, 5.0), _cfg(3, 1000, 9.0, 9.0))
    assert r["placeholder"] == "tie"
    assert r["gt"] == "A" and r["new"] == "A"
    assert r["changed"] and r["new_correct"] and not r["placeholder_correct"]
    assert r["corrected"]


def test_placeholder_inverts_when_ncols_misleads_new_corrects():
    r = classify_pair(_cfg(8, 50, 4.0, 4.0), _cfg(2, 2000, 10.0, 10.0))
    assert r["placeholder"] == "B" and r["gt"] == "A" and r["new"] == "A"
    assert r["corrected"]


def test_placeholder_already_correct_is_not_a_correction():
    r = classify_pair(_cfg(2, 100, 3.0, 3.0), _cfg(8, 100, 9.0, 9.0))
    assert r["placeholder_correct"] and r["new_correct"]
    assert not r["changed"] and not r["corrected"]


def test_summarize_counts_ties_inversions_corrections():
    pairs = [
        (_cfg(3, 100, 5.0, 5.0), _cfg(3, 1000, 9.0, 9.0)),
        (_cfg(8, 50, 4.0, 4.0), _cfg(2, 2000, 10.0, 10.0)),
        (_cfg(2, 100, 3.0, 3.0), _cfg(8, 100, 9.0, 9.0)),
    ]
    rows = [{**classify_pair(a, b), "a_gt_ms": a["gt_ms"], "b_gt_ms": b["gt_ms"]}
            for a, b in pairs]
    s = summarize(rows, "santos", queries=["q"], configs=[1, 2, 3],
                  skipped={"missing_csv": 0, "unusable_query": 0},
                  k=10, k_coarse_grid=[100, 1000], repeats=1, seed=0)
    assert s["coverage"]["pairs"] == 3
    assert s["placeholder_ties"] == 1
    assert s["placeholder_inversions"] == 1
    assert s["decisions_corrected"] == 2
    assert s["new_correct"] == 3


class _Cfg:
    def __init__(self, backend, ef=64):
        self.vector_backend = backend
        self.faiss_hnsw_ef_search = ef


def test_faiss_arm_ties_the_beam_to_the_fetch():
    """On faiss the ground truth is measured along ef == k_coarse, as the model was fitted."""
    assert _arm_overrides("lake", _Cfg("faiss"), 250) == {
        "dataset": "lake", "faiss_hnsw_ef_search": 250, "query_encode": "off"}


def test_pgvector_arm_keeps_the_configured_beam():
    assert _arm_overrides("lake", _Cfg("pgvector"), 250) == {
        "dataset": "lake", "query_encode": "off"}


def test_summary_records_the_beam_plane():
    s = summarize([], "lake", queries=[], configs=[], skipped={}, k=10,
                  k_coarse_grid=[100, 1000], repeats=3, seed=0,
                  beam_rule="ef=k_coarse")
    assert s["params"]["beam_rule"] == "ef=k_coarse"
