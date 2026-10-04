"""The exact-reference cache key must include the encoder, since gid sets match across indexes."""

from __future__ import annotations

import numpy as np
import pytest

from src.Semantic.config import OperatorConfig, SemanticConfig, SemanticOp
from src.Benchmark import ann_frontier as AF


@pytest.fixture()
def two_index_cfg(tmp_path, monkeypatch):
    """One lake, two approaches over the SAME gid space, different vectors."""
    root, name = tmp_path / "data", "ds"
    for approach, seed in (("liftus", 1), ("snoopy", 2)):
        d = root / name / "semantic" / approach / "default" / "index"
        d.mkdir(parents=True)
        v = np.random.default_rng(seed).normal(size=(32, 8)).astype(np.float32)
        np.save(d / "embeddings.fp32.npy", v)
    monkeypatch.setattr(AF, "bench_results_root", lambda _n: root / name / "bench_results")
    return SemanticConfig.load(
        path=tmp_path / "missing.ini",
        overrides={"dataset": name, "dataset_root": str(root)},
        operators={SemanticOp.SU: OperatorConfig("liftus", "default"),
                   SemanticOp.SJ: OperatorConfig("snoopy", "default")},
    )


def test_reference_is_keyed_by_encoder(two_index_cfg):
    cfg = two_index_cfg
    qs = [AF.QueryTable("t0.csv", tuple(range(8)), tuple(f"c{i}" for i in range(8)))]

    su = AF.exact_reference(qs, 4, cfg, SemanticOp.SU)
    sj = AF.exact_reference(qs, 4, cfg, SemanticOp.SJ)

    assert su != sj, "SJ reused the SU reference: cache key is missing the encoder"
