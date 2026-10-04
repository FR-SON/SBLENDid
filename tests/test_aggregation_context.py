import importlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.Semantic.aggregators.base import AggregationContext
from src.Semantic.aggregators.decay_vote import DecayVoteAggregator
from tests.test_semantic_union_seeker import blend_env  # noqa: F401


FIXTURE = Path(__file__).parent / "fixtures" / "semantic"


def _ctx():
    return AggregationContext(
        qmat=np.eye(2, dtype=np.float32),
        candidate_vectors_fn=lambda t: np.eye(2, dtype=np.float32),
        query_table_id="q.csv",
    )


def test_decay_vote_ignores_ctx():
    hits = [(["a.csv", "b.csv"], [0.1, 0.2])]
    agg = DecayVoteAggregator()
    assert agg.aggregate(hits, k=5) == agg.aggregate(hits, k=5, ctx=_ctx())


class RecordingAgg:
    name = "recording"
    requires_ctx = True

    def __init__(self):
        self.ctx = None

    def aggregate(self, per_col_hits, k, *, ctx=None):
        self.ctx = ctx
        return []


def test_search_builds_ctx_and_accepts_aggregator_override(blend_env):  # noqa: F811
    importlib.import_module("src.Plan")
    from src.Semantic.config import SemanticConfig, SemanticOp
    from src.Semantic.retrieve import IndexHandle, reset_semantic_cache, search

    cfg = SemanticConfig.load()
    reset_semantic_cache()
    handle = IndexHandle.open(cfg, "liftus", cfg.operator(SemanticOp.SU).index_name)

    qtid = "SG_CSV0000000000000007.csv"
    df = pd.read_csv(FIXTURE / "csvs" / qtid, dtype=str, keep_default_na=False)
    df.attrs["table_id"] = qtid

    inst = RecordingAgg()
    out = search(handle, df, k=3, query_table_id=qtid, aggregator=inst)

    assert out == []
    assert inst.ctx is not None
    assert inst.ctx.qmat.shape == (df.shape[1], handle.dim)
    assert np.allclose(np.linalg.norm(inst.ctx.qmat, axis=1), 1.0, atol=1e-5)

    known = "SG_CSV0000000000000008.csv"
    mat = inst.ctx.candidate_vectors_fn(known)
    assert mat.shape == (len(handle.table_to_gids[known]), handle.dim)
