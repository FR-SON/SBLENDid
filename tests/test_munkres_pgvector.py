"""Live munkres-on-pgvector smoke for the need_ctx branch of _search_pgvector."""
from __future__ import annotations

import pandas as pd

from src.Semantic.aggregators.munkres import MunkresAggregator
from src.Semantic.config import SemanticConfig
from src.Semantic.retrieve import IndexHandle, search, reset_semantic_cache


def test_munkres_search_pgvector(pg_semantic_index):
    dataset, approach, index_name = pg_semantic_index
    reset_semantic_cache()
    cfg = SemanticConfig.load(overrides={
        "dataset": dataset, "vector_backend": "pgvector",
        "faiss_quant": "flat"})
    handle = IndexHandle.open(cfg, approach, index_name)

    qtid = next(iter(handle.table_col_to_gid))[0]
    cols = [c for (t, c) in handle.table_col_to_gid if t == qtid]
    df = pd.DataFrame({c: ["x"] for c in cols})
    df.attrs["table_id"] = qtid

    agg = MunkresAggregator(threshold=0.0)
    ids = search(handle, df, k=3, query_table_id=qtid, aggregator=agg)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert set(ids) <= set(handle.table_to_int_id.values())
    assert agg.n_candidates_scored_total >= 1

    ids_default = search(handle, df, k=3, query_table_id=qtid)
    assert isinstance(ids_default, list)
