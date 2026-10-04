from pathlib import Path
import pytest
import pandas as pd

pytestmark = pytest.mark.skipif(
    not Path("tests/fixtures/semantic").exists(),
    reason="semantic fixture not built",
)


def _open(cfg_backend):
    from src.Semantic.config import SemanticConfig
    from src.Semantic.retrieve import IndexHandle, reset_semantic_cache
    reset_semantic_cache()
    cfg = SemanticConfig.load(overrides={"dataset": "pytest_pg",
                                         "vector_backend": cfg_backend})
    return cfg, IndexHandle.open(cfg, "liftus", "demo")


def test_pg_exact_matches_faiss_setwise(pg_semantic_index):
    """On the exact path both backends must return the same top-k set (ties may reorder)."""
    from src.Semantic.retrieve import search
    _, h_f = _open("faiss")
    _, h_p = _open("pgvector")
    qt, qc = next(iter(h_f.table_col_to_gid.keys()))
    df = pd.DataFrame({qc: ["x"]}); df.attrs["table_id"] = qt
    full_f = set(h_f.table_to_int_id)
    full_p = set(h_p.table_to_int_id)
    a = search(h_f, df, k=10, query_table_id=qt, table_filter=full_f, exact_threshold=10**9)
    b = search(h_p, df, k=10, query_table_id=qt, table_filter=full_p, exact_threshold=10**9)
    assert len(set(a) & set(b)) >= min(len(a), len(b)) - 1
    assert set(a[:5]) == set(b[:5])


def test_pg_encodes_unindexed_query_column(pg_semantic_index, capsys):
    """The pgvector branch searches an encoded vector via search_vec and ranks real tables."""
    import numpy as np
    from src.Semantic.retrieve import search
    _, h_p = _open("pgvector")

    class _Fake:
        na_cell = ""
        dim = h_p.dim

        def encode_batch(self, refs, *, cells_per_ref=None):
            return {r.global_id: np.random.default_rng(r.global_id)
                    .standard_normal(h_p.dim).astype(np.float32) for r in refs}

    h_p.__dict__["encoder"] = _Fake()
    df = pd.DataFrame({"name": ["alice", "bob"], "blank": ["", None]})
    ids = search(h_p, df, k=2, k_coarse=5, query_table_id="external.csv", query_encode="auto")
    assert 0 < len(ids) <= 2 and set(ids) <= set(h_p.table_to_int_id.values())
    err = capsys.readouterr().err
    assert "[SEMANTIC][QUERY-ENCODE]" in err and "skipped=['blank']" in err
