"""pgvector search() honours the Difference NOT IN exclusion under both pg_pushdown_modes."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.Operators import Seekers  # noqa: F401
from src.Semantic.config import SemanticConfig
from src.Semantic.retrieve import IndexHandle, search, reset_semantic_cache

_QT = "SG_CSV0000000000000007.csv"
_FIX = Path(__file__).parent / "fixtures" / "semantic"


def _pg_query_df():
    df = pd.read_csv(_FIX / "csvs" / _QT, dtype=str, keep_default_na=False)
    df.attrs["table_id"] = _QT
    return df


def _pg_handle(mode):
    cfg = SemanticConfig.load(overrides={
        "dataset": "pytest_pg", "vector_backend": "pgvector",
        "pg_pushdown_mode": mode,
    })
    return IndexHandle.open(cfg, "liftus", "demo")


@pytest.mark.parametrize("mode", ["prefilter", "postfilter"])
def test_pg_search_excludes_not_in(pg_semantic_index, mode):
    reset_semantic_cache()
    handle = _pg_handle(mode)
    df = _pg_query_df()
    full = search(handle, df, k=3, query_table_id=_QT)
    assert full, "pgvector search returned no candidates"
    drop = full[0]
    drop_name = handle.int_to_table[drop]
    got = search(handle, df, k=3, query_table_id=_QT, exclude_filter={drop_name})
    assert drop not in got, f"mode={mode}: excluded {drop} leaked into {got}"


@pytest.mark.parametrize("mode", ["prefilter", "postfilter"])
def test_pg_search_slot_refill_no_stealing(pg_semantic_index, mode):
    reset_semantic_cache()
    handle = _pg_handle(mode)
    df = _pg_query_df()
    full = search(handle, df, k=2, query_table_id=_QT)
    if len(full) < 2:
        pytest.skip("need >=2 candidates")
    top_name = handle.int_to_table[full[0]]
    got = search(handle, df, k=1, query_table_id=_QT, exclude_filter={top_name})
    assert got == [full[1]], f"mode={mode}: slot not refilled ({got})"


def test_pgvector_store_excludes_in_sql(pg_semantic_index):
    reset_semantic_cache()
    cfg = SemanticConfig.load(overrides={
        "dataset": "pytest_pg", "vector_backend": "pgvector",
        "pg_pushdown_mode": "prefilter",
    })
    handle = IndexHandle.open(cfg, "liftus", "demo")
    qgid = next(iter(handle.gid_to_table))
    excl_table = handle.gid_to_table[qgid]
    excl_int = handle.table_to_int_id[excl_table]
    scores, gids = handle.pg.search_gid(
        qgid, 50, allowed_int_ids=None,
        excluded_int_ids={excl_int}, exact=False)
    returned = {handle.gid_to_table[int(g)] for g in gids[0] if int(g) >= 0}
    assert excl_table not in returned, "NOT IN clause did not prune in SQL"
