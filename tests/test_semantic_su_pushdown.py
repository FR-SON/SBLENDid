import shutil
from pathlib import Path

import pandas as pd
import pytest

# Import Seekers first: importing union_base directly is a circular import.
from src.Operators import Seekers  # noqa: F401

from src.Semantic.config import SemanticConfig
from src.Semantic.index_build import build_semantic_index
from src.Semantic.retrieve import IndexHandle, search, apply_rollup_filters, reset_semantic_cache
from src.Semantic.seekers.union_base import (
    SemanticUnionSeekerBase, _parse_pushdown_ids, _parse_pushdown_exclude_ids,
)


FIXTURE = Path(__file__).parent / "fixtures" / "semantic"


def test_parse_no_pushdown_returns_none():
    assert _parse_pushdown_ids("") is None
    assert _parse_pushdown_ids("   ") is None
    assert _parse_pushdown_ids(" AND foo = 1 ") is None


def test_parse_single_in_clause():
    assert _parse_pushdown_ids(" AND TableId IN (1, 2, 3) ") == {1, 2, 3}


def test_parse_case_insensitive():
    assert _parse_pushdown_ids(" and tableid in (5) ") == {5}


def test_parse_multiple_clauses_intersect():
    s = " AND TableId IN (1, 2, 3, 4) AND TableId IN (3, 4, 5) "
    assert _parse_pushdown_ids(s) == {3, 4}


def test_parse_empty_in_clause_is_empty_set():
    assert _parse_pushdown_ids(" AND TableId IN () ") == set()


def test_parse_disjoint_clauses_is_empty_set():
    assert _parse_pushdown_ids(" AND TableId IN (1) AND TableId IN (2) ") == set()


def test_parse_exclude_absent_or_empty_is_empty_set():
    assert _parse_pushdown_exclude_ids("") == set()
    assert _parse_pushdown_exclude_ids(" AND foo = 1 ") == set()
    assert _parse_pushdown_exclude_ids(" AND TableId IN (1, 2) ") == set()
    assert _parse_pushdown_exclude_ids(" AND TableId NOT IN () ") == set()


def test_parse_exclude_single_clause():
    assert _parse_pushdown_exclude_ids(" AND TableId NOT IN (1, 2, 3) ") == {1, 2, 3}


def test_parse_exclude_case_insensitive():
    assert _parse_pushdown_exclude_ids(" and tableid not in (5) ") == {5}


def test_parse_exclude_multiple_clauses_union():
    s = " AND TableId NOT IN (1, 2) AND TableId NOT IN (2, 3) "
    assert _parse_pushdown_exclude_ids(s) == {1, 2, 3}


def test_parse_exclude_ignores_non_numeric():
    assert _parse_pushdown_exclude_ids(" AND TableId NOT IN ('a', 4) ") == {4}


def test_parse_in_and_not_in_are_independent():
    s = " AND TableId IN (1, 2) AND TableId NOT IN (2, 3) "
    assert _parse_pushdown_ids(s) == {1, 2}
    assert _parse_pushdown_exclude_ids(s) == {2, 3}


def test_parse_positive_returns_none_on_pure_not_in():
    assert _parse_pushdown_ids(" AND TableId NOT IN (1, 2) ") is None


def _basenames():
    return sorted(p.name for p in (FIXTURE / "csvs").glob("*.csv"))


def _write_basenames(p: Path) -> None:
    pd.DataFrame(
        [(i, b) for i, b in enumerate(_basenames())],
        columns=["table_int_id", "basename"],
    ).astype({"table_int_id": "int64", "basename": "string"}).to_parquet(p, index=False)


@pytest.fixture()
def cfg(tmp_path):
    root = tmp_path / "data"
    dataset_dir = root / "test_ds"
    approach_dir = dataset_dir / "semantic" / "liftus" / "demo"
    (approach_dir / "ckpt").mkdir(parents=True)
    shutil.copytree(FIXTURE / "ckpt", approach_dir / "ckpt", dirs_exist_ok=True)
    shutil.copytree(FIXTURE / "aspects", approach_dir / "aspects")

    _write_basenames(dataset_dir / "blend_index_basenames.parquet")

    cfg = SemanticConfig.load(
        path=tmp_path / "missing.ini",
        overrides={
            "dataset": "test_ds",
            "dataset_root": str(root),
            "faiss_quant": "flat",
        },
    )
    reset_semantic_cache()
    build_semantic_index(FIXTURE / "csvs", "liftus", "demo", cfg)
    return cfg


class _FakeDB:
    dbms = "duckdb"


def _su(cfg, k: int = 3) -> SemanticUnionSeekerBase:
    df = pd.read_csv(FIXTURE / "csvs" / "SG_CSV0000000000000007.csv",
                     dtype=str, keep_default_na=False)
    df.attrs["table_id"] = "SG_CSV0000000000000007.csv"
    return SemanticUnionSeekerBase(
        df, k=k, index_name="demo",
        config_overrides={
            "dataset": cfg.dataset.name,
            "dataset_root": str(cfg.dataset.root),
            "vector_backend": "faiss",
        },
    )


def test_su_returns_full_top_k_without_additionals(cfg):
    su = _su(cfg, k=3)
    su.create_sql_query(_FakeDB(), additionals="")
    _, full = su._cached
    assert set(full).issubset({0, 1, 2})
    assert len(full) > 0


def test_su_pushdown_restricts_output_to_allowlist(cfg):
    full = _su(cfg, k=3)
    full.create_sql_query(_FakeDB(), additionals="")
    _, full_ids = full._cached
    assert set(full_ids).issubset({0, 1, 2})

    pushed = _su(cfg, k=3)
    pushed.create_sql_query(_FakeDB(), additionals=" AND TableId IN (1, 2) ")
    _, pushed_ids = pushed._cached
    assert set(pushed_ids).issubset({1, 2})
    assert 0 not in pushed_ids


def test_su_empty_pushdown_short_circuits(cfg):
    su = _su(cfg, k=3)
    su.create_sql_query(_FakeDB(), additionals=" AND TableId IN () ")
    _, ids = su._cached
    assert ids == []


def test_su_pushdown_with_no_known_ids_short_circuits(cfg):
    su = _su(cfg, k=3)
    su.create_sql_query(_FakeDB(), additionals=" AND TableId IN (999, 1000) ")
    _, ids = su._cached
    assert ids == []


from src.Operators.Combiners.Intersection import Intersection
from src.Operators.Seekers.SeekerBase import Seeker


class _StubSeeker(Seeker):
    """Cost-controllable stub: cheap returns fixed ids, expensive captures its additionals."""
    HAS_ML_COST_MODEL = False

    def __init__(self, *, k: int, fixed_cost: int, fixed_result: list[int] | None = None):
        super().__init__(k)
        self._fixed_cost = fixed_cost
        self._fixed_result = list(fixed_result or [])
        self.received_additionals: str | None = None

    def cost(self) -> int:
        return self._fixed_cost

    def ml_cost(self, db) -> float:
        return 1.0

    def create_sql_query(self, db, additionals: str = "") -> str:
        self.received_additionals = additionals
        return "SELECT 0 AS TableId WHERE 1=0"

    def run(self, additionals: str = ""):
        self.received_additionals = additionals
        return list(self._fixed_result)


class _TestableSU(SemanticUnionSeekerBase):
    """Reads _cached directly so Intersection.run() needs no DB execution."""

    def run(self, additionals: str = ""):
        self.create_sql_query(self.DB, additionals=additionals)
        _, ids = self._cached
        return list(ids[: self.k])


def _testable_su(cfg, k: int = 3) -> _TestableSU:
    df = pd.read_csv(FIXTURE / "csvs" / "SG_CSV0000000000000007.csv",
                     dtype=str, keep_default_na=False)
    df.attrs["table_id"] = "SG_CSV0000000000000007.csv"
    return _TestableSU(
        df, k=k, index_name="demo",
        config_overrides={
            "dataset": cfg.dataset.name,
            "dataset_root": str(cfg.dataset.root),
            "vector_backend": "faiss",
        },
    )


def test_intersection_su_mid_chain_propagates_upstream_filter(cfg):
    cheap = _StubSeeker(k=3, fixed_cost=1, fixed_result=[1, 2])
    su = _testable_su(cfg, k=3)
    expensive = _StubSeeker(k=3, fixed_cost=100)

    inter = Intersection(k=3)
    inter.set_inputs([su, cheap, expensive])
    inter.create_sql_query(inter.DB, additionals="")

    assert expensive.received_additionals is not None, (
        "Intersection never reached the expensive (last-position) seeker"
    )
    ids = _parse_pushdown_ids(expensive.received_additionals)
    assert ids is not None, (
        f"expensive expected an IN clause, got {expensive.received_additionals!r}"
    )
    assert ids.issubset({1, 2}), (
        f"upstream K={{1,2}} lost through SU mid-chain; expensive received "
        f"{expensive.received_additionals!r}"
    )


def test_intersection_su_mid_chain_drops_unknown_upstream_ids(cfg):
    cheap = _StubSeeker(k=3, fixed_cost=1, fixed_result=[999, 1000])
    su = _testable_su(cfg, k=3)
    expensive = _StubSeeker(k=3, fixed_cost=100)

    inter = Intersection(k=3)
    inter.set_inputs([su, cheap, expensive])
    sql = inter.create_sql_query(inter.DB, additionals="")

    assert "WHERE 1=0" in sql, (
        f"empty-SU mid-chain should short-circuit Intersection; got {sql!r}"
    )
    assert expensive.received_additionals is None, (
        "expensive seeker should not be invoked once SU returns empty"
    )


def test_apply_rollup_filters_allow_then_deny():
    ranked = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    assert apply_rollup_filters(ranked, {"a", "b"}, {"a"}, allow_active=True) == [("b", 0.8)]
    assert apply_rollup_filters(ranked, {"a", "b"}, {"c"}, allow_active=False) == [
        ("a", 0.9), ("b", 0.8)]
    assert apply_rollup_filters(ranked, None, None, allow_active=True) == ranked
    assert apply_rollup_filters(ranked, None, set(), allow_active=True) == ranked


def _demo_query_df():
    df = pd.read_csv(FIXTURE / "csvs" / "SG_CSV0000000000000007.csv",
                     dtype=str, keep_default_na=False)
    df.attrs["table_id"] = "SG_CSV0000000000000007.csv"
    return df


def test_search_exclude_filter_drops_table_faiss(cfg):
    handle = IndexHandle.open(cfg, "liftus", "demo")
    df = _demo_query_df()
    full = search(handle, df, k=3, query_table_id=df.attrs["table_id"])
    assert full
    drop = full[0]
    drop_name = handle.int_to_table[drop]
    got = search(handle, df, k=3, query_table_id=df.attrs["table_id"],
                 exclude_filter={drop_name})
    assert drop not in got


def _su_ids(cfg, k, additionals):
    su = _su(cfg, k=k)
    su.create_sql_query(_FakeDB(), additionals=additionals)
    return su._cached[1]


def test_su_minuend_excludes_not_in(cfg):
    full = _su_ids(cfg, 3, "")
    assert full
    drop = full[0]
    got = _su_ids(cfg, 3, f" AND TableId NOT IN ({drop}) ")
    assert drop not in got


def test_su_minuend_slot_refill_no_stealing(cfg):
    full = _su_ids(cfg, 2, "")
    assert len(full) >= 2
    top, second = full[0], full[1]
    got = _su_ids(cfg, 1, f" AND TableId NOT IN ({top}) ")
    assert got == [second]


def test_su_combined_in_and_not_in(cfg):
    full = _su_ids(cfg, 2, "")
    assert len(full) >= 2
    a, b = full[0], full[1]
    got = _su_ids(cfg, 2, f" AND TableId IN ({a}, {b}) AND TableId NOT IN ({a}) ")
    assert a not in got and b in got
