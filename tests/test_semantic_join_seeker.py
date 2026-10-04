"""Integration tests for SemanticJoin over the snoopy plugin with a fake fasttext embedder."""

from __future__ import annotations
import configparser
import importlib
import shutil
from pathlib import Path

import pandas as pd
import pytest

from scripts.create_blend_csv_index import ingest_csv_dir
from src.Semantic.config import OperatorConfig, SemanticConfig, SemanticOp
from src.Semantic.index_build import build_semantic_index
from src.Semantic.retrieve import reset_semantic_cache
from tests.conftest import drop_blend_modules
from tests.fixtures.semantic_join._fake_embedder import fake_embedder


FIXTURE = Path(__file__).parent / "fixtures" / "semantic_join"


@pytest.fixture()
def sj_env(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "data" / "test_sj"
    csvs_dir = dataset_dir / "csvs"
    csvs_dir.mkdir(parents=True)
    for src in (FIXTURE / "csvs").iterdir():
        shutil.copy(src, csvs_dir / src.name)

    db_path = dataset_dir / "blend.duckdb"
    sidecar_path, n_tables = ingest_csv_dir(csvs_dir, db_path)
    assert n_tables == 5

    approach_dir = dataset_dir / "semantic" / "snoopy" / "default" / "ckpt"
    approach_dir.mkdir(parents=True)
    shutil.copy(FIXTURE / "ckpt" / "tiny.pth", approach_dir / "tiny.pth")
    shutil.copy(FIXTURE / "ckpt" / "tiny.json", approach_dir / "tiny.json")

    fake_bin = tmp_path / "fake.bin"
    fake_bin.touch()
    monkeypatch.setenv("BLEND_SNOOPY_FASTTEXT_PATH", str(fake_bin))
    import src.Semantic.encoders.snoopy as _snoopy_mod
    monkeypatch.setattr(_snoopy_mod, "_default_fasttext_embedder",
                        lambda d, fasttext_path: fake_embedder(d))

    semantic_cfg = SemanticConfig.load(
        path=tmp_path / "missing.ini",
        overrides={
            "dataset": "test_sj",
            "dataset_root": str(tmp_path / "data"),
        },
        operators={SemanticOp.SJ: OperatorConfig("snoopy", "default")},
    )
    reset_semantic_cache()
    build_semantic_index(csvs_dir, "snoopy", "default", semantic_cfg)

    cp = configparser.ConfigParser()
    cp["Dataset"] = {"name": "test_sj", "root": str(tmp_path / "data")}
    cp["Database"] = {
        "dbms": "duckdb",
        "db_filename": "blend.duckdb",
        "index_table": "blend_index",
    }
    cp["Semantic"] = {
        "sidecar_filename": "blend_index_basenames.parquet",
        "faiss_quant": "flat",
        "faiss_k_coarse": "10",
        "device": "cpu",
    }
    cp["Semantic.SJ"] = {
        "approach": "snoopy",
        "index_name": "default",
    }
    ini = tmp_path / "config.ini"
    with open(ini, "w") as f:
        cp.write(f)
    monkeypatch.setenv("BLEND_CONFIG", str(ini))
    drop_blend_modules()
    try:
        yield {"approach_dir": approach_dir, "csvs": csvs_dir}
    finally:
        reset_semantic_cache()
        drop_blend_modules()


def _query_df(table_id: str) -> pd.DataFrame:
    df = pd.read_csv(FIXTURE / "csvs" / table_id, dtype=str, keep_default_na=False)
    df.attrs["table_id"] = table_id
    return df


def test_sj_returns_blend_int_ids(sj_env):
    Plan = importlib.import_module("src.Plan").Plan
    Seekers = importlib.import_module("src.Operators.Seekers")
    df = _query_df("t0000003.csv")
    plan = Plan()
    plan.add("sj", Seekers.SJ(df, k=2))
    ids = plan.run()
    assert isinstance(ids, list)
    assert 0 < len(ids) <= 2
    assert all(isinstance(t, int) for t in ids)
    assert set(ids).issubset(set(range(5)))


def test_sj_unknown_table_id_encodes_when_opted_in(sj_env, capsys):
    Plan = importlib.import_module("src.Plan").Plan
    Seekers = importlib.import_module("src.Operators.Seekers")
    df = pd.DataFrame({"c0": ["x", "y"]})
    df.attrs["table_id"] = "missing.csv"
    plan = Plan()
    plan.add("sj", Seekers.SJ(df, k=2, config_overrides={"query_encode": "auto"}))
    ids = plan.run()
    assert 0 < len(ids) <= 2 and set(ids) <= set(range(5))
    assert "[SEMANTIC][QUERY-ENCODE]" in capsys.readouterr().err


def test_sj_unknown_table_id_hard_errors_by_default(sj_env):
    Plan = importlib.import_module("src.Plan").Plan
    Seekers = importlib.import_module("src.Operators.Seekers")
    df = pd.DataFrame({"c0": ["x", "y"]})
    df.attrs["table_id"] = "missing.csv"
    plan = Plan()
    plan.add("warm", Seekers.SJ(_query_df("t0000003.csv"), k=1,
                                config_overrides={"query_encode": "auto"}))   # handle opened under auto
    plan.run()
    plan = Plan()
    plan.add("sj", Seekers.SJ(df, k=1))                                        # default: off
    with pytest.raises(KeyError, match="not present in the snoopy/default registry"):
        plan.run()


def test_sj_missing_query_col_name_with_multi_col_df(sj_env):
    Seekers = importlib.import_module("src.Operators.Seekers")
    df = pd.DataFrame({"a": ["1"], "b": ["2"]})
    df.attrs["table_id"] = "t0000000.csv"
    with pytest.raises(ValueError, match="query_col_name"):
        Seekers.SJ(df, k=1)


def test_sj_query_col_name_not_in_df(sj_env):
    Seekers = importlib.import_module("src.Operators.Seekers")
    df = pd.DataFrame({"c0": ["1"]})
    df.attrs["table_id"] = "t0000000.csv"
    with pytest.raises(KeyError, match="ghost"):
        Seekers.SJ(df, k=1, query_col_name="ghost")


def test_sj_no_table_id_hard_errors_by_default(sj_env):
    Seekers = importlib.import_module("src.Operators.Seekers")
    df = pd.DataFrame({"c0": ["1"]})
    with pytest.raises(ValueError, match="requires a query_table_id"):
        Seekers.SJ(df, k=1)


def test_sj_no_table_id_encodes_under_auto(sj_env):
    Plan = importlib.import_module("src.Plan").Plan
    Seekers = importlib.import_module("src.Operators.Seekers")
    df = pd.DataFrame({"c0": ["x", "y"]})
    plan = Plan()
    plan.add("sj", Seekers.SJ(df, k=2, config_overrides={"query_encode": "auto"}))
    ids = plan.run()
    assert 0 < len(ids) <= 2


def test_sj_pushdown_empty_short_circuits(sj_env):
    importlib.import_module("src.Plan")
    Seekers = importlib.import_module("src.Operators.Seekers")
    DBHandler = importlib.import_module("src.DBHandler").DBHandler
    df = _query_df("t0000003.csv")
    sj = Seekers.SJ(df, k=2)
    db = DBHandler()
    sql = sj.create_sql_query(db, additionals=" AND TableId IN ()")
    assert "WHERE 1=0" in sql


def test_sj_union_of_two_sj_seekers(sj_env):
    Plan = importlib.import_module("src.Plan").Plan
    Seekers = importlib.import_module("src.Operators.Seekers")
    Combiners = importlib.import_module("src.Operators.Combiners")
    df_a = _query_df("t0000000.csv")
    df_b = _query_df("t0000003.csv")
    plan = Plan()
    plan.add("a", Seekers.SJ(df_a, k=3))
    plan.add("b", Seekers.SJ(df_b, k=3))
    plan.add("u", Combiners.Union(), ["a", "b"])
    ids = plan.run()
    assert isinstance(ids, list)
    assert all(isinstance(t, int) for t in ids)


def test_intersection_sj_mid_chain_propagates_upstream_filter(sj_env):
    importlib.import_module("src.Plan")
    Seeker = importlib.import_module("src.Operators.Seekers.SeekerBase").Seeker
    Intersection = importlib.import_module(
        "src.Operators.Combiners.Intersection"
    ).Intersection
    SJBase = importlib.import_module(
        "src.Semantic.seekers.join_base"
    ).SemanticJoinSeekerBase
    _parse_pushdown_ids = importlib.import_module(
        "src.Semantic.seekers.union_base"
    )._parse_pushdown_ids

    class _StubSeeker(Seeker):
        HAS_ML_COST_MODEL = False

        def __init__(self, *, k, fixed_cost, fixed_result=None):
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

    class _TestableSJ(SJBase):
        def run(self, additionals: str = ""):
            self.create_sql_query(self.DB, additionals=additionals)
            _, ids = self._cached
            return list(ids[: self.k])

    sj = _TestableSJ(_query_df("t0000003.csv"), k=3)
    cheap = _StubSeeker(k=3, fixed_cost=1, fixed_result=[1, 2])
    expensive = _StubSeeker(k=3, fixed_cost=100)

    inter = Intersection(k=3)
    inter.set_inputs([sj, cheap, expensive])
    inter.create_sql_query(inter.DB, additionals="")

    assert expensive.received_additionals is not None, (
        "Intersection never reached the expensive (last-position) seeker"
    )
    ids = _parse_pushdown_ids(expensive.received_additionals)
    assert ids is not None, (
        f"expensive expected an IN clause, got {expensive.received_additionals!r}"
    )
    assert ids.issubset({1, 2}), (
        f"upstream K={{1,2}} lost through SJ mid-chain; expensive received "
        f"{expensive.received_additionals!r}"
    )


def test_intersection_sj_mid_chain_drops_unknown_upstream_ids(sj_env):
    importlib.import_module("src.Plan")
    Seeker = importlib.import_module("src.Operators.Seekers.SeekerBase").Seeker
    Intersection = importlib.import_module(
        "src.Operators.Combiners.Intersection"
    ).Intersection
    SJBase = importlib.import_module(
        "src.Semantic.seekers.join_base"
    ).SemanticJoinSeekerBase

    class _StubSeeker(Seeker):
        HAS_ML_COST_MODEL = False

        def __init__(self, *, k, fixed_cost, fixed_result=None):
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

    class _TestableSJ(SJBase):
        def run(self, additionals: str = ""):
            self.create_sql_query(self.DB, additionals=additionals)
            _, ids = self._cached
            return list(ids[: self.k])

    sj = _TestableSJ(_query_df("t0000003.csv"), k=3)
    # cost=0 so `cheap` runs before SJ (cost=1), keeping SJ mid-chain
    cheap = _StubSeeker(k=3, fixed_cost=0, fixed_result=[999, 1000])
    expensive = _StubSeeker(k=3, fixed_cost=100)

    inter = Intersection(k=3)
    inter.set_inputs([sj, cheap, expensive])
    sql = inter.create_sql_query(inter.DB, additionals="")

    assert "WHERE 1=0" in sql, (
        f"empty-SJ mid-chain should short-circuit Intersection; got {sql!r}"
    )
    assert expensive.received_additionals is None, (
        "expensive seeker should not be invoked once SJ returns empty"
    )


def test_sj_minuend_excludes_not_in(sj_env):
    importlib.import_module("src.Plan")
    SJBase = importlib.import_module("src.Semantic.seekers.join_base").SemanticJoinSeekerBase

    class _FakeDB:
        dbms = "duckdb"

    def sj_ids(k, additionals):
        sj = SJBase(_query_df("t0000003.csv"), k=k)
        sj.create_sql_query(_FakeDB(), additionals=additionals)
        return sj._cached[1]

    full = sj_ids(4, "")
    assert full, "SJ returned no candidates"
    drop = full[0]
    got = sj_ids(4, f" AND TableId NOT IN ({drop}) ")
    assert drop not in got
    if len(full) >= 2:
        refill = sj_ids(1, f" AND TableId NOT IN ({full[0]}) ")
        assert refill == [full[1]]
