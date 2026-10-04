import os
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def unit_cfg(duckdb_unit_env):
    cfg = Path(os.environ["BLEND_CONFIG"])
    cfg.write_text(
        cfg.read_text()
        + "\n[Semantic.SU]\napproach = liftus\nindex_name = default\n"
        + "\n[Semantic.SJ]\napproach = snoopy\nindex_name = default\n"
    )
    yield cfg


def test_join_build_plan_scopes_semantic_config_to_dataset(unit_cfg):
    from src.Benchmark.adapters.join import _build_plan

    df = pd.DataFrame({"col": ["x"]})
    for seeker in ("sj", "sho"):
        plan = _build_plan(seeker, ["x"], df, "col", "q.csv", 5, dataset="ds-x")
        op = next(iter(plan._operators.values()))
        assert op._cfg.dataset.name == "ds-x"


def test_union_build_plan_scopes_semantic_config_to_dataset(unit_cfg):
    from src.Benchmark.adapters.union import _build_union_plan

    df = pd.DataFrame({"a": ["x"], "b": ["y"]})
    for seeker in ("su", "sho"):
        plan = _build_union_plan(seeker, df, 5, qtable="q.csv", dataset="ds-x")
        sem = [o for n, o in plan._operators.items() if n != "counter"]
        assert sem and all(o._cfg.dataset.name == "ds-x" for o in sem)


def _fake_join_deps(monkeypatch, J, plan):
    monkeypatch.setattr(J, "load_sidecar", lambda d: ({1: "t1.csv"}, {"t1.csv", "q.csv"}))
    monkeypatch.setattr(J, "load_join_queries", lambda d: [("q.csv", "col")])
    monkeypatch.setattr(J, "load_join_gt", lambda d: {("q.csv", "col"): {"t1.csv"}})
    monkeypatch.setattr(J, "load_query_table", lambda d, b: pd.DataFrame({"col": ["x"]}))
    monkeypatch.setattr(J, "open_dataset_db", lambda d: type("DB", (), {"close": lambda s: None})())
    monkeypatch.setattr(J, "bind_plan", lambda p, db: None)
    monkeypatch.setattr(J, "_build_plan", lambda *a, **kw: plan)


def test_join_adapter_skips_query_column_not_in_registry(monkeypatch, duckdb_unit_env):
    import src.Benchmark.adapters.join as J
    from src.Semantic.retrieve import QueryColumnNotIndexed

    class Plan:
        def run(self):
            raise QueryColumnNotIndexed("q.csv/col not enrolled")
    _fake_join_deps(monkeypatch, J, Plan())

    rows, summary = J.run_join("ds", 10, seeker="sj")
    assert rows == [] and summary["skipped"] == 1


def test_join_adapter_propagates_other_key_errors(monkeypatch, duckdb_unit_env):
    import src.Benchmark.adapters.join as J

    class Plan:
        def run(self):
            raise KeyError("basenames sidecar is missing 5 table_ids")
    _fake_join_deps(monkeypatch, J, Plan())

    with pytest.raises(KeyError):
        J.run_join("ds", 10, seeker="sj")


def test_to_basenames_names_the_dataset_on_miss():
    from src.Benchmark.datasource import to_basenames

    assert to_basenames({1: "t1.csv"}, [1], "ds-x") == ["t1.csv"]
    with pytest.raises(KeyError, match="ds-x"):
        to_basenames({1: "t1.csv"}, [7], "ds-x")
