import pytest


@pytest.fixture(autouse=True)
def _duckdb_env(duckdb_unit_env):
    yield


def test_run_parser_accepts_seeker_and_pair_hits():
    from src.Benchmark.cli import build_parser
    args = build_parser().parse_args(
        ["run", "--dataset", "d", "--task", "join", "--k", "10",
         "--seeker", "sho", "--pair-hits"])
    assert args.seeker == "sho" and args.pair_hits is True


def test_run_parser_defaults_keep_old_behavior():
    from src.Benchmark.cli import build_parser
    args = build_parser().parse_args(
        ["run", "--dataset", "d", "--task", "union", "--k", "10"])
    assert args.seeker == "sc" and args.pair_hits is False


def test_join_adapter_records_pair_hits(monkeypatch, tmp_path):
    import src.Benchmark.adapters.join as J

    monkeypatch.setattr(J, "load_sidecar", lambda d: ({1: "t1.csv", 2: "t2.csv"},
                                                      {"t1.csv", "t2.csv", "q.csv"}))
    monkeypatch.setattr(J, "load_join_queries", lambda d: [("q.csv", "col")])
    monkeypatch.setattr(J, "load_join_gt", lambda d: {("q.csv", "col"): {"t1.csv", "t2.csv"}})
    monkeypatch.setattr(J, "load_query_table", lambda d, b: __import__("pandas").DataFrame({"col": ["x"]}))
    monkeypatch.setattr(J, "open_dataset_db", lambda d: type("DB", (), {"close": lambda s: None})())
    monkeypatch.setattr(J, "bind_plan", lambda p, db: None)

    class FakePlan:
        def run(self):
            return [1]
    monkeypatch.setattr(J, "_build_plan",
                        lambda seeker, values, df, qcol, qtable, k, dataset: FakePlan())

    sink = []
    rows, summary = J.run_join("ds", 10, seeker="sho", pair_sink=sink)
    assert summary["evaluated"] == 1
    assert {r["candidate_table"] for r in sink} == {"t1.csv", "t2.csv"}
    hit = next(r for r in sink if r["candidate_table"] == "t1.csv")
    miss = next(r for r in sink if r["candidate_table"] == "t2.csv")
    assert hit["rank"] == 0 and miss["rank"] == -1
    assert hit["variant"] == "sho" and hit["task"] == "join"
    assert hit["candidate_column"] == ""


def test_union_adapter_records_pair_hits(monkeypatch, tmp_path):
    import src.Benchmark.adapters.union as U

    monkeypatch.setattr(U, "load_sidecar", lambda d: ({1: "t1.csv", 2: "t2.csv"},
                                                       {"t1.csv", "t2.csv", "q.csv"}))
    monkeypatch.setattr(U, "load_union_queries", lambda d: ["q.csv"])
    monkeypatch.setattr(U, "load_union_gt", lambda d: {"q.csv": {"t1.csv", "t2.csv"}})
    monkeypatch.setattr(U, "load_query_table", lambda d, b: __import__("pandas").DataFrame({"col": ["x"]}))
    monkeypatch.setattr(U, "open_dataset_db", lambda d: type("DB", (), {"close": lambda s: None})())
    monkeypatch.setattr(U, "bind_plan", lambda p, db: None)

    class FakePlan:
        def run(self):
            return [1]
    monkeypatch.setattr(U, "_build_union_plan",
                        lambda seeker, df, k, qtable=None, dataset=None: FakePlan())

    sink = []
    rows, summary = U.run_union("ds", 10, seeker="sho", pair_sink=sink)
    assert summary["evaluated"] == 1
    candidate_tables = {r["candidate_table"] for r in sink}
    assert candidate_tables == {"t1.csv", "t2.csv"}
    for r in sink:
        assert isinstance(r["candidate_table"], str)
    hit = next(r for r in sink if r["candidate_table"] == "t1.csv")
    miss = next(r for r in sink if r["candidate_table"] == "t2.csv")
    assert hit["rank"] == 0 and miss["rank"] == -1
    assert hit["variant"] == "sho" and hit["task"] == "union"
    assert hit["candidate_column"] == ""


def test_union_build_plan_sho_wires_counter(monkeypatch):
    import pandas as pd
    from src.Benchmark.adapters.union import _build_union_plan

    df = pd.DataFrame({"a": ["x"], "b": ["y"]})
    plan = _build_union_plan("sho", df, k=5)
    ops = plan._operators
    assert "counter" in ops and type(ops["counter"]).__name__ == "Counter"
    sho_ops = [o for n, o in ops.items() if n != "counter"]
    assert len(sho_ops) == 2
    assert all(type(o).__name__ == "SimHashOverlap" for o in sho_ops)
    assert all(o.k == 50 for o in sho_ops)
    assert ops["counter"].k == 5
