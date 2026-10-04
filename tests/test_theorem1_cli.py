import pytest

from src.Benchmark.cli import build_parser, main
from src.Benchmark.correctness.theorem1.run import FIELDS, field_values
from src.Benchmark.runspec import render_slug


def _parse(extra=""):
    return build_parser().parse_args(
        f"correctness-theorem1 --dataset santos {extra}".split())


def test_defaults_match_design():
    a = _parse()
    assert a.shapes == ["sc_sc_flat", "sc_sc_prov", "sc_kw"]
    assert (a.k, a.repeats, a.rows) == (10, 1, 5000)
    assert a.arms == ["a3"]
    assert a.bno_construction == "slice" and a.bno_order == "lega"
    assert a.k_semantics == "faithful" and a.sem_position == "first"
    assert a.width_grid[-1] is None and a.width_grid[0] == 10
    assert a.k_coarse is None and a.ef is None and a.exact_threshold is None
    assert not a.force_exact and not a.verify_soundness
    assert a.query_timeout is None


def test_query_timeout_draws_replacement(tmp_path, monkeypatch):
    import json

    import pandas as pd

    from src.Benchmark import datasource
    from src.Benchmark.correctness.theorem1 import run as T1
    from src.Optimizer.training import QueryTimeout

    monkeypatch.setattr(datasource, "load_query_table",
                        lambda dataset, q, nrows=None:
                        pd.DataFrame({"a": ["x"], "b": ["y"]}))
    monkeypatch.setattr(datasource, "load_sidecar", lambda dataset: ({}, set()))
    monkeypatch.setattr(T1, "select_columns", lambda df: ("a", "b"))

    def fake_eval(shape, df, qt, **kw):
        if qt == "q_slow":
            raise QueryTimeout("cap")
        return dict(query=qt, shape=shape.name, k=10, repeats=1,
                    opt_runs=[[1]], bno_runs=[[1]],
                    stable_opt=True, stable_bno=True, both_stable=True,
                    witness=False, witness_bno_empty=False,
                    witness_added=None, witness_dropped=None,
                    n_full_a=None, n_full_b=None, n_I=None,
                    shape_source="s", operators_upstream_faithful=True,
                    optimizer_reachable=True, arm3_only=False,
                    bno_construction_used=["slice", "slice"],
                    prov={"leg_costs": {"a": 4, "b": 4}})

    monkeypatch.setattr(T1, "_eval_one", fake_eval)

    class _Run:
        def __init__(self, p):
            self.p = p

        def file(self, name):
            return self.p / name

        def write_json(self, name, obj):
            (self.p / name).write_text(json.dumps(obj))
            return self.p / name

        def write_table(self, rows, summary):
            (self.p / "results.json").write_text(json.dumps(summary))
            return self.p / "results.json"

    summary = T1.run_theorem1(
        "lakeX", shape_names=["sc_sc_flat"], k=10, repeats=1, rows_cap=5000,
        arm_set={"a3"}, width_grid=[10], bno_construction="slice",
        bno_order="lega", k_semantics="faithful", sem_position="first",
        sem_knobs=None, force_exact=False, verify_soundness=False,
        resume_rows=[], queries=["q_slow", "q1", "q2", "q3"], db=object(),
        run=_Run(tmp_path), limit=2, query_timeout=60)
    c = summary["cohort"]
    assert c["completed_queries"] == 2 and c["target_queries"] == 2
    assert c["skipped"]["timeout"] == 1
    rows = [json.loads(l) for l in
            (tmp_path / "rows.jsonl").read_text().splitlines()]
    assert {r["query"] for r in rows} == {"q_slow", "q1", "q2"}
    assert any("QueryTimeout" in r.get("error", "") for r in rows)


def test_mixed_mc_shapes_refused(capsys):
    with pytest.raises(SystemExit):
        main(["correctness-theorem1", "--dataset", "x",
              "--shapes", "sc_sc_flat,mc_sc_repo"])
    assert "--rows 50" in capsys.readouterr().err


def test_verify_soundness_needs_width_arm():
    with pytest.raises(SystemExit):
        main(["correctness-theorem1", "--dataset", "x", "--verify-soundness"])


def test_unknown_shape_refused():
    with pytest.raises(SystemExit):
        main(["correctness-theorem1", "--dataset", "x", "--shapes", "nope"])


def test_terminal_order_mc_refused():
    with pytest.raises(SystemExit):
        main(["correctness-theorem1", "--dataset", "x",
              "--shapes", "mc_sc_repo", "--rows", "50",
              "--bno-order", "terminal"])


def test_plain_run_slugs_short_and_knobs_render():
    a = _parse()
    assert render_slug(FIELDS, field_values(a, "duckdb_faiss")) == "duckdb_faiss"
    b = _parse("--k 25 --repeats 5 --k-coarse 500")
    slug = render_slug(FIELDS, field_values(b, "duckdb_faiss"))
    assert "k25" in slug and "rep5" in slug and "kc500" in slug
    assert "ef500" in slug
