from pathlib import Path
import pandas as pd
import pytest
from src.Benchmark.impute_compare import build_ctx, resolve_query, TOKEN_LEGS

_ROOT = Path(__file__).resolve().parents[1]


def test_token_legs_registered():
    assert set(TOKEN_LEGS) == {"syntactic_join", "syntactic_paper", "mc_only"}


def test_mc_only_is_a_single_mc_seeker_at_k():
    import src.Benchmark.impute_compare as ic
    from src.Operators.Seekers import MultiColumnOverlap
    ctx = type("C", (), {"examples": pd.DataFrame({"a": ["x"], "b": ["y"]})})()
    plan = ic.TOKEN_LEGS["mc_only"](ctx, 10)
    ops = list(plan._operators.values())
    assert len(ops) == 1 and isinstance(ops[0], MultiColumnOverlap)
    assert ops[0].k == 10


def test_resolve_query_scores_against_true_value(tmp_path, monkeypatch):
    dsdir = tmp_path / "datasets" / "toy"
    csvs = dsdir / "csvs"; csvs.mkdir(parents=True)
    pd.DataFrame({"city": ["paris", "rome", "nice", "lyon", "metz", "bonn", "kiel"],
                  "country": ["france"] * 7}).to_csv(csvs / "src.csv", index=False)
    monkeypatch.setenv("BLEND_DATASETS_DIR", str(tmp_path / "datasets"))
    q = {"query_id": "0", "source_table": "src.csv", "key_col": "city", "val_col": "country"}
    ctx = build_ctx("toy", q)
    assert ctx is not None

    other = pd.DataFrame({0: ["paris", "rome", "bonn", "kiel"],
                          1: ["france", "france", "france", "wrong"]})
    decoy = pd.DataFrame({0: ["1"], 1: ["2"]})

    class _Store:
        _tables = {2: other, 3: decoy}
        def get(self, tid): return self._tables.get(int(tid))
        def col_sets(self, tid):
            df = self._tables.get(int(tid))
            return None if df is None else {c: set(df[c]) for c in df.columns}

    counts = resolve_query(ctx, [1, 2, 3], _Store(), source_tableid=1)
    assert counts["n_keys"] == 2 and counts["n_covered"] == 2 and counts["n_correct"] == 1


from pathlib import Path
from src.Benchmark.impute_compare import run_compare

_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    not (_ROOT / "datasets/santos/bench/imputation/query.csv").exists(),
    reason="run `prepare --task imputation --dataset santos` first")
def test_run_compare_token_legs_santos():
    out = run_compare("santos", k_grid=[10, 50], legs=TOKEN_LEGS)
    assert out["n_queries"] > 0
    for leg in TOKEN_LEGS:
        for k in ("10", "50"):
            cell = out["results"][leg][k]
            assert 0.0 <= cell["coverage"] <= 1.0
            assert 0.0 <= cell["overall_accuracy"] <= 1.0
            assert cell["retrieval_runtime_ms"] >= 0.0


@pytest.mark.skipif(
    not (_ROOT / "datasets/santos/semantic/snoopy/default/index/registry.parquet").exists(),
    reason="santos snoopy registry not present")
def test_semantic_join_leg_returns_ids():
    import src.Benchmark.impute_compare as ic
    from src.Benchmark.db import bind_plan, open_dataset_db
    q = ic.load_queries("santos")[0]
    ctx = ic.build_ctx("santos", q)
    ic._run_dataset = "santos"
    plan = ic.SEMANTIC_LEGS["semantic_join"](ctx, 10)
    db = open_dataset_db("santos")
    try:
        bind_plan(plan, db)
        ids = plan.run()
    finally:
        db.close()
    assert isinstance(ids, list)


def test_add_comparison_computes_deltas():
    from src.Benchmark.impute_compare import add_comparison
    out = {"k_grid": [10], "results": {
        "syntactic_join": {"10": {"overall_accuracy": 0.2, "coverage": 0.3, "accuracy_at_covered": 0.6}},
        "semantic_join":  {"10": {"overall_accuracy": 0.5, "coverage": 0.7, "accuracy_at_covered": 0.7}},
        "syntactic_paper":{"10": {"overall_accuracy": 0.1, "coverage": 0.1, "accuracy_at_covered": 1.0}},
        "semantic_paper": {"10": {"overall_accuracy": 0.15, "coverage": 0.15, "accuracy_at_covered": 1.0}},
    }}
    out = add_comparison(out)
    d = out["results"]["comparison"]["10"]
    assert abs(d["semantic_join_minus_syntactic_join"]["overall_accuracy"] - 0.3) < 1e-9
    assert abs(d["semantic_join_minus_semantic_paper"]["coverage"] - 0.55) < 1e-9
