import pandas as pd


def _pairs():
    return pd.DataFrame([
        {"dataset": "d", "task": "join", "variant": "sho", "query_table": "q.csv",
         "query_column": "c", "candidate_table": "a.csv", "candidate_column": "", "rank": 3},
        {"dataset": "d", "task": "join", "variant": "sc", "query_table": "q.csv",
         "query_column": "c", "candidate_table": "a.csv", "candidate_column": "", "rank": -1},
        {"dataset": "d", "task": "join", "variant": "sho", "query_table": "q.csv",
         "query_column": "c", "candidate_table": "b.csv", "candidate_column": "", "rank": 0},
        {"dataset": "d", "task": "join", "variant": "sc", "query_table": "q.csv",
         "query_column": "c", "candidate_table": "b.csv", "candidate_column": "", "rank": 0},
    ])


def _registry():
    return pd.DataFrame([
        {"query_table": "q.csv", "query_column": "c", "candidate_table": "a.csv",
         "candidate_column": "x", "n_inter": 0, "cont_q2x": 0.0, "cont_x2q": 0.0,
         "numeric_q": False, "status": "ok"},
        {"query_table": "q.csv", "query_column": "c", "candidate_table": "b.csv",
         "candidate_column": "x", "n_inter": 9, "cont_q2x": 0.9, "cont_x2q": 0.7,
         "numeric_q": False, "status": "ok"},
    ])


def test_run_regime_recall_buckets_by_variant():
    from src.Benchmark.regime import run_regime_recall
    out = run_regime_recall(_pairs(), _registry(), task="join", strata=False,
                            nonnumeric=False, pool_size=None, ks=[1, 10])
    row = out[(out["variant"] == "sho") & (out["bucket"] == "semantic")].iloc[0]
    assert row["hit@10"] == 1.0 and row["hit@1"] == 0.0 and row["n"] == 1
    row = out[(out["variant"] == "sc") & (out["bucket"] == "semantic")].iloc[0]
    assert row["hit@10"] == 0.0


def test_cli_regime_recall_writes_csv(tmp_path, capsys):
    import json

    from src.Benchmark.cli import main
    p1 = tmp_path / "pairs.parquet"
    _pairs().to_parquet(p1)
    reg = tmp_path / "reg.csv"
    _registry().to_csv(reg, index=False)
    out = tmp_path / "run"
    rc = main(["regime-recall", "--pairs", str(p1), "--registry", str(reg),
               "--task", "join", "--k", "1,10", "--out", str(out)])
    assert rc == 0
    written = pd.read_csv(out / "regime_recall.csv")
    assert {"variant", "bucket", "n", "hit@1", "hit@10"} <= set(written.columns)
    assert "semantic" in capsys.readouterr().out
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["cmd"] == "regime-recall" and manifest["status"] == "ok"
    assert manifest["args"]["task"] == "join" and manifest["args"]["k"] == [1, 10]
