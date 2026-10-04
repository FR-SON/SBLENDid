import pandas as pd

from src.Benchmark.regime import (
    record_pair_hits, pair_key, attach_labels, add_hit_cols, add_bucket,
    STRATA_LABELS,
)


def test_record_pair_hits_join():
    sink = []
    retrieved = [("t2", "c"), ("tX", "c"), ("t1", "c")]
    relevant = [("t1", "c"), ("t2", "c"), ("t9", "c")]
    record_pair_hits(sink, task="join", query_table="q", query_column="qc",
                     retrieved=retrieved, relevant=relevant, dataset="ds", variant="sj")
    by = {r["candidate_table"]: r for r in sink}
    assert by["t1"]["rank"] == 2 and by["t2"]["rank"] == 0 and by["t9"]["rank"] == -1
    assert by["t1"]["candidate_column"] == "c" and by["t1"]["task"] == "join"
    assert by["t1"]["dataset"] == "ds" and by["t1"]["variant"] == "sj"


def test_record_pair_hits_union_bare_table_keys():
    sink = []
    record_pair_hits(sink, task="union", query_table="q", query_column="",
                     retrieved=["t3", "t1"], relevant=["t1", "t2"], dataset="ds")
    by = {r["candidate_table"]: r for r in sink}
    assert by["t1"]["rank"] == 1 and by["t2"]["rank"] == -1
    assert by["t1"]["candidate_column"] == "" and by["t1"]["task"] == "union"


def test_pair_key_join_vs_union():
    df = pd.DataFrame([{"query_table": "q.csv", "candidate_table": "c.csv",
                        "query_column": "﻿Id", "candidate_column": "'ID'"}])
    assert pair_key(df, "union").iloc[0] == "q.csv|c.csv"
    assert pair_key(df, "join").iloc[0] == "q.csv|c.csv|id|id"


def test_attach_labels_join_and_unlabeled():
    pairs = pd.DataFrame([
        {"query_table": "q", "candidate_table": "a", "query_column": "k", "candidate_column": "k", "rank": 0},
        {"query_table": "q", "candidate_table": "z", "query_column": "k", "candidate_column": "k", "rank": 3},
    ])
    reg = pd.DataFrame([
        {"query_table": "q", "candidate_table": "a", "query_column": "k", "candidate_column": "k",
         "n_inter": 0, "cont_q2x": 0.0, "cont_x2q": 0.0, "numeric_q": "false", "status": "ok"},
        {"query_table": "q", "candidate_table": "z", "query_column": "k", "candidate_column": "k",
         "n_inter": 5, "cont_q2x": 0.8, "cont_x2q": 0.9, "numeric_q": "false", "status": "missing_table"},
    ])
    m, n_unlabeled = attach_labels(pairs, reg, "join")
    assert n_unlabeled == 1
    ok = m[m["n_inter"].notna()].iloc[0]
    assert ok["candidate_table"] == "a" and ok["n_inter"] == 0


def test_add_hit_cols():
    df = pd.DataFrame({"rank": [0, 4, -1, 24]})
    add_hit_cols(df, ks=(1, 5, 25))
    assert list(df["hit@1"]) == [True, False, False, False]
    assert list(df["hit@5"]) == [True, True, False, False]
    assert list(df["hit@25"]) == [True, True, False, True]


def test_add_bucket_binary_and_strata():
    df = pd.DataFrame({"n_inter": [0, 3], "cont_q2x": [0.0, 0.5]})
    add_bucket(df, strata=False)
    assert list(df["bucket"]) == ["semantic", "overlap"]
    add_bucket(df, strata=True)
    assert df["bucket"].tolist() == ["0", "(0.3,0.5]"]
    assert set(STRATA_LABELS) >= set(df["bucket"].dropna().astype(str))


def test_add_bucket_union_uses_symmetric_containment():
    df = pd.DataFrame({"n_inter": [5], "cont_q2x": [0.001], "cont_x2q": [1.0]})
    add_bucket(df, strata=True, task="join")
    assert df["bucket"].tolist() == ["(0,0.1]"]
    add_bucket(df, strata=True, task="union")
    assert df["bucket"].tolist() == ["(0.5,1]"]


def test_attach_labels_join_table_level_best_column_pair():
    import pandas as pd
    from src.Benchmark.regime import attach_labels_join_table_level

    pairs = pd.DataFrame([{
        "query_table": "q.csv", "query_column": "name",
        "candidate_table": "c.csv", "candidate_column": "", "rank": 0,
    }])
    registry = pd.DataFrame([
        {"query_table": "q.csv", "query_column": "name", "candidate_table": "c.csv",
         "candidate_column": "weak", "n_inter": 1, "cont_q2x": 0.1, "cont_x2q": 0.1,
         "numeric_q": False, "status": "ok"},
        {"query_table": "q.csv", "query_column": "name", "candidate_table": "c.csv",
         "candidate_column": "strong", "n_inter": 5, "cont_q2x": 0.8, "cont_x2q": 0.5,
         "numeric_q": False, "status": "ok"},
    ])
    merged, n_unlabeled = attach_labels_join_table_level(pairs, registry)
    assert n_unlabeled == 0
    assert merged.loc[0, "cont_q2x"] == 0.8
