"""Seeker-neutral regime-split recall."""
from __future__ import annotations

import unicodedata

import pandas as pd

RECALL_KS = (1, 5, 10, 25, 50)
STRATA_EDGES = [-0.01, 1e-9, 0.1, 0.3, 0.5, 1.01]
STRATA_LABELS = ["0", "(0,0.1]", "(0.1,0.3]", "(0.3,0.5]", "(0.5,1]"]


def norm_name(s):
    s = str(s).replace("﻿", "").replace("ï»¿", "")
    return unicodedata.normalize("NFKC", s).strip().strip('"').strip("'").strip().casefold()


def record_pair_hits(sink, *, task, query_table, query_column, retrieved, relevant, **stamp):
    """Append one row per relevant key with its rank in `retrieved` (-1 if absent)."""
    rank_of = {key: i for i, key in enumerate(retrieved)}
    for key in relevant:
        r = rank_of.get(key, -1)
        ct, cc = key if task == "join" else (key, "")
        sink.append({
            **stamp, "task": task,
            "query_table": query_table, "query_column": query_column,
            "candidate_table": ct, "candidate_column": cc, "rank": int(r),
        })


def pair_key(df, task):
    """Series join key for pairs <-> registry (4-key for join, 2-key for union)."""
    k = df["query_table"].astype(str) + "|" + df["candidate_table"].astype(str)
    if task == "join":
        k = k + "|" + df["query_column"].map(norm_name) + "|" + df["candidate_column"].map(norm_name)
    return k


def attach_labels(pairs, registry, task):
    """Left-join `pairs` to the ok rows of `registry`; returns (merged, n_unlabeled)."""
    reg = registry[registry["status"] == "ok"].copy()
    reg["_k"] = pair_key(reg, task)
    reg = reg.drop_duplicates("_k", keep="first")
    p = pairs.copy()
    p["_k"] = pair_key(p, task)
    merged = p.merge(reg[["_k", "n_inter", "cont_q2x", "cont_x2q", "numeric_q"]], on="_k", how="left")
    n_unlabeled = int(merged["n_inter"].isna().sum())
    return merged, n_unlabeled


def add_hit_cols(df, ks=RECALL_KS):
    for k in ks:
        df[f"hit@{k}"] = df["rank"].between(0, k - 1)
    return df


def add_bucket(df, strata=True, task="join"):
    if strata:
        cont = df["cont_q2x"] if task == "join" else df[["cont_q2x", "cont_x2q"]].max(axis=1)
        df["bucket"] = pd.cut(cont, STRATA_EDGES, labels=STRATA_LABELS)
    else:
        df["bucket"] = df["n_inter"].eq(0).map({True: "semantic", False: "overlap"})
    return df


def _method_col(df):
    for c in ("method", "variant"):
        if c in df.columns:
            return c
    df["_seeker"] = "seeker"
    return "_seeker"


def run_regime_recall(pairs, registry, *, task, strata, nonnumeric, pool_size, ks):
    """Pair-hits + overlap registry -> tidy recall@k rows per variant x bucket x pool_size."""
    table_level = task == "join" and pairs["candidate_column"].astype(str).eq("").all()
    if table_level:
        m, _ = attach_labels_join_table_level(pairs, registry)
    else:
        m, _ = attach_labels(pairs, registry, task)
    m = m[m["n_inter"].notna()].copy()

    if pool_size is not None and "pool_size" in m.columns:
        m = m[m["pool_size"] == pool_size]
    if nonnumeric:
        m = m[m["numeric_q"].astype(str).str.lower() == "false"]

    add_hit_cols(m, ks)
    add_bucket(m, strata=strata, task=task)
    method_col = _method_col(m)
    hitcols = [f"hit@{k}" for k in ks]

    groups = m.groupby("pool_size") if "pool_size" in m.columns else [("(all)", m)]
    tables = []
    for pool, g in groups:
        agg = g.groupby([method_col, "bucket"], observed=True)
        out = agg[hitcols].mean().round(3).join(agg.size().rename("n"))
        tables.append(out.reset_index().assign(pool_size=pool))
    combined = pd.concat(tables, ignore_index=True)
    if method_col != "variant":
        combined = combined.rename(columns={method_col: "variant"})
    return combined


def condensed_recall(pairs, registry, *, task, ks):
    """Condensed recall@k over the *semantic* (`n_inter==0`) GT pairs, per variant."""
    import bisect

    table_level = task == "join" and pairs["candidate_column"].astype(str).eq("").all()
    m, _ = (attach_labels_join_table_level(pairs, registry) if table_level
            else attach_labels(pairs, registry, task))
    m = m[m["n_inter"].notna()].copy()
    m["n_inter"] = m["n_inter"].astype(float)
    m["semantic"] = m["n_inter"] == 0.0
    m["rank"] = m["rank"].astype(int)
    method_col = _method_col(m)
    keycols = ["query_table"] + (["query_column"] if task == "join" else [])

    recs = []
    for key, g in m.groupby([method_col] + keycols, observed=True):
        meth = key[0]
        ov_ranks = sorted(r for r in g.loc[~g["semantic"], "rank"] if r >= 0)
        for r in g.loc[g["semantic"], "rank"]:
            cond = -1 if r < 0 else r - bisect.bisect_left(ov_ranks, r)
            recs.append({"variant": meth, "raw_rank": int(r), "cond_rank": int(cond)})
    d = pd.DataFrame(recs)
    if d.empty:
        return d

    rows = []
    for meth, g in d.groupby("variant"):
        row = {"variant": meth, "n_semantic": len(g)}
        for k in ks:
            row[f"raw@{k}"] = round(float(g["raw_rank"].between(0, k - 1).mean()), 4)
            row[f"cond@{k}"] = round(float(g["cond_rank"].between(0, k - 1).mean()), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def attach_labels_join_table_level(pairs, registry):
    """Label table-level join pair-hits via each candidate table's best-aligned column pair."""
    reg = registry[registry["status"] == "ok"].copy()
    reg = reg.sort_values(["cont_q2x", "n_inter"], ascending=False)
    reg["_k"] = (reg["query_table"].astype(str) + "|"
                 + reg["candidate_table"].astype(str) + "|"
                 + reg["query_column"].map(norm_name))
    reg = reg.drop_duplicates("_k", keep="first")
    p = pairs.copy()
    p["_k"] = (p["query_table"].astype(str) + "|"
               + p["candidate_table"].astype(str) + "|"
               + p["query_column"].map(norm_name))
    merged = p.merge(reg[["_k", "n_inter", "cont_q2x", "cont_x2q", "numeric_q"]],
                     on="_k", how="left")
    return merged, int(merged["n_inter"].isna().sum())
