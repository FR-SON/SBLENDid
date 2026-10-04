"""Negative-Example Search benchmark — RUNTIME ONLY (no accuracy by design)."""

from __future__ import annotations

import csv
import random
from statistics import mean
from time import perf_counter

import pandas as pd
from tqdm import tqdm

from src.Benchmark.datasource import load_query_table, load_sidecar
from src.Benchmark.db import bind_plan, open_dataset_db
from src.Benchmark.prepare import bench_task_dir
from src.Tasks.NegativeExampleSearch import NegativeExampleSearch


def _load_queries(dataset: str) -> list[dict]:
    base = bench_task_dir(dataset, "negex")
    with (base / "query.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def _sample_negatives(dataset: str, lake: set[str], query_table: str,
                      c1: str, c2: str, n_neg: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    qdf = load_query_table(dataset, query_table)
    forbidden = set(qdf[c1].astype(str)) if c1 in qdf.columns else set()
    pool = []
    for name in rng.sample(sorted(lake - {query_table}), min(50, len(lake) - 1)):
        cdf = load_query_table(dataset, name)
        if cdf.shape[1] < 2:
            continue
        for a, b in cdf.iloc[:, :2].astype(str).dropna().values:
            if a not in forbidden:
                pool.append((a, b))
        if len(pool) >= n_neg:
            break
    rng.shuffle(pool)
    return pd.DataFrame(pool[:n_neg], columns=[c1, c2])


def run_negex(dataset: str, k: int) -> tuple[list[dict], dict]:
    _, lake = load_sidecar(dataset)
    queries = _load_queries(dataset)
    db = open_dataset_db(dataset)
    rows: list[dict] = []
    t0 = perf_counter()
    try:
        for q in tqdm(queries, desc=f"negex/{dataset}"):
            qdf = load_query_table(dataset, q["query_table"])
            c1, c2 = q["col1"], q["col2"]
            if c1 not in qdf.columns or c2 not in qdf.columns:
                continue
            inclusive = qdf[[c1, c2]].dropna().drop_duplicates().reset_index(drop=True)
            negatives = _sample_negatives(dataset, lake, q["query_table"], c1, c2,
                                          int(q["n_neg"]), int(q["seed"]))
            exclusive = pd.concat([inclusive, negatives], ignore_index=True)
            plan = NegativeExampleSearch(inclusive, c1, c2, exclusive, c1, c2, k=k)
            bind_plan(plan, db)
            t = perf_counter()
            ids = plan.run()
            runtime_ms = (perf_counter() - t) * 1000.0
            rows.append({"query": q["query_table"], "n_inclusive": len(inclusive),
                         "n_negatives": len(negatives), "n_retrieved": len(ids),
                         "runtime_ms": runtime_ms})
    finally:
        db.close()
    summary = {
        "evaluated": len(rows), "total_queries": len(queries), "effective_n": len(rows),
        "skipped": len(queries) - len(rows),
        "mean_runtime_ms": mean(r["runtime_ms"] for r in rows) if rows else 0.0,
        "wall_s": perf_counter() - t0,
    }
    return rows, summary
