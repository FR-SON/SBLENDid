from __future__ import annotations

import csv
from collections import defaultdict
from statistics import mean
from time import perf_counter

import pandas as pd
from tqdm import tqdm

from src.Benchmark import metrics as M
from src.Benchmark.datasource import load_query_table, load_sidecar
from src.Benchmark.db import bind_plan, open_dataset_db
from src.Benchmark.prepare import bench_task_dir
from src.Tasks.CorrelationSearch import CorrelationSearch


def _load(dataset: str):
    base = bench_task_dir(dataset, "correlation")
    with (base / "query.csv").open(newline="") as f:
        q = list(csv.DictReader(f))
    gt: dict[str, set[str]] = defaultdict(set)
    with (base / "groundtruth.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            gt[r["query_id"]].add(r["candidate_table"])
    return q, gt


def run_correlation(dataset: str, k: int) -> tuple[list[dict], dict]:
    i2b, lake = load_sidecar(dataset)
    queries, gt = _load(dataset)
    db = open_dataset_db(dataset)
    rows: list[dict] = []
    t0 = perf_counter()
    try:
        for q in tqdm(queries, desc=f"correlation/{dataset}"):
            relevant = gt.get(q["query_id"], set()) & lake
            if not relevant:
                continue
            df = load_query_table(dataset, q["source_table"])
            source = df[q["key_col"]].tolist()
            target = pd.to_numeric(df[q["target_col"]], errors="coerce").tolist()
            plan = CorrelationSearch(source, target, k)
            bind_plan(plan, db)
            t = perf_counter()
            ids = plan.run()
            runtime_ms = (perf_counter() - t) * 1000.0
            retrieved = [i2b[int(i)] for i in ids]
            p = M.precision_at_k(retrieved, relevant)
            r = M.recall_at_k(retrieved, relevant)
            rows.append({
                "query": q["source_table"], "n_gt_in_lake": len(relevant),
                "p@k": p, "recall@k": r, "f1": M.f1(p, r), "runtime_ms": runtime_ms,
            })
    finally:
        db.close()
    summary = {
        "evaluated": len(rows), "total_queries": len(queries), "effective_n": len(rows),
        "skipped": len(queries) - len(rows),
        "mean_p@k": mean(r["p@k"] for r in rows) if rows else 0.0,
        "mean_recall@k": mean(r["recall@k"] for r in rows) if rows else 0.0,
        "mean_f1": mean(r["f1"] for r in rows) if rows else 0.0,
        "wall_s": perf_counter() - t0,
    }
    return rows, summary
