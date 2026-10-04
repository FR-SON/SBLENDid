from __future__ import annotations

import csv
from statistics import mean
from time import perf_counter

from tqdm import tqdm

from src.Benchmark import metrics as M
from src.Benchmark.datasource import dataset_dir, load_sidecar
from src.Benchmark.db import bind_plan, open_dataset_db
from src.Benchmark.prepare import bench_task_dir
from src.Benchmark.recipes.imputation import _reconstruct
from src.Tasks.DataImputation import DataImputation


def _load_manifest(dataset: str):
    base = bench_task_dir(dataset, "imputation")
    with (base / "query.csv").open(newline="") as f:
        q = list(csv.DictReader(f))
    with (base / "groundtruth.csv").open(newline="") as f:
        gt = {r["query_id"]: r["candidate_table"] for r in csv.DictReader(f)}
    return q, gt


def run_imputation(dataset: str, k: int) -> tuple[list[dict], dict]:
    i2b, lake = load_sidecar(dataset)
    csvs = dataset_dir(dataset) / "csvs"
    queries, gt = _load_manifest(dataset)
    db = open_dataset_db(dataset)
    rows: list[dict] = []
    t0 = perf_counter()
    try:
        for q in tqdm(queries, desc=f"imputation/{dataset}"):
            source = gt[q["query_id"]]
            examples, qvals = _reconstruct(csvs / q["source_table"], q["key_col"], q["val_col"])
            if len(qvals) == 0:
                continue
            plan = DataImputation(examples, qvals, k=k)
            bind_plan(plan, db)
            t = perf_counter()
            ids = plan.run()
            runtime_ms = (perf_counter() - t) * 1000.0
            retrieved = [i2b[int(i)] for i in ids]
            rows.append({
                "query": q["source_table"],
                "n_retrieved": len(retrieved),
                "hit": M.hit_rate(retrieved, {source}),
                "runtime_ms": runtime_ms,
            })
    finally:
        db.close()
    summary = {
        "evaluated": len(rows),
        "total_queries": len(queries),
        "effective_n": len(rows),
        "skipped": len(queries) - len(rows),
        "mean_hit_rate": mean(r["hit"] for r in rows) if rows else 0.0,
        "wall_s": perf_counter() - t0,
    }
    return rows, summary
