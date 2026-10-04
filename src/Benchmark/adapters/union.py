from __future__ import annotations

from statistics import mean
from time import perf_counter

from tqdm import tqdm

from src.Benchmark import metrics as M
from src.Benchmark.datasource import (
    load_query_table, load_sidecar, load_union_gt, load_union_queries,
    to_basenames)
from src.Benchmark.db import bind_plan, open_dataset_db
from src.Benchmark.regime import record_pair_hits
from src.Benchmark.self_filter import without_self
from src.Tasks.UnionCounterSearch import UnionCounterSearch


def _build_union_plan(seeker: str, df, k: int, qtable=None, dataset=None):
    if seeker == "sc":
        return UnionCounterSearch(df, k)
    if seeker == "sj":
        raise ValueError("sj is a join seeker; not valid for union")
    from src.Operators import Combiners, Seekers
    from src.Plan import Plan
    # index + sidecar follow --dataset, not config [Dataset].name
    ov = {"query_encode": "off", **({"dataset": dataset} if dataset else {})}
    if seeker == "su":
        plan = Plan()
        plan.add("su", Seekers.SU(df, k=k, query_table_id=qtable,
                                  config_overrides=ov))
        return plan
    if seeker != "sho":
        raise ValueError(seeker)
    plan = Plan()
    for i, col in enumerate(df.columns):
        plan.add(col, Seekers.SHO(df[col].tolist(), k * 10,
                                  query_table_id=qtable, query_col_id=i,
                                  config_overrides=ov))
    plan.add("counter", Combiners.Counter(k=k), inputs=df.columns)
    return plan


def run_union(dataset: str, k: int, seeker: str = "sc",
              pair_sink: list | None = None) -> tuple[list[dict], dict]:
    i2b, lake = load_sidecar(dataset)
    queries = load_union_queries(dataset)
    gt = load_union_gt(dataset)
    db = open_dataset_db(dataset)
    rows: list[dict] = []
    planned = []
    for q in queries:
        if q not in lake:
            continue
        relevant = gt.get(q, set()) & lake
        if relevant:
            planned.append((q, relevant))
    skipped = len(queries) - len(planned)
    t0 = perf_counter()
    try:
        for q, relevant in tqdm(planned, desc=f"union/{dataset}"):
            df = load_query_table(dataset, q)
            # over-fetch by 1 so dropping self still leaves k results
            plan = _build_union_plan(seeker, df, k + 1, qtable=q, dataset=dataset)
            bind_plan(plan, db)
            t = perf_counter()
            ids = plan.run()
            runtime_ms = (perf_counter() - t) * 1000.0
            retrieved = without_self(to_basenames(i2b, ids, dataset), q)[:k]
            if pair_sink is not None:
                record_pair_hits(pair_sink, task="union", query_table=q, query_column="",
                                 retrieved=retrieved, relevant=relevant,
                                 dataset=dataset, variant=seeker)
            rows.append({
                "query": q,
                "n_gt_in_lake": len(relevant),
                "n_retrieved": len(retrieved),
                "hits": sum(1 for r in retrieved if r in relevant),
                "p@k": M.precision_at_k(retrieved, relevant),
                "recall@k": M.recall_at_k(retrieved, relevant),
                "ap": M.average_precision(retrieved, relevant),
                "rr": M.reciprocal_rank(retrieved, relevant),
                "runtime_ms": runtime_ms,
            })
    finally:
        db.close()
    summary = {
        "evaluated": len(rows),
        "skipped": skipped,
        "effective_n": len(rows),
        "total_queries": len(queries),
        "mean_p@k": mean(r["p@k"] for r in rows) if rows else 0.0,
        "mean_recall@k": mean(r["recall@k"] for r in rows) if rows else 0.0,
        "MAP": mean(r["ap"] for r in rows) if rows else 0.0,
        "MRR": mean(r["rr"] for r in rows) if rows else 0.0,
        "wall_s": perf_counter() - t0,
    }
    return rows, summary
