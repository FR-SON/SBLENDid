from __future__ import annotations

from statistics import mean
from time import perf_counter

from tqdm import tqdm

from src.Benchmark import metrics as M
from src.Benchmark.datasource import (
    load_join_gt, load_join_queries, load_query_table, load_sidecar,
    to_basenames)
from src.Benchmark.db import bind_plan, open_dataset_db
from src.Benchmark.regime import record_pair_hits
from src.Benchmark.self_filter import without_self
from src.Tasks.SingleColumnJoinSearch import SingleColumnJoinSearch


def _build_plan(seeker: str, values, df, qcol: str, qtable: str, k: int,
                dataset: str):
    if seeker == "sc":
        return SingleColumnJoinSearch(values, k)
    from src.Operators import Seekers
    from src.Plan import Plan
    plan = Plan()
    # index + sidecar follow --dataset, not config [Dataset].name
    ov = {"dataset": dataset, "query_encode": "off"}
    if seeker == "sho":
        colid = list(df.columns).index(qcol)
        plan.add("sho", Seekers.SHO(values, k, query_table_id=qtable,
                                    query_col_id=colid, config_overrides=ov))
    elif seeker == "sj":
        plan.add("sj", Seekers.SJ(df[[qcol]], k=k, query_table_id=qtable,
                                  config_overrides=ov))
    else:
        raise ValueError(seeker)
    return plan


def run_join(dataset: str, k: int, seeker: str = "sc",
            pair_sink: list | None = None) -> tuple[list[dict], dict]:
    i2b, lake = load_sidecar(dataset)
    queries = load_join_queries(dataset)
    gt = load_join_gt(dataset)
    db = open_dataset_db(dataset)
    rows: list[dict] = []
    planned = []
    for qtable, qcol in queries:
        if qtable not in lake:
            continue
        relevant = gt.get((qtable, qcol), set()) & lake
        if relevant:
            planned.append((qtable, qcol, relevant))
    skipped = len(queries) - len(planned)
    # empty tuple catches nothing; keeps torch/faiss off the sc path
    miss_exc: tuple = ()
    if seeker in {"sj", "sho"}:
        from src.Semantic.retrieve import QueryColumnNotIndexed
        miss_exc = (QueryColumnNotIndexed,)
    t0 = perf_counter()
    try:
        for qtable, qcol, relevant in tqdm(planned, desc=f"join/{dataset}"):
            df = load_query_table(dataset, qtable)
            if qcol not in df.columns:
                skipped += 1
                continue
            values = df[qcol].tolist()
            # over-fetch by 1 so dropping self still leaves k results
            plan = _build_plan(seeker, values, df, qcol, qtable, k + 1, dataset)
            bind_plan(plan, db)
            t = perf_counter()
            try:
                ids = plan.run()
            except miss_exc:
                skipped += 1
                continue
            runtime_ms = (perf_counter() - t) * 1000.0
            retrieved = without_self(to_basenames(i2b, ids, dataset), qtable)[:k]
            if pair_sink is not None:
                record_pair_hits(pair_sink, task="join", query_table=qtable, query_column=qcol,
                                 retrieved=[(r, "") for r in retrieved],
                                 relevant={(r, "") for r in relevant},
                                 dataset=dataset, variant=seeker)
            rows.append({
                "query": qtable,
                "query_column": qcol,
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
