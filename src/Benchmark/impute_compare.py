from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import csv as _csv
from collections import namedtuple
from math import ceil
from statistics import mean, median
from time import perf_counter

import pandas as pd
from tqdm import tqdm

from src.Benchmark.candidate_store import CandidateStore
from src.Benchmark.datasource import dataset_dir, load_query_table, load_sidecar
from src.Benchmark.db import bind_plan, open_dataset_db
from src.Benchmark.impute_resolve import _norm, predict, score
from src.Benchmark.recipes.imputation import _reconstruct
from src.Operators import Combiners, Seekers
from src.Plan import Plan
from src.Semantic.retrieve import QueryColumnNotIndexed
from src.Tasks.DataImputation import DataImputation
from src.Tasks.SingleColumnJoinSearch import SingleColumnJoinSearch

_QueryCtx = namedtuple(
    "_QueryCtx",
    "dataset source key_col val_col df examples query_keys masked_keys truth")


def build_ctx(dataset: str, q: dict):
    src = q["source_table"]
    df = load_query_table(dataset, src)
    examples, query_keys = _reconstruct(dataset_dir(dataset) / "csvs" / src,
                                        q["key_col"], q["val_col"])
    if len(query_keys) == 0:
        return None
    masked = df.iloc[5:][[q["key_col"], q["val_col"]]]
    truth: dict[str, str] = {}
    for k, v in zip(masked[q["key_col"]].map(_norm), masked[q["val_col"]].map(_norm)):
        if k and v and k not in truth:
            truth[k] = v
    if not truth:
        return None
    return _QueryCtx(dataset, src, q["key_col"], q["val_col"], df, examples,
                     query_keys, set(truth), truth)


def _build_syntactic_join(ctx: _QueryCtx, k: int, k_coarse: int | None = None):
    return SingleColumnJoinSearch(ctx.df[ctx.key_col].tolist(), k)


def _build_syntactic_paper(ctx: _QueryCtx, k: int, k_coarse: int | None = None):
    return DataImputation(ctx.examples, ctx.query_keys, k=k)


def _build_mc_only(ctx: _QueryCtx, k: int, k_coarse: int | None = None):
    plan = Plan()
    plan.add("examples_seeker", Seekers.MC(ctx.examples, k))
    return plan


def _build_semantic_join(ctx: _QueryCtx, k: int, k_coarse: int | None = None):
    qdf = ctx.df[[ctx.key_col]].copy()
    qdf.attrs["table_id"] = ctx.source
    plan = Plan()
    plan.add("sj", Seekers.SJ(qdf, query_col_name=ctx.key_col,
                              query_table_id=ctx.source, k=k, k_coarse=k_coarse,
                              config_overrides={"dataset": ctx.dataset, "query_encode": "off"}))
    return plan


def _build_semantic_paper(ctx: _QueryCtx, k: int, k_coarse: int | None = None):
    qdf = ctx.df[[ctx.key_col]].copy()
    qdf.attrs["table_id"] = ctx.source
    plan = Plan()
    plan.add("examples_seeker", Seekers.MC(ctx.examples, k * 10))
    plan.add("sj", Seekers.SJ(qdf, query_col_name=ctx.key_col,
                              query_table_id=ctx.source, k=k * 30, k_coarse=k_coarse,
                              config_overrides={"dataset": ctx.dataset, "query_encode": "off"}))
    plan.add("intersection", Combiners.Intersection(k=k),
             inputs=["examples_seeker", "sj"])
    return plan


TOKEN_LEGS = {
    "syntactic_join": _build_syntactic_join,
    "syntactic_paper": _build_syntactic_paper,
    "mc_only": _build_mc_only,
}
SEMANTIC_LEGS = {
    "semantic_join": _build_semantic_join,
    "semantic_paper": _build_semantic_paper,
}
ALL_LEGS = {**TOKEN_LEGS, **SEMANTIC_LEGS}


def resolve_query(ctx: _QueryCtx, retrieved_ids, store, source_tableid) -> dict:
    ranked, col_sets = [], {}
    for i in retrieved_ids:
        tid = int(i)
        if tid == source_tableid:
            continue
        df = store.get(tid)
        if df is None:
            continue
        ranked.append((tid, df))
        col_sets[tid] = store.col_sets(tid)
    preds = predict(ranked, ctx.examples, ctx.masked_keys,
                    cand_normalized=True, col_sets_by_key=col_sets)
    return score(preds, ctx.truth)


def load_queries(dataset: str) -> list[dict]:
    p = dataset_dir(dataset) / "bench" / "imputation" / "query.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"no prepared imputation queries at {p} "
            f"(run `bench prepare --task imputation --dataset {dataset}`)")
    with p.open(newline="") as f:
        return list(_csv.DictReader(f))


def _agg(counts: list[dict]) -> dict:
    nk = sum(c["n_keys"] for c in counts)
    nc = sum(c["n_covered"] for c in counts)
    ncorr = sum(c["n_correct"] for c in counts)
    return {
        "coverage": nc / nk if nk else 0.0,
        "accuracy_at_covered": ncorr / nc if nc else 0.0,
        "overall_accuracy": ncorr / nk if nk else 0.0,
        "n_keys": nk,
    }


def _freqs_path(dataset: str):
    from src.optimizer_paths import freqs_path
    try:
        p = freqs_path(dataset_dir(dataset))
    except FileNotFoundError:
        return None
    return p if p.exists() else None


def _load_freqs(path, tokens: set[str]) -> dict:
    out: dict[str, int] = {}
    for chunk in tqdm(pd.read_csv(path, chunksize=1_000_000),
                      desc=f"freqs/{path.parent.parent.name}", unit="Mrow"):
        hit = chunk[chunk["tokenized"].astype(str).isin(tokens)]
        out.update(zip(hit["tokenized"].astype(str), hit["frequency"]))
    return out


def _query_tokens(ctxs) -> set[str]:
    from src.DBHandler import DBHandler
    return {t for c in ctxs for col in c.examples.columns
            for t in DBHandler.clean_value_collection(c.examples[col])}


def mc_cost_score(examples: pd.DataFrame, freqs: dict) -> dict:
    """Predicted MC scan size for one query, without running anything."""
    from src.DBHandler import DBHandler
    f_cols = [sum(freqs.get(t, 0) for t in set(DBHandler.clean_value_collection(examples[c])))
              for c in examples.columns]
    prod = 1
    for f in f_cols:
        prod *= f
    return {"f_cols": f_cols, "scan_rows": sum(f_cols),
            "est_rows": min(f_cols) if f_cols else 0, "worst_case": prod}


def _report_mc_cost(scored: list[tuple], cap: int | None) -> None:
    ranked = sorted(scored, key=lambda t: t[1]["scan_rows"], reverse=True)
    rows = [s["scan_rows"] for _, s in ranked]
    p90 = rows[int(0.1 * len(rows))]
    print(f"\n--- predicted MC cost ({len(rows)} queries), ranked by scan_rows ---")
    print(f"  scan_rows: max={rows[0]:,}  p90={p90:,}  median={median(rows):,.0f}  min={rows[-1]:,}")
    for src, s in ranked[:10]:
        flag = "  EXCLUDED" if cap is not None and s["scan_rows"] > cap else ""
        print(f"  scan={s['scan_rows']:>12,}  cols={s['f_cols']}  est={s['est_rows']:>10,}"
              f"  worst={s['worst_case']:>16,}  {src}{flag}")
    if len(ranked) > 10:
        print(f"  ... {len(ranked) - 10} more")


def _p95(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[max(0, ceil(0.95 * len(s)) - 1)]


def _warm(build, ctxs, k, k_coarse, db, n: int) -> None:
    for _ in range(n):
        for ctx in ctxs[:3]:
            try:
                plan = build(ctx, k, k_coarse)
                bind_plan(plan, db)
                plan.run()
                break
            except Exception as e:
                print(f"warmup skipped for {ctx.source}: {e!r}", flush=True)


def run_compare(dataset: str, *, k_grid: list[int], legs: dict,
                k_coarse: int | None = None, resolve_cache_max: int | None = None,
                warmup: int = 1, mc_cost_cap: int | None = None) -> dict:
    i2b, _ = load_sidecar(dataset)
    b2i = {v: k for k, v in i2b.items()}
    queries = load_queries(dataset)
    ctxs = [c for c in (build_ctx(dataset, q)
                        for q in tqdm(queries, desc=f"build-ctx/{dataset}"))
            if c is not None]
    # score MC cost before any seeker runs: a frequent token makes MC's scan OOM
    fp = _freqs_path(dataset)
    mc_cost, excluded = {}, []
    if fp is None:
        msg = (f"no freqs.csv for {dataset} — build it with "
               f"`python -m src.Optimizer.cli build-freqs --lake {dataset}`")
        if mc_cost_cap is not None:
            raise SystemExit(f"--mc-cost-cap needs the token frequencies: {msg}")
        print(f"{msg}; MC cost scoring off", flush=True)
    else:
        freqs = _load_freqs(fp, _query_tokens(ctxs))
        scored = [(c.source, mc_cost_score(c.examples, freqs)) for c in ctxs]
        mc_cost = dict(scored)
        _report_mc_cost(scored, mc_cost_cap)
        if mc_cost_cap is not None:
            excluded = [s for s, sc in scored if sc["scan_rows"] > mc_cost_cap]
            drop = set(excluded)
            ctxs = [c for c in ctxs if c.source not in drop]
            print(f"excluded {len(excluded)} of {len(scored)} queries "
                  f"above --mc-cost-cap {mc_cost_cap:,}", flush=True)
    db = open_dataset_db(dataset)
    store = CandidateStore(
        db,
        sql="SELECT colid, rowid, tokenized FROM AllTables WHERE tableid = {tableid}",
        max_tables=resolve_cache_max,
    )
    results: dict[str, dict] = {leg: {} for leg in legs}
    try:
        for leg, build in legs.items():
            for k in k_grid:
                counts, times, resolve_ms, n_ret = [], [], [], []
                if warmup and ctxs:
                    _warm(build, ctxs, k, k_coarse, db, warmup)
                misses0 = getattr(store, "misses", 0)
                unindexed = 0
                for ctx in tqdm(ctxs, desc=f"{leg}/k{k}/{dataset}"):
                    plan = build(ctx, k, k_coarse)
                    bind_plan(plan, db)
                    t = perf_counter()
                    try:
                        ids = plan.run()
                    except QueryColumnNotIndexed:
                        unindexed += 1
                        continue
                    times.append((perf_counter() - t) * 1000.0)
                    n_ret.append(len(ids))
                    t = perf_counter()
                    counts.append(resolve_query(ctx, ids, store, b2i.get(ctx.source)))
                    resolve_ms.append((perf_counter() - t) * 1000.0)
                cell = _agg(counts)
                cell["retrieval_runtime_ms"] = mean(times) if times else 0.0
                cell["retrieval_runtime_ms_median"] = median(times) if times else 0.0
                cell["retrieval_runtime_ms_p95"] = _p95(times)
                cell["resolve_runtime_ms"] = mean(resolve_ms) if resolve_ms else 0.0
                cell["n_retrieved"] = mean(n_ret) if n_ret else 0.0
                cell["resolve_cold_fetches"] = getattr(store, "misses", 0) - misses0
                cell["skipped_unindexed"] = unindexed
                results[leg][str(k)] = cell
    finally:
        db.close()
    return {"dataset": dataset, "k_grid": k_grid, "n_queries": len(ctxs),
            "k_coarse": k_coarse, "warmup": warmup, "mc_cost_cap": mc_cost_cap,
            "mc_cost": mc_cost, "excluded": excluded, "results": results}


_METRICS = ("overall_accuracy", "coverage", "accuracy_at_covered")
_DELTAS = [("semantic_join", "syntactic_join"),
           ("semantic_paper", "syntactic_paper"),
           ("semantic_join", "semantic_paper")]


def add_comparison(out: dict) -> dict:
    res = out["results"]
    comp: dict[str, dict] = {}
    for k in out["k_grid"]:
        ks = str(k)
        block = {}
        for a, b in _DELTAS:
            if a in res and b in res:
                block[f"{a}_minus_{b}"] = {
                    m: res[a][ks][m] - res[b][ks][m] for m in _METRICS}
        comp[ks] = block
    res["comparison"] = comp
    return out
