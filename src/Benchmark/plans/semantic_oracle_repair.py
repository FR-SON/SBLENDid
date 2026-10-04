"""Bench plan: semantic seeker (SU/SJ) + oracle keyword recall/precision repair."""
from __future__ import annotations

from src.Benchmark.plans.registry import (
    ArmResult, PlanContext, PlanResult, PlanSpec, register,
)

_POINT = (
    "A semantic seeker is more useful inside Blend with a syntactic keyword leg "
    "than standalone: an oracle keyword recovers missed true positives "
    "(recall up) and removes false positives (precision up)."
)
_ARMS = ["seeker_only", "keyword_only", "recall_repair", "precision_repair", "both"]
_KW_K = 100


def _seeker_ids(seeker, db, i2b) -> list[str]:
    from src.Plan import Plan
    from src.Benchmark.db import bind_plan
    plan = Plan()
    plan.add("s", seeker)
    bind_plan(plan, db)
    return [i2b[int(i)] for i in plan.run()]


def _plan_units(ctx, enrolled_tables, enrolled_cols, i2b, seeker_k=None) -> list[dict]:
    import pandas as pd
    from src.Benchmark.datasource import (
        load_union_queries, load_union_gt, load_join_queries, load_join_gt,
        load_query_table,
    )
    from src.Operators import Seekers

    k = seeker_k if seeker_k is not None else ctx.k
    ov = {"dataset": ctx.dataset, "query_encode": "off"}
    lake = set(i2b.values())
    units: list[dict] = []

    if ctx.task == "union":
        gt = load_union_gt(ctx.dataset)
        for q in load_union_queries(ctx.dataset):
            if q not in enrolled_tables:
                continue
            rel = gt.get(q, set()) & lake
            if not rel:
                continue
            df = load_query_table(ctx.dataset, q)
            df.attrs["table_id"] = q

            def make(df=df):
                return Seekers.SU(df, k=k, config_overrides=ov)

            units.append({"qid": q, "qtable": q, "relevant": rel, "seeker": make})
            if ctx.limit and len(units) >= ctx.limit:
                return units
        return units

    gt = load_join_gt(ctx.dataset)
    for (tbl, col) in load_join_queries(ctx.dataset):
        if (tbl, col) not in enrolled_cols:
            continue
        rel = gt.get((tbl, col), set()) & lake
        if not rel:
            continue
        df = pd.DataFrame({col: []})
        df.attrs["table_id"] = tbl

        def make(df=df, col=col):
            return Seekers.SJ(df, k=k, query_col_name=col, config_overrides=ov)

        units.append({"qid": f"{tbl}::{col}", "qtable": tbl, "relevant": rel, "seeker": make})
        if ctx.limit and len(units) >= ctx.limit:
            return units
    return units


def _keyword_ids(terms, ctx, i2b):
    import time
    if not terms:
        return [], 0.0
    from src.Operators import Seekers
    t = time.perf_counter()
    ids = _seeker_ids(Seekers.Keyword(terms, k=_KW_K), ctx.db, i2b)
    return ids, (time.perf_counter() - t) * 1000.0


def _metrics_row(qid, arm, ranked, relevant, runtime_ms, M):
    return {
        "query": qid, "arm": arm,
        "n_relevant": len(relevant), "n_retrieved": len(ranked),
        "hits": sum(1 for r in ranked if r in relevant),
        "precision": M.precision_at_k(ranked, relevant),
        "recall": M.recall_at_k(ranked, relevant),
        "ndcg": M.ndcg_at_k(ranked, relevant, len(ranked) or 1),
        "ap": M.average_precision(ranked, relevant),
        "rr": M.reciprocal_rank(ranked, relevant),
        "runtime_ms": runtime_ms,
    }


def _summary(rows):
    from statistics import mean

    def f(key):
        return mean(r[key] for r in rows) if rows else 0.0

    return {
        "evaluated": len(rows),
        "mean_precision": f("precision"), "mean_recall": f("recall"),
        "mean_ndcg": f("ndcg"), "MAP": f("ap"), "MRR": f("rr"),
        "mean_runtime_ms": f("runtime_ms"),
    }


def run(ctx: PlanContext) -> PlanResult:
    import time
    from tqdm import tqdm

    from src.Benchmark import metrics as M
    from src.Benchmark.datasource import load_sidecar
    from src.Benchmark.self_filter import without_self
    from src.Benchmark.plans.oracle import (
        load_oracle, repair_recall, repair_precision, repair_both,
    )
    from src.Semantic.config import SemanticConfig, SemanticOp
    from src.Semantic.retrieve import IndexHandle

    i2b, _lake = load_sidecar(ctx.dataset)
    cfg = SemanticConfig.load(overrides={"dataset": ctx.dataset})
    op = SemanticOp.SU if ctx.task == "union" else SemanticOp.SJ
    oc = cfg.operator(op)
    handle = IndexHandle.open(cfg, oc.approach, oc.index_name)
    enrolled_tables = set(handle.table_to_int_id)
    enrolled_cols = set(handle.table_col_to_gid)
    oracle = load_oracle(ctx.dataset, ctx.task)

    units = _plan_units(ctx, enrolled_tables, enrolled_cols, i2b)
    if ctx.limit:
        units = units[: ctx.limit]

    rows = {a: [] for a in _ARMS}
    for u in tqdm(units, desc=f"semantic_oracle_repair/{ctx.task}"):
        relevant = u["relevant"]
        t = time.perf_counter()
        R = without_self(_seeker_ids(u["seeker"](), ctx.db, i2b), u["qtable"])
        t_seek = (time.perf_counter() - t) * 1000.0
        misses = relevant - set(R)
        fps = set(R) - relevant
        terms = oracle.get(u["qid"], {})
        rec_ids, t_rec = _keyword_ids(terms.get("terms_recall"), ctx, i2b)
        fp_ids, t_fp = _keyword_ids(terms.get("terms_fp"), ctx, i2b)
        rec_ids = without_self(rec_ids, u["qtable"])
        fp_ids = without_self(fp_ids, u["qtable"])
        ranked = {
            "seeker_only": (R, t_seek),
            "keyword_only": (rec_ids[: ctx.k], t_rec),
            "recall_repair": (repair_recall(R, rec_ids, misses), t_seek + t_rec),
            "precision_repair": (repair_precision(R, fp_ids, fps), t_seek + t_fp),
            "both": (repair_both(R, rec_ids, fp_ids, misses, fps),
                     t_seek + t_rec + t_fp),
        }
        for a, (lst, rt) in ranked.items():
            rows[a].append(_metrics_row(u["qid"], a, lst, relevant, rt, M))

    arms = [ArmResult(a, rows[a], _summary(rows[a])) for a in _ARMS]
    base = arms[0].summary
    deltas = {
        f"{a.name}_minus_seeker_only": {
            "mean_recall": a.summary["mean_recall"] - base["mean_recall"],
            "mean_precision": a.summary["mean_precision"] - base["mean_precision"],
            "mean_ndcg": a.summary["mean_ndcg"] - base["mean_ndcg"],
            "MAP": a.summary["MAP"] - base["MAP"],
        }
        for a in arms[1:]
    }
    return PlanResult(
        plan="semantic_oracle_repair", point=_POINT, dataset=ctx.dataset,
        task=ctx.task, arms=arms, deltas=deltas,
        meta={"git_sha": ctx.git_sha, "config_snapshot": ctx.config_snapshot,
              "seed": ctx.seed, "k": ctx.k, "limit": ctx.limit,
              "evaluated": len(units), "oracle_entries": len(oracle)},
    )


_SPEC = PlanSpec(
    name="semantic_oracle_repair",
    description="Semantic seeker (SU union / SJ join) + oracle keyword recall/precision repair.",
    point=_POINT, run=run,
)
register(_SPEC)
