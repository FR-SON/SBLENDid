"""Bench plan: deployable keyword-assist (SU/SJ + oracle keyword legs, no GT filter)."""
from __future__ import annotations

from src.Benchmark.plans.registry import (
    ArmResult, PlanContext, PlanResult, PlanSpec, register,
)

_POINT = (
    "Deployable keyword-assist using real Blend Union/Difference combiners: "
    "Union(SU, keyword_recall) improves recall; Difference(SU, keyword_fp) removes FPs "
    "(SU/SJ honour the NOT IN exclusion pushdown). "
    "k-sweep over set-retrieval metrics (nDCG/MAP unreliable for Union arms)."
)
_ARMS = ["seeker_only", "keyword_only", "union_assist", "precision_assist", "both"]
_K_GRID = [5, 10, 20, 50]
_MAX_K = max(_K_GRID)


def _combiner_ids(a_seeker, b_seeker, Comb, k, db, i2b) -> list[str]:
    from src.Plan import Plan
    from src.Benchmark.db import bind_plan
    p = Plan()
    p.add("a", a_seeker)
    p.add("b", b_seeker)
    p.add("c", Comb(k=k), inputs=["a", "b"])
    bind_plan(p, db)
    return [i2b[int(i)] for i in p.run()]


def _both_ids(su, kwf, kwr, k, db, i2b) -> list[str]:
    from src.Plan import Plan
    from src.Operators import Combiners
    from src.Benchmark.db import bind_plan
    p = Plan()
    p.add("su", su)
    p.add("kwf", kwf)
    p.add("kwr", kwr)
    p.add("d", Combiners.Difference(k=k), inputs=["su", "kwf"])
    p.add("u", Combiners.Union(k=k), inputs=["d", "kwr"])
    bind_plan(p, db)
    return [i2b[int(i)] for i in p.run()]


def run(ctx: PlanContext) -> PlanResult:
    import time
    from statistics import mean
    from tqdm import tqdm

    from src.Benchmark import metrics as M
    from src.Benchmark.datasource import load_sidecar
    from src.Benchmark.self_filter import without_self
    from src.Benchmark.plans.oracle import load_oracle
    from src.Benchmark.plans.semantic_oracle_repair import _seeker_ids, _plan_units, _keyword_ids
    from src.Operators import Combiners, Seekers
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

    units_max = _plan_units(ctx, enrolled_tables, enrolled_cols, i2b, seeker_k=_MAX_K)
    if ctx.limit:
        units_max = units_max[:ctx.limit]

    gk_units_map: dict[int, list] = {}
    for gk in _K_GRID:
        gu = _plan_units(ctx, enrolled_tables, enrolled_cols, i2b, seeker_k=gk)
        if ctx.limit:
            gu = gu[:ctx.limit]
        gk_units_map[gk] = gu

    query_records: list[dict] = []
    for u in tqdm(units_max, desc=f"semantic_assist/{ctx.task}"):
        qid = u["qid"]
        relevant = u["relevant"]

        t0 = time.perf_counter()
        su_max = _seeker_ids(u["seeker"](), ctx.db, i2b)
        t_seek = (time.perf_counter() - t0) * 1000.0

        terms = oracle.get(qid, {})
        kwr_max, t_rec = _keyword_ids(terms.get("terms_recall"), ctx, i2b)
        kwf_max, t_fp = _keyword_ids(terms.get("terms_fp"), ctx, i2b)
        su_max = without_self(su_max, u["qtable"])
        kwr_max = without_self(kwr_max, u["qtable"])
        kwf_max = without_self(kwf_max, u["qtable"])
        has_terms = bool(terms.get("terms_recall") or terms.get("terms_fp"))

        query_records.append({
            "qid": qid,
            "qtable": u["qtable"],
            "relevant": relevant,
            "has_terms": has_terms,
            "terms": terms,
            "su_max": su_max,
            "kwr_max": kwr_max,
            "kwf_max": kwf_max,
            "t_seek": t_seek,
            "t_rec": t_rec,
            "t_fp": t_fp,
        })

    k_sweep_arm_results: dict[int, list[dict]] = {gk: [] for gk in _K_GRID}

    for gk in tqdm(_K_GRID, desc="k-sweep"):
        gk_units = gk_units_map[gk]
        for i, rec in enumerate(query_records):
            arm_lists: dict[str, list[str]] = {}
            arm_lists["seeker_only"] = rec["su_max"][:gk]
            arm_lists["keyword_only"] = rec["kwr_max"][:gk]

            if not rec["has_terms"]:
                arm_lists["union_assist"] = arm_lists["seeker_only"]
                arm_lists["precision_assist"] = arm_lists["seeker_only"]
                arm_lists["both"] = arm_lists["seeker_only"]
            else:
                tr = rec["terms"].get("terms_recall") or []
                tf = rec["terms"].get("terms_fp") or []
                gu = gk_units[i]

                if tr:
                    arm_lists["union_assist"] = _combiner_ids(
                        gu["seeker"](), Seekers.Keyword(tr, k=gk),
                        Combiners.Union, gk, ctx.db, i2b)
                else:
                    arm_lists["union_assist"] = arm_lists["seeker_only"]

                if tf:
                    arm_lists["precision_assist"] = _combiner_ids(
                        gu["seeker"](), Seekers.Keyword(tf, k=gk),
                        Combiners.Difference, gk, ctx.db, i2b)
                else:
                    arm_lists["precision_assist"] = arm_lists["seeker_only"]

                if tf and tr:
                    arm_lists["both"] = _both_ids(
                        gu["seeker"](), Seekers.Keyword(tf, k=gk),
                        Seekers.Keyword(tr, k=gk), gk, ctx.db, i2b)
                elif tr:
                    arm_lists["both"] = _combiner_ids(
                        gu["seeker"](), Seekers.Keyword(tr, k=gk),
                        Combiners.Union, gk, ctx.db, i2b)
                else:
                    arm_lists["both"] = _combiner_ids(
                        gu["seeker"](), Seekers.Keyword(tf, k=gk),
                        Combiners.Difference, gk, ctx.db, i2b)

            arm_lists = {a: without_self(lst, rec["qtable"]) for a, lst in arm_lists.items()}
            k_sweep_arm_results[gk].append(arm_lists)

    hl_k = ctx.k if ctx.k in _K_GRID else _K_GRID[-1]
    hl_results = k_sweep_arm_results[hl_k]

    rows = {a: [] for a in _ARMS}
    for i, rec in enumerate(query_records):
        relevant = rec["relevant"]
        qid = rec["qid"]
        arm_lists = hl_results[i]
        rt = {
            "seeker_only": rec["t_seek"],
            "keyword_only": rec["t_rec"],
            "union_assist": rec["t_seek"] + rec["t_rec"],
            "precision_assist": rec["t_seek"] + rec["t_fp"],
            "both": rec["t_seek"] + rec["t_rec"] + rec["t_fp"],
        }
        for arm in _ARMS:
            lst = arm_lists[arm]
            rows[arm].append({
                "query": qid,
                "n_relevant": len(relevant),
                "n_retrieved": len(lst),
                "hits": sum(1 for r in lst if r in relevant),
                "precision": M.precision_at_k(lst, relevant),
                "recall": M.recall_at_k(lst, relevant),
                "ndcg": M.ndcg_at_k(lst, relevant, hl_k),
                "ap": M.average_precision(lst, relevant),
                "rr": M.reciprocal_rank(lst, relevant),
                "has_terms": rec["has_terms"],
                "runtime_ms": rt[arm],
            })

    def _summary(arm_rows: list[dict]) -> dict:
        def f(key):
            return mean(r[key] for r in arm_rows) if arm_rows else 0.0
        return {
            "evaluated": len(arm_rows),
            "mean_precision": f("precision"),
            "mean_recall": f("recall"),
            "mean_ndcg": f("ndcg"),
            "MAP": f("ap"),
            "MRR": f("rr"),
            "mean_runtime_ms": f("runtime_ms"),
        }

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

    def _sweep_agg(gk: int, indices: list[int]) -> dict[str, dict]:
        result = {}
        for arm in _ARMS:
            recs, precs, f1s = [], [], []
            for i in indices:
                lst = k_sweep_arm_results[gk][i][arm]
                rel = query_records[i]["relevant"]
                prec = M.precision_at_k(lst, rel)
                rec = M.recall_at_k(lst, rel)
                recs.append(rec)
                precs.append(prec)
                f1s.append(M.f1(prec, rec))
            result[arm] = {
                "mean_recall": mean(recs) if recs else 0.0,
                "mean_precision": mean(precs) if precs else 0.0,
                "mean_f1": mean(f1s) if f1s else 0.0,
            }
        return result

    all_indices = list(range(len(query_records)))
    oracle_indices = [i for i, r in enumerate(query_records) if r["has_terms"]]

    k_sweep: dict[str, dict] = {}
    for gk in _K_GRID:
        k_sweep[str(gk)] = {
            "overall": _sweep_agg(gk, all_indices),
            "oracle_subset": _sweep_agg(gk, oracle_indices),
        }

    return PlanResult(
        plan="semantic_assist",
        point=_POINT,
        dataset=ctx.dataset,
        task=ctx.task,
        arms=arms,
        deltas=deltas,
        meta={
            "git_sha": ctx.git_sha,
            "config_snapshot": ctx.config_snapshot,
            "seed": ctx.seed,
            "k": ctx.k,
            "limit": ctx.limit,
            "evaluated": len(query_records),
            "oracle_entries": len(oracle),
            "n_oracle_subset": len(oracle_indices),
            "k_sweep": k_sweep,
            "rank_metrics_note": "nDCG/MAP unreliable for union_assist/both: Blend Union has no ORDER BY",
        },
    )


_SPEC = PlanSpec(
    name="semantic_assist",
    description=(
        "Deployable keyword-assist: semantic seeker (SU/SJ) combined with oracle keyword "
        "legs via real Blend Union/Difference combiners (no GT filter). "
        "k-sweep over set-retrieval metrics."
    ),
    point=_POINT,
    run=run,
)
register(_SPEC)
