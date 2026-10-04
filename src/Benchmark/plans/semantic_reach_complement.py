"""Bench plan: deployable hybrid (semantic + syntactic) vs each standalone leg."""
from __future__ import annotations

from src.Benchmark.plans.registry import (
    ArmResult, PlanContext, PlanResult, PlanSpec, register,
)

_POINT = (
    "Deployable comparison of semantic (SU/SJ) vs the proper syntactic seeker "
    "(UnionCounterSearch for union, SC for join) and their RRF fusion, with a "
    "reachability diagnostic; quantifies realizable complementarity per "
    "difficulty stratum."
)
_ARMS = ["semantic", "syntactic", "hybrid"]
_Kc = 200
_RRF_KC = 60


def fuse_rrf(
    legs: list[list[str]],
    weights: list[float],
    kc: int = 60,
    cut: int | None = None,
) -> list[str]:
    """Weighted RRF fusion: score(t) = sum_i weights[i] / (kc + rank_i(t)), rank 0-based."""
    scores: dict[str, float] = {}
    seen_order: list[str] = []
    seen_set: set[str] = set()
    for leg, w in zip(legs, weights):
        for rank, item in enumerate(leg):
            if item not in seen_set:
                seen_set.add(item)
                seen_order.append(item)
                scores[item] = 0.0
            scores[item] += w / (kc + rank)
    result = sorted(seen_order, key=lambda t: -scores[t])
    return result[:cut] if cut is not None else result


def _dedup_ordered(lst: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in lst:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _gather_candidate_tokens(df_table) -> list[str]:
    from src.Index.tokenize import tokenize_cell
    seen: set[str] = set()
    for col in df_table.columns:
        for val in df_table[col]:
            tok = tokenize_cell(val)
            if len(tok) >= 2 and not tok.isdigit():
                seen.add(tok)
    return list(seen)


from src.Benchmark.plans.semantic_oracle_repair import _seeker_ids, _plan_units  # noqa: E402


def _plan_ids(plan_obj, db, i2b) -> list[str]:
    from src.Benchmark.db import bind_plan
    bind_plan(plan_obj, db)
    return [i2b[int(i)] for i in plan_obj.run()]


def _summary(rows: list[dict]) -> dict:
    from statistics import mean

    def f(key):
        return mean(r[key] for r in rows) if rows else 0.0

    return {
        "evaluated": len(rows),
        "mean_precision": f("precision"),
        "mean_recall": f("recall"),
        "mean_ndcg": f("ndcg"),
        "MAP": f("ap"),
        "MRR": f("rr"),
        "mean_runtime_ms": f("runtime_ms"),
    }


def _metrics_row(qid: str, arm: str, ranked_k: list[str], relevant: set[str],
                 runtime_ms: float, M) -> dict:
    return {
        "query": qid, "arm": arm,
        "n_relevant": len(relevant), "n_retrieved": len(ranked_k),
        "hits": sum(1 for r in ranked_k if r in relevant),
        "precision": M.precision_at_k(ranked_k, relevant),
        "recall": M.recall_at_k(ranked_k, relevant),
        "ndcg": M.ndcg_at_k(ranked_k, relevant, len(ranked_k) or 1),
        "ap": M.average_precision(ranked_k, relevant),
        "rr": M.reciprocal_rank(ranked_k, relevant),
        "runtime_ms": runtime_ms,
    }


def _bucket(sem_recall: float) -> str:
    if sem_recall == 0.0:
        return "hard"
    if sem_recall <= 0.3:
        return "low"
    if sem_recall <= 0.7:
        return "mid"
    return "high"


def run(ctx: PlanContext) -> PlanResult:  # noqa: C901
    import time
    from statistics import mean
    from tqdm import tqdm

    from src.Benchmark import metrics as M
    from src.Benchmark.datasource import load_sidecar, load_query_table
    from src.Operators import Seekers
    from src.Semantic.config import SemanticConfig, SemanticOp
    from src.Semantic.retrieve import IndexHandle
    from src.Tasks.UnionCounterSearch import UnionCounterSearch

    i2b, _lake = load_sidecar(ctx.dataset)
    n_tables = len(i2b)

    cfg = SemanticConfig.load(overrides={"dataset": ctx.dataset})
    op = SemanticOp.SU if ctx.task == "union" else SemanticOp.SJ
    oc = cfg.operator(op)
    handle = IndexHandle.open(cfg, oc.approach, oc.index_name)
    enrolled_tables = set(handle.table_to_int_id)
    enrolled_cols = set(handle.table_col_to_gid)

    units = _plan_units(ctx, enrolled_tables, enrolled_cols, i2b)
    if ctx.limit:
        units = units[: ctx.limit]

    k = ctx.k
    rows: dict[str, list[dict]] = {a: [] for a in _ARMS}
    diagnostics: dict[str, dict] = {}
    cached_lists: dict[str, tuple] = {}

    for u in tqdm(units, desc=f"semantic_reach_complement/{ctx.task}"):
        qid = u["qid"]
        relevant_raw = u["relevant"]

        if ctx.task == "union":
            qbn = qid
        else:
            qbn = qid.split("::", 1)[0]

        relevant = relevant_raw - {qbn}
        if not relevant:
            continue

        t0 = time.perf_counter()
        sem_full = [x for x in _seeker_ids(u["seeker"](), ctx.db, i2b) if x != qbn]
        t_sem = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        if ctx.task == "union":
            df_table = load_query_table(ctx.dataset, qbn)
            plan_obj = UnionCounterSearch(df_table, _Kc)
            syn_full = _dedup_ordered(
                [x for x in _plan_ids(plan_obj, ctx.db, i2b) if x != qbn])
        else:
            _, col = qid.split("::", 1)
            df_table = load_query_table(ctx.dataset, qbn)
            cells = [c for c in df_table[col].dropna().tolist() if str(c).strip()]
            if cells:
                syn_raw = _seeker_ids(Seekers.SC(cells, k=_Kc), ctx.db, i2b)
                syn_full = _dedup_ordered([x for x in syn_raw if x != qbn])
            else:
                syn_full = []
        t_syn = (time.perf_counter() - t1) * 1000.0

        hybrid_full = fuse_rrf([sem_full, syn_full], [0.5, 0.5], kc=_RRF_KC)

        if ctx.task == "union":
            all_tokens = _gather_candidate_tokens(load_query_table(ctx.dataset, qbn))
            if all_tokens:
                reach_raw = _seeker_ids(
                    Seekers.Keyword(all_tokens, k=n_tables), ctx.db, i2b)
                reach_full = [x for x in reach_raw if x != qbn]
            else:
                reach_full = []
        else:
            all_cells = [c for c in df_table[col].dropna().tolist() if str(c).strip()]
            if all_cells:
                reach_raw = _seeker_ids(
                    Seekers.SC(all_cells, k=n_tables), ctx.db, i2b)
                reach_full = _dedup_ordered([x for x in reach_raw if x != qbn])
            else:
                reach_full = []

        sem_k = sem_full[:k]
        syn_k = syn_full[:k]
        hyb_k = hybrid_full[:k]

        rows["semantic"].append(
            _metrics_row(qid, "semantic", sem_k, relevant, t_sem, M))
        rows["syntactic"].append(
            _metrics_row(qid, "syntactic", syn_k, relevant, t_syn, M))

        reach_set = relevant & set(reach_full)
        unreach = relevant - reach_set
        sem_tp = set(sem_k) & relevant
        syn_tp = set(syn_k) & relevant
        union_tp = sem_tp | syn_tp
        sem_recall = M.recall_at_k(sem_k, relevant)

        diag = {
            "n_relevant": len(relevant),
            "n_reach": len(reach_set),
            "n_unreach": len(unreach),
            "semantic_only_recovery": len(sem_tp & unreach),
            "sem_tp": len(sem_tp),
            "syn_tp": len(syn_tp),
            "sem_only_at_k": len(sem_tp - syn_tp),
            "syn_only_at_k": len(syn_tp - sem_tp),
            "hit_jaccard": len(sem_tp & syn_tp) / len(union_tp) if union_tp else 0.0,
            "sem_recall": sem_recall,
        }
        diagnostics[qid] = diag

        rows["hybrid"].append(
            _metrics_row(qid, "hybrid", hyb_k, relevant, t_sem + t_syn, M))

        cached_lists[qid] = (sem_full, syn_full, relevant)

    arms = [ArmResult(a, rows[a], _summary(rows[a])) for a in _ARMS]
    base_sem = arms[0].summary
    deltas = {
        f"{a.name}_minus_semantic": {
            "mean_recall": a.summary["mean_recall"] - base_sem["mean_recall"],
            "mean_precision": a.summary["mean_precision"] - base_sem["mean_precision"],
            "mean_ndcg": a.summary["mean_ndcg"] - base_sem["mean_ndcg"],
            "MAP": a.summary["MAP"] - base_sem["MAP"],
        }
        for a in arms[1:]
    }

    _BUCKETS = ["hard", "low", "mid", "high"]
    strata_arm_rows: dict[str, dict[str, list[dict]]] = {
        bkt: {a: [] for a in _ARMS} for bkt in _BUCKETS
    }
    for arm_name in _ARMS:
        for r in rows[arm_name]:
            bkt = _bucket(diagnostics.get(r["query"], {}).get("sem_recall", 0.0))
            strata_arm_rows[bkt][arm_name].append(r)

    strata: dict[str, dict] = {}
    for bkt in _BUCKETS:
        n = len(strata_arm_rows[bkt]["semantic"])
        bkt_stat: dict = {"n": n}
        for arm_name in _ARMS:
            arm_r = strata_arm_rows[bkt][arm_name]
            if arm_r:
                bkt_stat[arm_name] = {
                    "mean_recall": mean(r["recall"] for r in arm_r),
                    "mean_precision": mean(r["precision"] for r in arm_r),
                    "mean_ndcg": mean(r["ndcg"] for r in arm_r),
                    "MAP": mean(r["ap"] for r in arm_r),
                }
        sem_rows_bkt = strata_arm_rows[bkt]["semantic"]
        if sem_rows_bkt:
            bkt_stat["mean_semantic_only_recovery"] = mean(
                diagnostics[r["query"]].get("semantic_only_recovery", 0)
                for r in sem_rows_bkt)
            bkt_stat["mean_sem_only_at_k"] = mean(
                diagnostics[r["query"]].get("sem_only_at_k", 0)
                for r in sem_rows_bkt)
            bkt_stat["mean_syn_only_at_k"] = mean(
                diagnostics[r["query"]].get("syn_only_at_k", 0)
                for r in sem_rows_bkt)
            bkt_stat["mean_hit_jaccard"] = mean(
                diagnostics[r["query"]].get("hit_jaccard", 0.0)
                for r in sem_rows_bkt)
        strata[bkt] = bkt_stat

    w_vals = [round(i * 0.1, 1) for i in range(11)]
    weight_sweep: dict[str, dict] = {}
    for w in w_vals:
        ws_recalls: list[float] = []
        ws_ndcgs: list[float] = []
        ws_strata: dict[str, dict[str, list]] = {bkt: {"recalls": [], "ndcgs": []} for bkt in _BUCKETS}
        for qid, (sem_list, syn_list, rel) in cached_lists.items():
            ranked = fuse_rrf([sem_list, syn_list], [w, 1.0 - w], kc=_RRF_KC, cut=k)
            rec = M.recall_at_k(ranked, rel)
            ndg = M.ndcg_at_k(ranked, rel, k)
            ws_recalls.append(rec)
            ws_ndcgs.append(ndg)
            bkt = _bucket(diagnostics.get(qid, {}).get("sem_recall", 0.0))
            ws_strata[bkt]["recalls"].append(rec)
            ws_strata[bkt]["ndcgs"].append(ndg)

        strata_ws: dict[str, dict] = {}
        for bkt in _BUCKETS:
            recs = ws_strata[bkt]["recalls"]
            ndcs = ws_strata[bkt]["ndcgs"]
            strata_ws[bkt] = {
                "n": len(recs),
                "mean_recall": mean(recs) if recs else 0.0,
                "mean_ndcg": mean(ndcs) if ndcs else 0.0,
            }
        weight_sweep[str(w)] = {
            "overall": {
                "mean_recall": mean(ws_recalls) if ws_recalls else 0.0,
                "mean_ndcg": mean(ws_ndcgs) if ws_ndcgs else 0.0,
            },
            "strata": strata_ws,
        }

    w_star = max(weight_sweep, key=lambda w: weight_sweep[w]["overall"]["mean_ndcg"])

    return PlanResult(
        plan="semantic_reach_complement",
        point=_POINT,
        dataset=ctx.dataset,
        task=ctx.task,
        arms=arms,
        deltas=deltas,
        meta={
            "git_sha": ctx.git_sha,
            "config_snapshot": ctx.config_snapshot,
            "seed": ctx.seed,
            "k": k,
            "limit": ctx.limit,
            "evaluated": len(units),
            "strata": strata,
            "weight_sweep": weight_sweep,
            "w_star": w_star,
            "per_query_diagnostics": diagnostics,
        },
    )


_SPEC = PlanSpec(
    name="semantic_reach_complement",
    description=(
        "Deployable hybrid (semantic SU/SJ + syntactic keyword/SC) vs each standalone "
        "leg. Fixed-k scoring, reachability diagnostics, RRF weight sweep stratified by "
        "semantic recall."
    ),
    point=_POINT,
    run=run,
)
register(_SPEC)
