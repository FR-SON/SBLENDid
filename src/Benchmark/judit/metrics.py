"""JUDIT-faithful metric math shared by the bench engine and the bench_*_seeker_metrics scripts."""

from __future__ import annotations

import math
from typing import Iterable

K_RANGE = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 24, 25, 36, 48, 50,
           60, 75, 100, 130, 150)
K_COARSE = 1000
K_FINAL = 150


# verbatim from JUDIT src/judit/eval.py


def _precision_at_k(ranked: list, relevant: set, k: int) -> float:
    return sum(1 for t in ranked[:k] if t in relevant) / k if k else 0.0


def _recall_at_k(ranked: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for t in ranked[:k] if t in relevant) / len(relevant)


def _ndcg_at_k(ranked: list, relevant: set, k: int) -> float:
    dcg = sum((1 / math.log2(i + 2)) for i, t in enumerate(ranked[:k]) if t in relevant)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg else 0.0


def _ap_at_k(ranked: list, relevant: set, k: int) -> float:
    hits = 0
    total = 0.0
    for i, t in enumerate(ranked[:k]):
        if t in relevant:
            hits += 1
            total += hits / (i + 1)
    return total / min(len(relevant), k) if relevant else 0.0


def _liftus_map_at_k(ranked: list, relevant: set, k: int) -> float:
    """LIFTus Eq. 13: MAP@k = (1/k) * sum_{i=1..k} P@i (divisor k, not min(|S|, k))."""
    if k <= 0 or not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for i in range(1, k + 1):
        if i <= len(ranked) and ranked[i - 1] in relevant:
            hits += 1
        total += hits / i
    return total / k


def _precision_lb_at_k(ranked: list, relevant: set, k: int) -> float:
    found = ranked[:k]
    if not found and not relevant:
        return 1.0
    if not found:
        return 0.0
    return sum(1 for t in found if t in relevant) / len(found)


def _recall_lb_at_k(ranked: list, relevant: set, k: int) -> float:
    found = ranked[:k]
    if not found and not relevant:
        return 1.0
    if not relevant:
        return 1.0
    return sum(1 for t in found if t in relevant) / len(relevant)


# verbatim from JUDIT src/judit/metrics_lib/gt_density.py


def _percentile_nearest_rank(sorted_xs: list[int], p: float) -> float:
    n = len(sorted_xs)
    if n == 0:
        return 0.0
    if n == 1:
        return float(sorted_xs[0])
    idx = max(0, min(n - 1, math.ceil(p / 100.0 * n) - 1))
    return float(sorted_xs[idx])


def gt_density_block(counts: list[int]) -> dict:
    if not counts:
        return {
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "min": 0.0,
            "max": 0.0,
            "n_queries": 0,
            "n_queries_with_gt": 0,
        }
    s = sorted(counts)
    return {
        "mean": sum(counts) / len(counts),
        "p50": _percentile_nearest_rank(s, 50.0),
        "p90": _percentile_nearest_rank(s, 90.0),
        "min": float(s[0]),
        "max": float(s[-1]),
        "n_queries": len(counts),
        "n_queries_with_gt": sum(1 for c in counts if c > 0),
    }


def pr_ceilings(counts: list[int], ks: Iterable[int]) -> tuple[dict, dict]:
    nonzero = [c for c in counts if c > 0]
    ks_list = list(ks)
    if not nonzero:
        zero = {str(k): 0.0 for k in ks_list}
        return zero, dict(zero)
    n = len(nonzero)
    prec: dict[str, float] = {}
    rec: dict[str, float] = {}
    for k in ks_list:
        if k <= 0:
            prec[str(k)] = 0.0
            rec[str(k)] = 0.0
            continue
        p_sum = 0.0
        r_sum = 0.0
        for c in nonzero:
            cap = c if c < k else k
            p_sum += cap / k
            r_sum += cap / c
        prec[str(k)] = p_sum / n
        rec[str(k)] = r_sum / n
    return prec, rec


def _corrected_block(achieved: dict, multiplier: float | None) -> dict | None:
    if multiplier is None:
        return None
    return {k: v * multiplier for k, v in achieved.items()}


def _util_pct_block(corrected: dict | None, ceiling: dict | None) -> dict | None:
    if corrected is None:
        return None
    if ceiling is None:
        return {k: v * 100 for k, v in corrected.items()}
    return {
        k: (corrected[k] / ceiling[k] * 100) if ceiling[k] > 0 else 0.0
        for k in corrected
    }


def per_query_metrics(ranked, relevant, ks=K_RANGE):
    row = {}
    for k in ks:
        row[f"precision_at_{k}"] = _precision_at_k(ranked, relevant, k)
        row[f"recall_at_{k}"] = _recall_at_k(ranked, relevant, k)
        row[f"ndcg_at_{k}"] = _ndcg_at_k(ranked, relevant, k)
        row[f"map_at_{k}"] = _ap_at_k(ranked, relevant, k)
        row[f"liftus_map_at_{k}"] = _liftus_map_at_k(ranked, relevant, k)
        row[f"precision_lb_at_{k}"] = _precision_lb_at_k(ranked, relevant, k)
        row[f"recall_lb_at_{k}"] = _recall_lb_at_k(ranked, relevant, k)
    return row


def per_query_col_metrics(retrieved_cols, relevant_cols, ks=K_RANGE):
    row = {}
    for k in ks:
        row[f"precision_col_at_{k}"] = _precision_at_k(retrieved_cols, relevant_cols, k)
        row[f"recall_col_at_{k}"] = _recall_at_k(retrieved_cols, relevant_cols, k)
        row[f"ndcg_col_at_{k}"] = _ndcg_at_k(retrieved_cols, relevant_cols, k)
        row[f"map_col_at_{k}"] = _ap_at_k(retrieved_cols, relevant_cols, k)
        row[f"liftus_map_col_at_{k}"] = _liftus_map_at_k(retrieved_cols, relevant_cols, k)
        row[f"precision_lb_col_at_{k}"] = _precision_lb_at_k(retrieved_cols, relevant_cols, k)
        row[f"recall_lb_col_at_{k}"] = _recall_lb_at_k(retrieved_cols, relevant_cols, k)
    return row


def _mean_at_k(rows: list[dict], prefix: str, ks) -> dict:
    n = len(rows) or 1
    return {str(k): sum(r[f"{prefix}_at_{k}"] for r in rows) / n for k in ks}


def _mean_at_k_opt(rows: list[dict], prefix: str, ks) -> dict | None:
    """`_mean_at_k` for an optional metric: None when the rows lack the key."""
    if any(f"{prefix}_at_{k}" not in r for r in rows for k in ks):
        return None
    return _mean_at_k(rows, prefix, ks)


def assemble_union_metrics(
    rows, gt_counts, *, ks, k_coarse, k_vote, ef_used, depths_derived,
    n_saturated, n_specs_total,
    n_queries_input, n_gt_rows_input, n_gt_rows_kept, n_dropped_q, n_dropped_c,
    wall_clock_seconds,
) -> dict:
    prec_ceil, rec_ceil = pr_ceilings(gt_counts, ks)
    coarse_saturation_pct = round(n_saturated / max(1, n_specs_total) * 100, 2)
    density = gt_density_block(gt_counts)
    nz = density["n_queries_with_gt"]
    multiplier = (density["n_queries"] / nz) if nz > 0 else None

    p_at_k = _mean_at_k(rows, "precision", ks)
    r_at_k = _mean_at_k(rows, "recall", ks)
    n_at_k = _mean_at_k(rows, "ndcg", ks)
    m_at_k = _mean_at_k(rows, "map", ks)
    p_corr = _corrected_block(p_at_k, multiplier)
    r_corr = _corrected_block(r_at_k, multiplier)
    n_corr = _corrected_block(n_at_k, multiplier)
    m_corr = _corrected_block(m_at_k, multiplier)
    lm_at_k = _mean_at_k_opt(rows, "liftus_map", ks)
    lm_corr = _corrected_block(lm_at_k, multiplier) if lm_at_k is not None else None

    return {
        "task": "union",
        "n_queries_input": n_queries_input,
        "n_queries_kept": len(rows),
        "n_gt_rows_input": n_gt_rows_input,
        "n_gt_rows_kept": n_gt_rows_kept,
        "n_dropped_missing_query": n_dropped_q,
        "n_dropped_missing_candidate": n_dropped_c,
        "gt_density": density,
        "k_coarse_used": k_coarse,
        "k_vote_used": k_vote,
        "ef_search_used": ef_used,
        "depths_derived": depths_derived,
        "coarse_saturation_pct": coarse_saturation_pct,
        "precision_at_k": p_at_k,
        "precision_at_k_ceiling": prec_ceil,
        "precision_at_k_corrected": p_corr,
        "precision_at_k_util_pct": _util_pct_block(p_corr, prec_ceil),
        "recall_at_k": r_at_k,
        "recall_at_k_ceiling": rec_ceil,
        "recall_at_k_corrected": r_corr,
        "recall_at_k_util_pct": _util_pct_block(r_corr, rec_ceil),
        "ndcg_at_k": n_at_k,
        "ndcg_at_k_corrected": n_corr,
        "ndcg_at_k_util_pct": _util_pct_block(n_corr, None),
        "map_at_k": m_at_k,
        "map_at_k_corrected": m_corr,
        "map_at_k_util_pct": _util_pct_block(m_corr, None),
        "liftus_map_at_k": lm_at_k,
        "liftus_map_at_k_corrected": lm_corr,
        "precision_lb_at_k": _mean_at_k(rows, "precision_lb", ks),
        "recall_lb_at_k": _mean_at_k(rows, "recall_lb", ks),
        "wall_clock_seconds": wall_clock_seconds,
    }


def assemble_join_metrics(
    rows, gt_counts, gt_counts_col, *, ks, k_coarse, k_vote, ef_used,
    depths_derived, n_saturated,
    n_queries_input, n_gt_rows_input, n_gt_rows_kept, n_dropped_q, n_dropped_c,
    n_gt_col_kept, n_gt_col_dropped_missing, wall_clock_seconds,
) -> dict:
    n = len(rows) or 1
    prec_ceil, rec_ceil = pr_ceilings(gt_counts, ks)
    prec_ceil_col, rec_ceil_col = pr_ceilings(gt_counts_col, ks)
    coarse_saturation_pct = round(n_saturated / max(1, n) * 100, 2)
    density = gt_density_block(gt_counts)
    density_col = gt_density_block(gt_counts_col)
    nz = density["n_queries_with_gt"]
    multiplier = (density["n_queries"] / nz) if nz > 0 else None
    nz_col = density_col["n_queries_with_gt"]
    multiplier_col = (density_col["n_queries"] / nz_col) if nz_col > 0 else None

    p_at_k = _mean_at_k(rows, "precision", ks)
    r_at_k = _mean_at_k(rows, "recall", ks)
    n_at_k = _mean_at_k(rows, "ndcg", ks)
    m_at_k = _mean_at_k(rows, "map", ks)
    pc_at_k = _mean_at_k(rows, "precision_col", ks)
    rc_at_k = _mean_at_k(rows, "recall_col", ks)
    nc_at_k = _mean_at_k(rows, "ndcg_col", ks)
    mc_at_k = _mean_at_k(rows, "map_col", ks)
    p_corr = _corrected_block(p_at_k, multiplier)
    r_corr = _corrected_block(r_at_k, multiplier)
    n_corr = _corrected_block(n_at_k, multiplier)
    m_corr = _corrected_block(m_at_k, multiplier)
    pc_corr = _corrected_block(pc_at_k, multiplier_col)
    rc_corr = _corrected_block(rc_at_k, multiplier_col)
    nc_corr = _corrected_block(nc_at_k, multiplier_col)
    mc_corr = _corrected_block(mc_at_k, multiplier_col)
    lm_at_k = _mean_at_k_opt(rows, "liftus_map", ks)
    lm_corr = _corrected_block(lm_at_k, multiplier) if lm_at_k is not None else None
    lmc_at_k = _mean_at_k_opt(rows, "liftus_map_col", ks)
    lmc_corr = (_corrected_block(lmc_at_k, multiplier_col)
                if lmc_at_k is not None else None)

    return {
        "task": "join",
        "n_queries_input": n_queries_input,
        "n_queries_kept": len(rows),
        "n_gt_rows_input": n_gt_rows_input,
        "n_gt_rows_kept": n_gt_rows_kept,
        "n_dropped_missing_query": n_dropped_q,
        "n_dropped_missing_candidate": n_dropped_c,
        "gt_density": density,
        "gt_density_col": density_col,
        "k_coarse_used": k_coarse,
        "k_vote_used": k_vote,
        "ef_search_used": ef_used,
        "depths_derived": depths_derived,
        "coarse_saturation_pct": coarse_saturation_pct,
        "precision_at_k": p_at_k,
        "precision_at_k_ceiling": prec_ceil,
        "precision_at_k_corrected": p_corr,
        "precision_at_k_util_pct": _util_pct_block(p_corr, prec_ceil),
        "recall_at_k": r_at_k,
        "recall_at_k_ceiling": rec_ceil,
        "recall_at_k_corrected": r_corr,
        "recall_at_k_util_pct": _util_pct_block(r_corr, rec_ceil),
        "ndcg_at_k": n_at_k,
        "ndcg_at_k_corrected": n_corr,
        "ndcg_at_k_util_pct": _util_pct_block(n_corr, None),
        "map_at_k": m_at_k,
        "map_at_k_corrected": m_corr,
        "map_at_k_util_pct": _util_pct_block(m_corr, None),
        "liftus_map_at_k": lm_at_k,
        "liftus_map_at_k_corrected": lm_corr,
        "precision_lb_at_k": _mean_at_k(rows, "precision_lb", ks),
        "recall_lb_at_k": _mean_at_k(rows, "recall_lb", ks),
        "precision_col_at_k": pc_at_k,
        "precision_col_at_k_ceiling": prec_ceil_col,
        "precision_col_at_k_corrected": pc_corr,
        "precision_col_at_k_util_pct": _util_pct_block(pc_corr, prec_ceil_col),
        "recall_col_at_k": rc_at_k,
        "recall_col_at_k_ceiling": rec_ceil_col,
        "recall_col_at_k_corrected": rc_corr,
        "recall_col_at_k_util_pct": _util_pct_block(rc_corr, rec_ceil_col),
        "ndcg_col_at_k": nc_at_k,
        "ndcg_col_at_k_corrected": nc_corr,
        "ndcg_col_at_k_util_pct": _util_pct_block(nc_corr, None),
        "map_col_at_k": mc_at_k,
        "map_col_at_k_corrected": mc_corr,
        "map_col_at_k_util_pct": _util_pct_block(mc_corr, None),
        "liftus_map_col_at_k": lmc_at_k,
        "liftus_map_col_at_k_corrected": lmc_corr,
        "precision_lb_col_at_k": _mean_at_k(rows, "precision_lb_col", ks),
        "recall_lb_col_at_k": _mean_at_k(rows, "recall_lb_col", ks),
        "n_gt_col_rows_kept": n_gt_col_kept,
        "n_gt_col_rows_dropped_missing_candidate_column": n_gt_col_dropped_missing,
        "wall_clock_seconds": wall_clock_seconds,
    }
