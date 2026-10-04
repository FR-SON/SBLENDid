"""SC token-baseline benchmark: Counter, paper-mode and Blend-plan variants plus --nonnumeric-only."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd
from tqdm import tqdm

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

from src.Semantic.config import SemanticConfig, SemanticOp
from src.Benchmark.self_filter import (
    drop_self_gt, rollup_without_self, without_self)
from src.DBHandler import DBHandler
from src.Operators import Combiners, Seekers
from src.Operators.OperatorBase import Operator
from src.Plan import Plan
from src.Tasks.UnionSearch import UnionSearch

from src.Benchmark.judit.metrics import (  # noqa: E402
    K_RANGE,
    K_COARSE,
    K_FINAL,
    _precision_at_k,
    _recall_at_k,
    _ndcg_at_k,
    _ap_at_k,
    _precision_lb_at_k,
    _recall_lb_at_k,
    gt_density_block,
    pr_ceilings,
    _corrected_block,
    _util_pct_block,
)
from src.Benchmark.nonnumeric_filter import (  # noqa: E402
    intersect_reg,
    load_sho_nonnumeric,
)


def _print_summary(metrics: dict, label: str) -> None:
    print(f"\n=== {label} achieved @ k ∈ {list(K_RANGE)} ===")
    p = metrics["precision_at_k"]
    r = metrics["recall_at_k"]
    nd = metrics["ndcg_at_k"]
    m = metrics["map_at_k"]
    print(f"  {'k':>4}  {'prec':>8}  {'recall':>8}  {'ndcg':>8}  {'map':>8}")
    print(f"  {'-' * 4}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 8}")
    for k in K_RANGE:
        sk = str(k)
        print(f"  {k:>4}  {p[sk]:>8.4f}  {r[sk]:>8.4f}  {nd[sk]:>8.4f}  {m[sk]:>8.4f}")


def _load_id_maps(
    cfg: SemanticConfig, registry_path: Path
) -> tuple[dict[int, str], dict[tuple[str, int], str]]:
    sidecar = pd.read_parquet(cfg.blend_basenames_path)
    int_to_basename = dict(
        zip(
            sidecar["table_int_id"].astype(int).tolist(),
            sidecar["basename"].astype(str).tolist(),
        )
    )
    reg = pd.read_parquet(registry_path)
    table_col_to_name: dict[tuple[str, int], str] = {}
    for tid, cidx, cname in zip(
        reg["table_id"].astype(str).tolist(),
        reg["col_idx"].astype(int).tolist(),
        reg["col_name"].astype(str).tolist(),
    ):
        table_col_to_name[(tid, cidx)] = cname
    return int_to_basename, table_col_to_name


def _build_reg_from_registry(registry_path: Path) -> dict[str, set[str]]:
    reg = pd.read_parquet(registry_path)
    out: dict[str, set[str]] = {}
    for tid, cname in zip(
        reg["table_id"].astype(str).tolist(),
        reg["col_name"].astype(str).tolist(),
    ):
        out.setdefault(tid, set()).add(cname)
    return out


def _sc_search(
    db: DBHandler, values, k_coarse: int
) -> tuple[list[tuple[int, int, int]], bool]:
    """Ranked (TableId, ColumnId, overlap) rows; WHERE/GROUP BY/ORDER BY/LIMIT mirror SingleColumnOverlap."""
    cleaned = db.clean_value_collection(values)
    if not cleaned:
        return [], False
    tokens_sql = db.create_sql_list_str(cleaned)
    sql = f"""
    SELECT TableId, ColumnId, COUNT(DISTINCT CellValue) AS overlap_cnt
    FROM AllTables
    WHERE CellValue IN ({tokens_sql})
    GROUP BY TableId, ColumnId
    ORDER BY overlap_cnt DESC
    LIMIT {k_coarse}
    """
    rows = db.execute_and_fetchall(sql)
    saturated = len(rows) == k_coarse
    return [(int(r[0]), int(r[1]), int(r[2])) for r in rows], saturated


def _run_union_sc(
    cfg: SemanticConfig,
    db: DBHandler,
    reg: dict[str, set[str]],
    int_to_basename: dict[int, str],
    dataset: str,
    k_coarse: int,
    k_final: int,
    *,
    allow_pairs: set[tuple[int, int]] | None = None,
) -> dict:
    t_wall = time.perf_counter()
    base = cfg.dataset.dir()
    queries_df = pd.read_csv(base / "query" / f"{dataset}_union_query.csv")
    gt = pd.read_csv(base / "groundtruth" / f"{dataset}_union_ground_truth.csv")
    gt = gt[gt["query_table"] != "query_table"].reset_index(drop=True)
    gt = drop_self_gt(gt)

    queries = queries_df["query_table"].astype(str).tolist()
    n_q_input = len(queries)
    queries_kept = [qt for qt in queries if qt in reg]
    n_dropped_q = n_q_input - len(queries_kept)

    n_gt_input = len(gt)
    valid_qt = set(queries_kept)
    gt = gt[gt["query_table"].astype(str).isin(valid_qt)]
    gt = gt[gt["candidate_table"].astype(str).isin(reg)]
    n_dropped_c = n_gt_input - len(gt)

    relevant_by_q: dict[str, set[str]] = {}
    for r in gt.itertuples(index=False):
        relevant_by_q.setdefault(str(r.query_table), set()).add(str(r.candidate_table))

    print(
        f"\nSC-UNION  n_q_input={n_q_input}  kept={len(queries_kept)}  "
        f"dropped_q={n_dropped_q}  gt_rows_kept={len(gt)}  "
        f"dropped_gt={n_dropped_c}",
        flush=True,
    )

    csvs_dir = base / "csvs"
    rows: list[dict] = []
    accum = {m: {k: 0.0 for k in K_RANGE} for m in ("p", "r", "n", "m", "plb", "rlb")}
    gt_counts: list[int] = []
    n_saturated = 0
    n_specs_total = 0
    pbar = tqdm(queries_kept, desc="sc-union", unit="q", dynamic_ncols=True)
    for qt in pbar:
        df_q = pd.read_csv(csvs_dir / qt, dtype=str, keep_default_na=False)
        per_col: list[tuple[list[str], list[float]]] = []
        cols_kept = [c for c in df_q.columns if c in reg[qt]]
        col_iter = tqdm(
            cols_kept,
            desc=f"  cols[{qt}]",
            unit="col",
            leave=False,
            dynamic_ncols=True,
        )
        for col_name in col_iter:
            sc_rows, sat = _sc_search(db, df_q[col_name].tolist(), k_coarse)
            if allow_pairs is not None:
                sc_rows = [r for r in sc_rows if (r[0], r[1]) in allow_pairs]
            sc_rows_trunc = sc_rows[:k_final]
            target_tables: list[str] = []
            distances: list[float] = []
            for tid, _cid, cnt in sc_rows_trunc:
                if tid not in int_to_basename:
                    continue
                target_tables.append(int_to_basename[tid])
                distances.append(-float(cnt))
            per_col.append((target_tables, distances))
            n_specs_total += 1
            if sat:
                n_saturated += 1
        rolled = rollup_without_self(per_col, qt, k=k_final)
        ranked = [t for t, _s in rolled]
        relevant = relevant_by_q.get(qt, set())
        gt_counts.append(len(relevant))
        row = {"query_table": qt}
        for k in K_RANGE:
            row[f"precision_at_{k}"] = _precision_at_k(ranked, relevant, k)
            row[f"recall_at_{k}"] = _recall_at_k(ranked, relevant, k)
            row[f"ndcg_at_{k}"] = _ndcg_at_k(ranked, relevant, k)
            row[f"map_at_{k}"] = _ap_at_k(ranked, relevant, k)
            row[f"precision_lb_at_{k}"] = _precision_lb_at_k(ranked, relevant, k)
            row[f"recall_lb_at_{k}"] = _recall_lb_at_k(ranked, relevant, k)
            accum["p"][k] += row[f"precision_at_{k}"]
            accum["r"][k] += row[f"recall_at_{k}"]
            accum["n"][k] += row[f"ndcg_at_{k}"]
            accum["m"][k] += row[f"map_at_{k}"]
            accum["plb"][k] += row[f"precision_lb_at_{k}"]
            accum["rlb"][k] += row[f"recall_lb_at_{k}"]
        rows.append(row)
        pbar.set_postfix(sat=f"{n_saturated}/{n_specs_total}")

    n = len(rows) or 1
    prec_ceil, rec_ceil = pr_ceilings(gt_counts, K_RANGE)
    coarse_saturation_pct = round(n_saturated / max(1, n_specs_total) * 100, 2)
    density = gt_density_block(gt_counts)
    nz = density["n_queries_with_gt"]
    multiplier = (density["n_queries"] / nz) if nz > 0 else None

    p_at_k = {str(k): accum["p"][k] / n for k in K_RANGE}
    r_at_k = {str(k): accum["r"][k] / n for k in K_RANGE}
    n_at_k = {str(k): accum["n"][k] / n for k in K_RANGE}
    m_at_k = {str(k): accum["m"][k] / n for k in K_RANGE}
    p_corr = _corrected_block(p_at_k, multiplier)
    r_corr = _corrected_block(r_at_k, multiplier)
    n_corr = _corrected_block(n_at_k, multiplier)
    m_corr = _corrected_block(m_at_k, multiplier)

    return {
        "task": "union",
        "method": "sc_per_col_rollup",
        "n_queries_input": n_q_input,
        "n_queries_kept": len(rows),
        "n_gt_rows_input": n_gt_input,
        "n_gt_rows_kept": len(gt),
        "n_dropped_missing_query": n_dropped_q,
        "n_dropped_missing_candidate": n_dropped_c,
        "gt_density": density,
        "k_coarse_used": k_coarse,
        "k_final_used": k_final,
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
        "precision_lb_at_k": {str(k): accum["plb"][k] / n for k in K_RANGE},
        "recall_lb_at_k": {str(k): accum["rlb"][k] / n for k in K_RANGE},
        "wall_clock_seconds": round(time.perf_counter() - t_wall, 6),
    }


def _run_union_blend_plan(
    cfg: SemanticConfig,
    reg: dict[str, set[str]],
    int_to_basename: dict[int, str],
    dataset: str,
    k_final: int,
) -> dict:
    t_wall = time.perf_counter()
    base = cfg.dataset.dir()
    queries_df = pd.read_csv(base / "query" / f"{dataset}_union_query.csv")
    gt = pd.read_csv(base / "groundtruth" / f"{dataset}_union_ground_truth.csv")
    gt = gt[gt["query_table"] != "query_table"].reset_index(drop=True)
    gt = drop_self_gt(gt)

    queries = queries_df["query_table"].astype(str).tolist()
    n_q_input = len(queries)
    queries_kept = [qt for qt in queries if qt in reg]

    valid_qt = set(queries_kept)
    gt = gt[gt["query_table"].astype(str).isin(valid_qt)]
    gt = gt[gt["candidate_table"].astype(str).isin(reg)]
    relevant_by_q: dict[str, set[str]] = {}
    for r in gt.itertuples(index=False):
        relevant_by_q.setdefault(str(r.query_table), set()).add(str(r.candidate_table))

    print(
        f"\nBLEND-PLAN-UNION  kept={len(queries_kept)} gt_rows_kept={len(gt)}",
        flush=True,
    )

    csvs_dir = base / "csvs"
    p_sum = 0.0
    r_sum = 0.0
    plb_sum = 0.0
    rlb_sum = 0.0
    gt_counts: list[int] = []
    for qt in tqdm(queries_kept, desc="blend-plan", unit="q", dynamic_ncols=True):
        df_q = pd.read_csv(csvs_dir / qt, dtype=str, keep_default_na=False)
        df_q = df_q[[c for c in df_q.columns if c in reg[qt]]]
        if df_q.shape[1] == 0:
            ranked: list[str] = []
        else:
            plan = UnionSearch(df_q, k=k_final)
            tids = plan.run()
            ranked = [
                int_to_basename[int(t)]
                for t in tids
                if int(t) in int_to_basename
            ]
        ranked = without_self(ranked, qt)
        relevant = relevant_by_q.get(qt, set())
        gt_counts.append(len(relevant))
        p_sum += _precision_at_k(ranked, relevant, k_final)
        r_sum += _recall_at_k(ranked, relevant, k_final)
        plb_sum += _precision_lb_at_k(ranked, relevant, k_final)
        rlb_sum += _recall_lb_at_k(ranked, relevant, k_final)

    n = len(queries_kept) or 1
    prec_ceil, rec_ceil = pr_ceilings(gt_counts, [k_final])
    density = gt_density_block(gt_counts)
    nz = density["n_queries_with_gt"]
    multiplier = (density["n_queries"] / nz) if nz > 0 else None

    p_at_k = {str(k_final): p_sum / n}
    r_at_k = {str(k_final): r_sum / n}
    p_corr = _corrected_block(p_at_k, multiplier)
    r_corr = _corrected_block(r_at_k, multiplier)

    return {
        "task": "union",
        "method": "blend_plan_unionsearch",
        "n_queries_input": n_q_input,
        "n_queries_kept": n,
        "gt_density": density,
        "k_final_used": k_final,
        "precision_at_k": p_at_k,
        "precision_at_k_ceiling": prec_ceil,
        "precision_at_k_corrected": p_corr,
        "precision_at_k_util_pct": _util_pct_block(p_corr, prec_ceil),
        "recall_at_k": r_at_k,
        "recall_at_k_ceiling": rec_ceil,
        "recall_at_k_corrected": r_corr,
        "recall_at_k_util_pct": _util_pct_block(r_corr, rec_ceil),
        "precision_lb_at_k": {str(k_final): plb_sum / n},
        "recall_lb_at_k": {str(k_final): rlb_sum / n},
        "wall_clock_seconds": round(time.perf_counter() - t_wall, 6),
    }


def _run_union_sc_counter(
    cfg: SemanticConfig,
    reg: dict[str, set[str]],
    int_to_basename: dict[int, str],
    dataset: str,
    k_coarse: int,
    k_final: int,
) -> dict:
    t_wall = time.perf_counter()
    base = cfg.dataset.dir()
    queries_df = pd.read_csv(base / "query" / f"{dataset}_union_query.csv")
    gt = pd.read_csv(base / "groundtruth" / f"{dataset}_union_ground_truth.csv")
    gt = gt[gt["query_table"] != "query_table"].reset_index(drop=True)
    gt = drop_self_gt(gt)

    queries = queries_df["query_table"].astype(str).tolist()
    n_q_input = len(queries)
    queries_kept = [qt for qt in queries if qt in reg]
    n_dropped_q = n_q_input - len(queries_kept)

    n_gt_input = len(gt)
    valid_qt = set(queries_kept)
    gt = gt[gt["query_table"].astype(str).isin(valid_qt)]
    gt = gt[gt["candidate_table"].astype(str).isin(reg)]
    n_dropped_c = n_gt_input - len(gt)

    relevant_by_q: dict[str, set[str]] = {}
    for r in gt.itertuples(index=False):
        relevant_by_q.setdefault(str(r.query_table), set()).add(str(r.candidate_table))

    print(
        f"\nSC-COUNTER  n_q_input={n_q_input}  kept={len(queries_kept)}  "
        f"dropped_q={n_dropped_q}  gt_rows_kept={len(gt)}  "
        f"dropped_gt={n_dropped_c}",
        flush=True,
    )

    csvs_dir = base / "csvs"
    rows: list[dict] = []
    accum = {m: {k: 0.0 for k in K_RANGE} for m in ("p", "r", "n", "m", "plb", "rlb")}
    gt_counts: list[int] = []
    for qt in tqdm(queries_kept, desc="sc-counter", unit="q", dynamic_ncols=True):
        df_q = pd.read_csv(csvs_dir / qt, dtype=str, keep_default_na=False)
        df_q = df_q[[c for c in df_q.columns if c in reg[qt]]]
        if df_q.shape[1] == 0:
            ranked: list[str] = []
        else:
            plan = Plan()
            col_names = list(df_q.columns)
            for col_name in col_names:
                plan.add(col_name, Seekers.SC(df_q[col_name], k=k_coarse))
            plan.add("counter", Combiners.Counter(k=k_final), inputs=col_names)
            tids = plan.run()
            ranked = [
                int_to_basename[int(t)]
                for t in tids
                if int(t) in int_to_basename
            ]
        ranked = without_self(ranked, qt)
        relevant = relevant_by_q.get(qt, set())
        gt_counts.append(len(relevant))
        row = {"query_table": qt}
        for k in K_RANGE:
            row[f"precision_at_{k}"] = _precision_at_k(ranked, relevant, k)
            row[f"recall_at_{k}"] = _recall_at_k(ranked, relevant, k)
            row[f"ndcg_at_{k}"] = _ndcg_at_k(ranked, relevant, k)
            row[f"map_at_{k}"] = _ap_at_k(ranked, relevant, k)
            row[f"precision_lb_at_{k}"] = _precision_lb_at_k(ranked, relevant, k)
            row[f"recall_lb_at_{k}"] = _recall_lb_at_k(ranked, relevant, k)
            accum["p"][k] += row[f"precision_at_{k}"]
            accum["r"][k] += row[f"recall_at_{k}"]
            accum["n"][k] += row[f"ndcg_at_{k}"]
            accum["m"][k] += row[f"map_at_{k}"]
            accum["plb"][k] += row[f"precision_lb_at_{k}"]
            accum["rlb"][k] += row[f"recall_lb_at_{k}"]
        rows.append(row)

    n = len(rows) or 1
    prec_ceil, rec_ceil = pr_ceilings(gt_counts, K_RANGE)
    density = gt_density_block(gt_counts)
    nz = density["n_queries_with_gt"]
    multiplier = (density["n_queries"] / nz) if nz > 0 else None

    p_at_k = {str(k): accum["p"][k] / n for k in K_RANGE}
    r_at_k = {str(k): accum["r"][k] / n for k in K_RANGE}
    n_at_k = {str(k): accum["n"][k] / n for k in K_RANGE}
    m_at_k = {str(k): accum["m"][k] / n for k in K_RANGE}
    p_corr = _corrected_block(p_at_k, multiplier)
    r_corr = _corrected_block(r_at_k, multiplier)
    n_corr = _corrected_block(n_at_k, multiplier)
    m_corr = _corrected_block(m_at_k, multiplier)

    return {
        "task": "union",
        "method": "sc_per_col_counter_combiner",
        "n_queries_input": n_q_input,
        "n_queries_kept": len(rows),
        "n_gt_rows_input": n_gt_input,
        "n_gt_rows_kept": len(gt),
        "n_dropped_missing_query": n_dropped_q,
        "n_dropped_missing_candidate": n_dropped_c,
        "gt_density": density,
        "k_coarse_used": k_coarse,
        "k_final_used": k_final,
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
        "precision_lb_at_k": {str(k): accum["plb"][k] / n for k in K_RANGE},
        "recall_lb_at_k": {str(k): accum["rlb"][k] / n for k in K_RANGE},
        "wall_clock_seconds": round(time.perf_counter() - t_wall, 6),
    }


def _run_union_sc_counter_paper_sweep(
    cfg: SemanticConfig,
    db: DBHandler,
    reg: dict[str, set[str]],
    int_to_basename: dict[int, str],
    dataset: str,
) -> dict:
    from collections import Counter as _PyCounter

    t_wall = time.perf_counter()
    base = cfg.dataset.dir()
    queries_df = pd.read_csv(base / "query" / f"{dataset}_union_query.csv")
    gt = pd.read_csv(base / "groundtruth" / f"{dataset}_union_ground_truth.csv")
    gt = gt[gt["query_table"] != "query_table"].reset_index(drop=True)
    gt = drop_self_gt(gt)

    queries = queries_df["query_table"].astype(str).tolist()
    n_q_input = len(queries)
    queries_kept = [qt for qt in queries if qt in reg]
    n_dropped_q = n_q_input - len(queries_kept)

    n_gt_input = len(gt)
    valid_qt = set(queries_kept)
    gt = gt[gt["query_table"].astype(str).isin(valid_qt)]
    gt = gt[gt["candidate_table"].astype(str).isin(reg)]
    n_dropped_c = n_gt_input - len(gt)

    relevant_by_q: dict[str, set[str]] = {}
    for r in gt.itertuples(index=False):
        relevant_by_q.setdefault(str(r.query_table), set()).add(str(r.candidate_table))

    max_kc = 10 * max(K_RANGE)
    print(
        f"\nSC-COUNTER-PAPER  n_q_input={n_q_input}  kept={len(queries_kept)}  "
        f"dropped_q={n_dropped_q}  gt_rows_kept={len(gt)}  "
        f"dropped_gt={n_dropped_c}  SC-cache k_coarse={max_kc}",
        flush=True,
    )

    csvs_dir = base / "csvs"
    accum = {m: {k: 0.0 for k in K_RANGE} for m in ("p", "r", "n", "m", "plb", "rlb")}
    gt_counts: list[int] = []
    n_saturated_at_max = 0
    basename_to_int = {v: k for k, v in int_to_basename.items()}
    for qt in tqdm(queries_kept, desc="sc-counter-paper", unit="q", dynamic_ncols=True):
        df_q = pd.read_csv(csvs_dir / qt, dtype=str, keep_default_na=False)
        cols_kept = [c for c in df_q.columns if c in reg[qt]]
        col_sc: dict[str, list[tuple[int, int]]] = {}
        for col_name in cols_kept:
            sc_rows, sat = _sc_search(db, df_q[col_name].tolist(), max_kc)
            if sat:
                n_saturated_at_max += 1
            col_sc[col_name] = [(tid, cid) for tid, cid, _cnt in sc_rows]
        rel = relevant_by_q.get(qt, set())
        gt_counts.append(len(rel))
        self_tid = basename_to_int.get(qt)

        for k_target in K_RANGE:
            kc = 10 * k_target
            votes: _PyCounter[int] = _PyCounter()
            for col_name in cols_kept:
                for tid, _cid in col_sc[col_name][:kc]:
                    votes[tid] += 1
            if self_tid is not None:
                votes.pop(self_tid, None)
            top = votes.most_common(k_target)
            ranked = [
                int_to_basename[tid] for tid, _ in top if tid in int_to_basename
            ]
            accum["p"][k_target] += _precision_at_k(ranked, rel, k_target)
            accum["r"][k_target] += _recall_at_k(ranked, rel, k_target)
            accum["n"][k_target] += _ndcg_at_k(ranked, rel, k_target)
            accum["m"][k_target] += _ap_at_k(ranked, rel, k_target)
            accum["plb"][k_target] += _precision_lb_at_k(ranked, rel, k_target)
            accum["rlb"][k_target] += _recall_lb_at_k(ranked, rel, k_target)

    n = len(queries_kept) or 1
    prec_ceil, rec_ceil = pr_ceilings(gt_counts, K_RANGE)
    density = gt_density_block(gt_counts)
    nz = density["n_queries_with_gt"]
    multiplier = (density["n_queries"] / nz) if nz > 0 else None

    p_at_k = {str(k): accum["p"][k] / n for k in K_RANGE}
    r_at_k = {str(k): accum["r"][k] / n for k in K_RANGE}
    n_at_k = {str(k): accum["n"][k] / n for k in K_RANGE}
    m_at_k = {str(k): accum["m"][k] / n for k in K_RANGE}
    p_corr = _corrected_block(p_at_k, multiplier)
    r_corr = _corrected_block(r_at_k, multiplier)
    n_corr = _corrected_block(n_at_k, multiplier)
    m_corr = _corrected_block(m_at_k, multiplier)

    return {
        "task": "union",
        "method": "sc_per_col_counter_combiner_paper_sweep",
        "n_queries_input": n_q_input,
        "n_queries_kept": n,
        "n_gt_rows_input": n_gt_input,
        "n_gt_rows_kept": len(gt),
        "n_dropped_missing_query": n_dropped_q,
        "n_dropped_missing_candidate": n_dropped_c,
        "gt_density": density,
        "k_coarse_used": "10*k per evaluation k",
        "k_final_used": "k per evaluation k",
        "max_sc_kc_cached": max_kc,
        "n_queries_saturated_at_max_kc": n_saturated_at_max,
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
        "precision_lb_at_k": {str(k): accum["plb"][k] / n for k in K_RANGE},
        "recall_lb_at_k": {str(k): accum["rlb"][k] / n for k in K_RANGE},
        "wall_clock_seconds": round(time.perf_counter() - t_wall, 6),
    }


def _run_join_sc(
    cfg: SemanticConfig,
    db: DBHandler,
    reg: dict[str, set[str]],
    int_to_basename: dict[int, str],
    table_col_to_name: dict[tuple[str, int], str],
    dataset: str,
    k_coarse: int,
    k_final: int,
    *,
    allow_pairs: set[tuple[int, int]] | None = None,
) -> dict:
    t_wall = time.perf_counter()
    base = cfg.dataset.dir()
    queries_df = pd.read_csv(base / "query" / f"{dataset}_join_query.csv")
    gt = pd.read_csv(base / "groundtruth" / f"{dataset}_join_ground_truth.csv")
    gt = drop_self_gt(gt)

    pairs = [
        (str(qt), str(qc))
        for qt, qc in zip(queries_df["query_table"], queries_df["query_column"])
    ]
    n_q_input = len(pairs)
    pairs_kept = [(qt, qc) for qt, qc in pairs if qt in reg and qc in reg[qt]]
    n_dropped_q = n_q_input - len(pairs_kept)

    n_gt_input = len(gt)
    valid_qt = {qt for qt, _ in pairs_kept}
    gt = gt[gt["query_table"].astype(str).isin(valid_qt)]
    gt = gt[gt["candidate_table"].astype(str).isin(reg)]
    n_dropped_c = n_gt_input - len(gt)

    relevant_by_q: dict[tuple[str, str], set[str]] = {}
    for r in gt.itertuples(index=False):
        relevant_by_q.setdefault(
            (str(r.query_table), str(r.query_column)), set()
        ).add(str(r.candidate_table))

    if len(gt):
        mask = [
            c in reg.get(t, set())
            for t, c in zip(
                gt["candidate_table"].astype(str),
                gt["candidate_column"].astype(str),
            )
        ]
        gt_col = gt[mask]
    else:
        gt_col = gt
    n_gt_col_kept = len(gt_col)
    n_gt_col_dropped_missing = len(gt) - n_gt_col_kept

    relevant_cols_by_q: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for r in gt_col.itertuples(index=False):
        relevant_cols_by_q.setdefault(
            (str(r.query_table), str(r.query_column)), set()
        ).add((str(r.candidate_table), str(r.candidate_column)))

    print(
        f"\nSC-JOIN  n_q_input={n_q_input}  kept={len(pairs_kept)}  "
        f"dropped_q={n_dropped_q}  gt_rows_kept={len(gt)}  "
        f"dropped_gt={n_dropped_c}",
        flush=True,
    )

    csvs_dir = base / "csvs"
    rows: list[dict] = []
    accum = {m: {k: 0.0 for k in K_RANGE} for m in ("p", "r", "n", "m", "plb", "rlb")}
    accum_col = {
        m: {k: 0.0 for k in K_RANGE} for m in ("p", "r", "n", "m", "plb", "rlb")
    }
    gt_counts: list[int] = []
    gt_counts_col: list[int] = []
    n_saturated = 0
    n_unmapped_col_rows = 0
    pbar = tqdm(pairs_kept, desc="sc-join", unit="pair", dynamic_ncols=True)
    for qt, qc in pbar:
        col_vals = pd.read_csv(
            csvs_dir / qt,
            dtype=str,
            keep_default_na=False,
            usecols=[qc],
        )[qc].tolist()
        sc_rows, sat = _sc_search(db, col_vals, k_coarse)
        if allow_pairs is not None:
            sc_rows = [r for r in sc_rows if (r[0], r[1]) in allow_pairs]
        sc_rows_trunc = sc_rows[:k_final]
        if sat:
            n_saturated += 1

        target_tables: list[str] = []
        distances: list[float] = []
        retrieved_cols: list[tuple[str, str]] = []
        for tid, cid, cnt in sc_rows_trunc:
            if tid not in int_to_basename:
                continue
            tname = int_to_basename[tid]
            target_tables.append(tname)
            distances.append(-float(cnt))
            cname = table_col_to_name.get((tname, cid))
            if cname is None:
                n_unmapped_col_rows += 1
                continue
            retrieved_cols.append((tname, cname))

        retrieved_cols = without_self(retrieved_cols, qt)
        rolled = rollup_without_self([(target_tables, distances)], qt, k=k_final)
        ranked = [t for t, _s in rolled]

        relevant = relevant_by_q.get((qt, qc), set())
        relevant_cols = relevant_cols_by_q.get((qt, qc), set())
        gt_counts.append(len(relevant))
        gt_counts_col.append(len(relevant_cols))

        row = {"query_table": qt, "query_column": qc}
        for k in K_RANGE:
            row[f"precision_at_{k}"] = _precision_at_k(ranked, relevant, k)
            row[f"recall_at_{k}"] = _recall_at_k(ranked, relevant, k)
            row[f"ndcg_at_{k}"] = _ndcg_at_k(ranked, relevant, k)
            row[f"map_at_{k}"] = _ap_at_k(ranked, relevant, k)
            row[f"precision_lb_at_{k}"] = _precision_lb_at_k(ranked, relevant, k)
            row[f"recall_lb_at_{k}"] = _recall_lb_at_k(ranked, relevant, k)
            accum["p"][k] += row[f"precision_at_{k}"]
            accum["r"][k] += row[f"recall_at_{k}"]
            accum["n"][k] += row[f"ndcg_at_{k}"]
            accum["m"][k] += row[f"map_at_{k}"]
            accum["plb"][k] += row[f"precision_lb_at_{k}"]
            accum["rlb"][k] += row[f"recall_lb_at_{k}"]

            row[f"precision_col_at_{k}"] = _precision_at_k(
                retrieved_cols, relevant_cols, k
            )
            row[f"recall_col_at_{k}"] = _recall_at_k(retrieved_cols, relevant_cols, k)
            row[f"ndcg_col_at_{k}"] = _ndcg_at_k(retrieved_cols, relevant_cols, k)
            row[f"map_col_at_{k}"] = _ap_at_k(retrieved_cols, relevant_cols, k)
            row[f"precision_lb_col_at_{k}"] = _precision_lb_at_k(
                retrieved_cols, relevant_cols, k
            )
            row[f"recall_lb_col_at_{k}"] = _recall_lb_at_k(
                retrieved_cols, relevant_cols, k
            )
            accum_col["p"][k] += row[f"precision_col_at_{k}"]
            accum_col["r"][k] += row[f"recall_col_at_{k}"]
            accum_col["n"][k] += row[f"ndcg_col_at_{k}"]
            accum_col["m"][k] += row[f"map_col_at_{k}"]
            accum_col["plb"][k] += row[f"precision_lb_col_at_{k}"]
            accum_col["rlb"][k] += row[f"recall_lb_col_at_{k}"]
        rows.append(row)
        pbar.set_postfix(sat=f"{n_saturated}/{len(rows)}")

    n = len(rows) or 1
    prec_ceil, rec_ceil = pr_ceilings(gt_counts, K_RANGE)
    prec_ceil_col, rec_ceil_col = pr_ceilings(gt_counts_col, K_RANGE)
    coarse_saturation_pct = round(n_saturated / max(1, n) * 100, 2)
    density = gt_density_block(gt_counts)
    density_col = gt_density_block(gt_counts_col)
    nz = density["n_queries_with_gt"]
    multiplier = (density["n_queries"] / nz) if nz > 0 else None
    nz_col = density_col["n_queries_with_gt"]
    multiplier_col = (
        (density_col["n_queries"] / nz_col) if nz_col > 0 else None
    )

    p_at_k = {str(k): accum["p"][k] / n for k in K_RANGE}
    r_at_k = {str(k): accum["r"][k] / n for k in K_RANGE}
    n_at_k = {str(k): accum["n"][k] / n for k in K_RANGE}
    m_at_k = {str(k): accum["m"][k] / n for k in K_RANGE}
    pc_at_k = {str(k): accum_col["p"][k] / n for k in K_RANGE}
    rc_at_k = {str(k): accum_col["r"][k] / n for k in K_RANGE}
    nc_at_k = {str(k): accum_col["n"][k] / n for k in K_RANGE}
    mc_at_k = {str(k): accum_col["m"][k] / n for k in K_RANGE}
    p_corr = _corrected_block(p_at_k, multiplier)
    r_corr = _corrected_block(r_at_k, multiplier)
    n_corr = _corrected_block(n_at_k, multiplier)
    m_corr = _corrected_block(m_at_k, multiplier)
    pc_corr = _corrected_block(pc_at_k, multiplier_col)
    rc_corr = _corrected_block(rc_at_k, multiplier_col)
    nc_corr = _corrected_block(nc_at_k, multiplier_col)
    mc_corr = _corrected_block(mc_at_k, multiplier_col)

    return {
        "task": "join",
        "method": "sc",
        "n_queries_input": n_q_input,
        "n_queries_kept": len(rows),
        "n_gt_rows_input": n_gt_input,
        "n_gt_rows_kept": len(gt),
        "n_dropped_missing_query": n_dropped_q,
        "n_dropped_missing_candidate": n_dropped_c,
        "gt_density": density,
        "gt_density_col": density_col,
        "k_coarse_used": k_coarse,
        "k_final_used": k_final,
        "coarse_saturation_pct": coarse_saturation_pct,
        "n_unmapped_col_rows": n_unmapped_col_rows,
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
        "precision_lb_at_k": {str(k): accum["plb"][k] / n for k in K_RANGE},
        "recall_lb_at_k": {str(k): accum["rlb"][k] / n for k in K_RANGE},
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
        "precision_lb_col_at_k": {
            str(k): accum_col["plb"][k] / n for k in K_RANGE
        },
        "recall_lb_col_at_k": {
            str(k): accum_col["rlb"][k] / n for k in K_RANGE
        },
        "n_gt_col_rows_kept": n_gt_col_kept,
        "n_gt_col_rows_dropped_missing_candidate_column": n_gt_col_dropped_missing,
        "wall_clock_seconds": round(time.perf_counter() - t_wall, 6),
    }


def _run_join_sc_paper_sweep(
    cfg: SemanticConfig,
    db: DBHandler,
    reg: dict[str, set[str]],
    int_to_basename: dict[int, str],
    table_col_to_name: dict[tuple[str, int], str],
    dataset: str,
) -> dict:
    t_wall = time.perf_counter()
    base = cfg.dataset.dir()
    queries_df = pd.read_csv(base / "query" / f"{dataset}_join_query.csv")
    gt = pd.read_csv(base / "groundtruth" / f"{dataset}_join_ground_truth.csv")
    gt = drop_self_gt(gt)

    pairs = [
        (str(qt), str(qc))
        for qt, qc in zip(queries_df["query_table"], queries_df["query_column"])
    ]
    n_q_input = len(pairs)
    pairs_kept = [(qt, qc) for qt, qc in pairs if qt in reg and qc in reg[qt]]
    n_dropped_q = n_q_input - len(pairs_kept)

    n_gt_input = len(gt)
    valid_qt = {qt for qt, _ in pairs_kept}
    gt = gt[gt["query_table"].astype(str).isin(valid_qt)]
    gt = gt[gt["candidate_table"].astype(str).isin(reg)]
    n_dropped_c = n_gt_input - len(gt)

    relevant_by_q: dict[tuple[str, str], set[str]] = {}
    for r in gt.itertuples(index=False):
        relevant_by_q.setdefault(
            (str(r.query_table), str(r.query_column)), set()
        ).add(str(r.candidate_table))

    if len(gt):
        mask = [
            c in reg.get(t, set())
            for t, c in zip(
                gt["candidate_table"].astype(str),
                gt["candidate_column"].astype(str),
            )
        ]
        gt_col = gt[mask]
    else:
        gt_col = gt
    n_gt_col_kept = len(gt_col)
    n_gt_col_dropped_missing = len(gt) - n_gt_col_kept

    relevant_cols_by_q: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for r in gt_col.itertuples(index=False):
        relevant_cols_by_q.setdefault(
            (str(r.query_table), str(r.query_column)), set()
        ).add((str(r.candidate_table), str(r.candidate_column)))

    max_kc = 10 * max(K_RANGE)
    print(
        f"\nSC-JOIN-PAPER  n_q_input={n_q_input}  kept={len(pairs_kept)}  "
        f"dropped_q={n_dropped_q}  gt_rows_kept={len(gt)}  "
        f"dropped_gt={n_dropped_c}  SC-cache k_coarse={max_kc}",
        flush=True,
    )

    csvs_dir = base / "csvs"
    accum = {m: {k: 0.0 for k in K_RANGE} for m in ("p", "r", "n", "m", "plb", "rlb")}
    accum_col = {m: {k: 0.0 for k in K_RANGE} for m in ("p", "r", "n", "m", "plb", "rlb")}
    gt_counts: list[int] = []
    gt_counts_col: list[int] = []
    n_saturated_at_max = 0
    n_unmapped_col_rows = 0
    n_self_filtered = 0
    pbar = tqdm(pairs_kept, desc="sc-join-paper", unit="pair", dynamic_ncols=True)
    for qt, qc in pbar:
        col_vals = pd.read_csv(
            csvs_dir / qt,
            dtype=str,
            keep_default_na=False,
            usecols=[qc],
        )[qc].tolist()
        sc_rows, sat = _sc_search(db, col_vals, max_kc)
        if sat:
            n_saturated_at_max += 1

        cached_tables: list[str] = []
        cached_cols: list[tuple[str, str]] = []
        for tid, cid, _cnt in sc_rows:
            if tid not in int_to_basename:
                continue
            tname = int_to_basename[tid]
            if tname == qt:
                n_self_filtered += 1
                continue
            cached_tables.append(tname)
            cname = table_col_to_name.get((tname, cid))
            if cname is None:
                n_unmapped_col_rows += 1
                continue
            cached_cols.append((tname, cname))

        relevant = relevant_by_q.get((qt, qc), set())
        relevant_cols = relevant_cols_by_q.get((qt, qc), set())
        gt_counts.append(len(relevant))
        gt_counts_col.append(len(relevant_cols))

        for k_target in K_RANGE:
            kc = 10 * k_target
            seen: set[str] = set()
            ranked_tables: list[str] = []
            for tname in cached_tables[:kc]:
                if tname not in seen:
                    seen.add(tname)
                    ranked_tables.append(tname)
                    if len(ranked_tables) == k_target:
                        break
            ranked_cols = cached_cols[:k_target]

            accum["p"][k_target] += _precision_at_k(ranked_tables, relevant, k_target)
            accum["r"][k_target] += _recall_at_k(ranked_tables, relevant, k_target)
            accum["n"][k_target] += _ndcg_at_k(ranked_tables, relevant, k_target)
            accum["m"][k_target] += _ap_at_k(ranked_tables, relevant, k_target)
            accum["plb"][k_target] += _precision_lb_at_k(
                ranked_tables, relevant, k_target
            )
            accum["rlb"][k_target] += _recall_lb_at_k(ranked_tables, relevant, k_target)

            accum_col["p"][k_target] += _precision_at_k(
                ranked_cols, relevant_cols, k_target
            )
            accum_col["r"][k_target] += _recall_at_k(
                ranked_cols, relevant_cols, k_target
            )
            accum_col["n"][k_target] += _ndcg_at_k(
                ranked_cols, relevant_cols, k_target
            )
            accum_col["m"][k_target] += _ap_at_k(ranked_cols, relevant_cols, k_target)
            accum_col["plb"][k_target] += _precision_lb_at_k(
                ranked_cols, relevant_cols, k_target
            )
            accum_col["rlb"][k_target] += _recall_lb_at_k(
                ranked_cols, relevant_cols, k_target
            )

    n = len(pairs_kept) or 1
    prec_ceil, rec_ceil = pr_ceilings(gt_counts, K_RANGE)
    prec_ceil_col, rec_ceil_col = pr_ceilings(gt_counts_col, K_RANGE)
    coarse_saturation_pct = round(n_saturated_at_max / max(1, n) * 100, 2)
    density = gt_density_block(gt_counts)
    density_col = gt_density_block(gt_counts_col)
    nz = density["n_queries_with_gt"]
    multiplier = (density["n_queries"] / nz) if nz > 0 else None
    nz_col = density_col["n_queries_with_gt"]
    multiplier_col = (
        (density_col["n_queries"] / nz_col) if nz_col > 0 else None
    )

    p_at_k = {str(k): accum["p"][k] / n for k in K_RANGE}
    r_at_k = {str(k): accum["r"][k] / n for k in K_RANGE}
    n_at_k = {str(k): accum["n"][k] / n for k in K_RANGE}
    m_at_k = {str(k): accum["m"][k] / n for k in K_RANGE}
    pc_at_k = {str(k): accum_col["p"][k] / n for k in K_RANGE}
    rc_at_k = {str(k): accum_col["r"][k] / n for k in K_RANGE}
    nc_at_k = {str(k): accum_col["n"][k] / n for k in K_RANGE}
    mc_at_k = {str(k): accum_col["m"][k] / n for k in K_RANGE}
    p_corr = _corrected_block(p_at_k, multiplier)
    r_corr = _corrected_block(r_at_k, multiplier)
    n_corr = _corrected_block(n_at_k, multiplier)
    m_corr = _corrected_block(m_at_k, multiplier)
    pc_corr = _corrected_block(pc_at_k, multiplier_col)
    rc_corr = _corrected_block(rc_at_k, multiplier_col)
    nc_corr = _corrected_block(nc_at_k, multiplier_col)
    mc_corr = _corrected_block(mc_at_k, multiplier_col)

    return {
        "task": "join",
        "method": "sc_join_paper_sweep",
        "n_queries_input": n_q_input,
        "n_queries_kept": n,
        "n_gt_rows_input": n_gt_input,
        "n_gt_rows_kept": len(gt),
        "n_dropped_missing_query": n_dropped_q,
        "n_dropped_missing_candidate": n_dropped_c,
        "gt_density": density,
        "gt_density_col": density_col,
        "k_coarse_used": "10*k per evaluation k",
        "k_final_used": "k per evaluation k",
        "max_sc_kc_cached": max_kc,
        "n_queries_saturated_at_max_kc": n_saturated_at_max,
        "n_unmapped_col_rows": n_unmapped_col_rows,
        "n_self_filtered_rows": n_self_filtered,
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
        "precision_lb_at_k": {str(k): accum["plb"][k] / n for k in K_RANGE},
        "recall_lb_at_k": {str(k): accum["rlb"][k] / n for k in K_RANGE},
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
        "precision_lb_col_at_k": {
            str(k): accum_col["plb"][k] / n for k in K_RANGE
        },
        "recall_lb_col_at_k": {
            str(k): accum_col["rlb"][k] / n for k in K_RANGE
        },
        "n_gt_col_rows_kept": n_gt_col_kept,
        "n_gt_col_rows_dropped_missing_candidate_column": n_gt_col_dropped_missing,
        "wall_clock_seconds": round(time.perf_counter() - t_wall, 6),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=["union", "join", "all"], default="all")
    ap.add_argument(
        "--dataset",
        default="opendata",
        help="benchmark file prefix under {query,groundtruth}/, "
        "e.g. 'opendata' → opendata_union_query.csv",
    )
    ap.add_argument("--k-coarse", type=int, default=K_COARSE)
    ap.add_argument("--k-final", type=int, default=K_FINAL)
    ap.add_argument(
        "--skip-rollup",
        action="store_true",
        help="omit the SC + JUDIT rollup baseline (per-col SC + decay-vote rollup)",
    )
    ap.add_argument(
        "--skip-counter",
        action="store_true",
        help="omit the SC + Counter combiner baseline (pure-Blend rank-aware Plan)",
    )
    ap.add_argument(
        "--skip-blend-plan",
        action="store_true",
        help="omit the stock Tasks.UnionSearch baseline (SC + Union combiner, no ranking)",
    )
    ap.add_argument(
        "--paper-mode",
        action="store_true",
        help="union: replace the single-(k_coarse,k_final) Counter run with a "
        "per-k sweep (SC k=10·k, Counter k=k for each k ∈ K_RANGE). join: "
        "replace the single-(k_coarse,k_final) SC run with a per-k sweep "
        "(SC k=10·k, dedup to k unique TableIds per k ∈ K_RANGE). Matches "
        "Blend paper §VII.A / §IV.A.1 methodology; one number per k from "
        "its own optimal-ratio sub-run. Ignores --k-coarse/--k-final for "
        "the affected variants; other variants still use those flags.",
    )
    ap.add_argument(
        "--nonnumeric-only",
        action="store_true",
        help="restrict query cols, candidate pool, and GT to the non-numeric "
        "columns SHO holds (its codes.parquet) — apples-to-apples vs the SHO "
        "bench. Runs only the per-col rollup (union) + SC join primary paths "
        "(counter/paper/blend-plan variants skipped). Requires the SHO index; "
        "off by default (unchanged full-lake run).",
    )
    ap.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="write combined JSON to this path",
    )
    args = ap.parse_args()

    cfg = SemanticConfig.load()
    print(f"dataset      : {cfg.dataset.name}")
    print(f"k_coarse     : {args.k_coarse}")
    print(f"k_final      : {args.k_final}")
    print(f"K_RANGE      : {K_RANGE}")

    allow: dict[str, set[str]] | None = None
    allow_pairs: set[tuple[int, int]] | None = None
    if args.nonnumeric_only:
        u = load_sho_nonnumeric(cfg)
        allow, allow_pairs = u.by_name, u.pairs_int
        print(
            f"nonnumeric   : SHO universe |tables|={len(allow)}  "
            f"|(tid,cid)|={len(allow_pairs)}  "
            f"(counter/paper/blend-plan variants skipped)",
            flush=True,
        )

    def _load_for_op(op: SemanticOp):
        opc = cfg.operator(op)
        rp = cfg.approach_dir(opc.approach, opc.index_name) / "index" / "registry.parquet"
        i2b, tc2n = _load_id_maps(cfg, rp)
        r = _build_reg_from_registry(rp)
        if allow is not None:
            r = intersect_reg(r, allow)
        print(
            f"\nregistry [{op.name}]: {rp.relative_to(cfg.dataset.dir())}  "
            f"|tables|={len(r)}  |int→basename|={len(i2b)}"
            + ("  [nonnumeric-only]" if allow is not None else ""),
            flush=True,
        )
        return r, i2b, tc2n

    db = Operator.DB

    out: dict = {}

    if args.task in ("union", "all"):
        reg, int_to_basename, table_col_to_name = _load_for_op(SemanticOp.SU)
        if not args.skip_rollup:
            m = _run_union_sc(
                cfg, db, reg, int_to_basename, args.dataset,
                args.k_coarse, args.k_final, allow_pairs=allow_pairs,
            )
            _print_summary(m, "SC UNION (per-col + rollup)")
            out["union"] = m
        if not args.skip_counter and not args.nonnumeric_only:
            if args.paper_mode:
                mc = _run_union_sc_counter_paper_sweep(
                    cfg, db, reg, int_to_basename, args.dataset,
                )
                _print_summary(
                    mc,
                    "SC UNION (per-col + Counter combiner, paper-mode "
                    "[SC k=10·k, Counter k=k per evaluation k])",
                )
                out["union_sc_counter_paper_sweep"] = mc
            else:
                mc = _run_union_sc_counter(
                    cfg, reg, int_to_basename, args.dataset,
                    args.k_coarse, args.k_final,
                )
                _print_summary(mc, "SC UNION (per-col + Counter combiner)")
                out["union_sc_counter"] = mc
        if not args.skip_blend_plan and not args.nonnumeric_only:
            mb = _run_union_blend_plan(
                cfg, reg, int_to_basename, args.dataset, args.k_final
            )
            k = str(args.k_final)
            print(f"\n=== BLEND-PLAN UNION (Tasks.UnionSearch, Union combiner, k={args.k_final}) ===")
            print(f"  prec @ {args.k_final}: {mb['precision_at_k'][k]:.4f}")
            print(f"  rec  @ {args.k_final}: {mb['recall_at_k'][k]:.4f}")
            print(f"  wall:           {mb['wall_clock_seconds']:.1f}s")
            out["union_blend_plan"] = mb

    if args.task in ("join", "all"):
        reg, int_to_basename, table_col_to_name = _load_for_op(SemanticOp.SJ)
        if args.paper_mode and not args.nonnumeric_only:
            m = _run_join_sc_paper_sweep(
                cfg, db, reg, int_to_basename, table_col_to_name, args.dataset,
            )
            label_suffix = ", paper-mode [SC k=10·k, dedup top-k unique tables]"
            out_key = "join_paper_sweep"
        else:
            m = _run_join_sc(
                cfg, db, reg, int_to_basename, table_col_to_name, args.dataset,
                args.k_coarse, args.k_final, allow_pairs=allow_pairs,
            )
            label_suffix = ""
            out_key = "join"
        _print_summary(m, f"SC JOIN (table-level{label_suffix})")
        print(f"\n=== SC JOIN (column-level) achieved @ k ∈ {list(K_RANGE)} ===")
        pc = m["precision_col_at_k"]
        rc = m["recall_col_at_k"]
        nc = m["ndcg_col_at_k"]
        mc = m["map_col_at_k"]
        print(f"  {'k':>4}  {'prec':>8}  {'recall':>8}  {'ndcg':>8}  {'map':>8}")
        print(f"  {'-' * 4}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 8}")
        for k in K_RANGE:
            sk = str(k)
            print(
                f"  {k:>4}  {pc[sk]:>8.4f}  {rc[sk]:>8.4f}  "
                f"{nc[sk]:>8.4f}  {mc[sk]:>8.4f}"
            )
        out[out_key] = m

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
