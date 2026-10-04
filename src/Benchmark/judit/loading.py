"""Query/GT CSV loading and registry-based filtering for the union/join harness."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.Benchmark.self_filter import drop_self_gt
from src.Semantic.config import SemanticConfig


@dataclass
class UnionQGT:
    queries_kept: list[str]
    relevant_by_q: dict[str, set[str]]
    header_cols: dict[str, list[str]]
    n_queries_input: int
    n_gt_rows_input: int
    n_gt_rows_kept: int
    n_dropped_q: int
    n_dropped_c: int


@dataclass
class JoinQGT:
    pairs_kept: list[tuple[str, str]]
    relevant_by_q: dict[tuple[str, str], set[str]]
    relevant_cols_by_q: dict[tuple[str, str], set[tuple[str, str]]]
    n_queries_input: int
    n_gt_rows_input: int
    n_gt_rows_kept: int
    n_dropped_q: int
    n_dropped_c: int
    n_gt_col_kept: int
    n_gt_col_dropped_missing: int


def load_union_qgt(
    cfg: SemanticConfig, prefix: str, reg: dict[str, set[str]]
) -> UnionQGT:
    base = cfg.dataset.dir()
    queries_df = pd.read_csv(base / "query" / f"{prefix}_union_query.csv")
    gt = pd.read_csv(base / "groundtruth" / f"{prefix}_union_ground_truth.csv")
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

    csvs_dir = base / "csvs"
    header_cols: dict[str, list[str]] = {}
    for qt in queries_kept:
        header_cols[qt] = pd.read_csv(
            csvs_dir / qt, dtype=str, keep_default_na=False, nrows=0
        ).columns.tolist()

    return UnionQGT(
        queries_kept=queries_kept,
        relevant_by_q=relevant_by_q,
        header_cols=header_cols,
        n_queries_input=n_q_input,
        n_gt_rows_input=n_gt_input,
        n_gt_rows_kept=len(gt),
        n_dropped_q=n_dropped_q,
        n_dropped_c=n_dropped_c,
    )


def load_join_qgt(
    cfg: SemanticConfig, prefix: str, reg: dict[str, set[str]]
) -> JoinQGT:
    base = cfg.dataset.dir()
    queries_df = pd.read_csv(base / "query" / f"{prefix}_join_query.csv")
    gt = pd.read_csv(base / "groundtruth" / f"{prefix}_join_ground_truth.csv")
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
        relevant_by_q.setdefault((str(r.query_table), str(r.query_column)), set()).add(
            str(r.candidate_table)
        )

    if len(gt):
        mask = [
            c in reg.get(t, set())
            for t, c in zip(
                gt["candidate_table"].astype(str), gt["candidate_column"].astype(str)
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

    return JoinQGT(
        pairs_kept=pairs_kept,
        relevant_by_q=relevant_by_q,
        relevant_cols_by_q=relevant_cols_by_q,
        n_queries_input=n_q_input,
        n_gt_rows_input=n_gt_input,
        n_gt_rows_kept=len(gt),
        n_dropped_q=n_dropped_q,
        n_dropped_c=n_dropped_c,
        n_gt_col_kept=n_gt_col_kept,
        n_gt_col_dropped_missing=n_gt_col_dropped_missing,
    )
