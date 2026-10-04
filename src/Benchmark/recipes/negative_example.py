from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from src.Benchmark.datasource import (
    load_query_table, load_sidecar, load_union_gt, load_union_queries)
from src.Benchmark.prepare import guard_outputs


def _row_set(df: pd.DataFrame, c1: str, c2: str) -> set[tuple[str, str]]:
    sub = df[[c1, c2]].dropna().drop_duplicates()
    return set(map(tuple, sub.astype(str).values))


def _is_superset(candidate_rows: set, query_rows: set) -> bool:
    return query_rows.issubset(candidate_rows)


def _restrict_gt(dataset: str, query_table: str, c1: str, c2: str,
                 candidates: set[str], lake: set[str]) -> set[str]:
    """Paper p.112: keep unionable tables that do NOT completely overlap the query."""
    qdf = load_query_table(dataset, query_table)
    if c1 not in qdf.columns or c2 not in qdf.columns:
        return set()
    qrows = _row_set(qdf, c1, c2)
    keep = set()
    for cand in candidates & lake:
        if cand == query_table:
            continue
        cdf = load_query_table(dataset, cand)
        if c1 not in cdf.columns or c2 not in cdf.columns:
            continue
        crows = _row_set(cdf, c1, c2)
        if not _is_superset(crows, qrows):
            keep.add(cand)
    return keep


def prepare_negex(dataset: str, out: Path, *, n_neg: int, seed: int, force: bool) -> None:
    _, lake = load_sidecar(dataset)
    union_gt = load_union_gt(dataset)
    queries = [q for q in load_union_queries(dataset) if q in lake]
    q_path, gt_path, mf_path = out / "query.csv", out / "groundtruth.csv", out / "manifest.json"
    guard_outputs([q_path, gt_path, mf_path], force)
    with q_path.open("w", newline="") as qf, gt_path.open("w", newline="") as gf:
        qw = csv.writer(qf); qw.writerow(["query_id", "query_table", "col1", "col2", "n_neg", "seed"])
        gw = csv.writer(gf); gw.writerow(["query_id", "candidate_table"])
        written = 0
        for q in queries:
            qdf = load_query_table(dataset, q)
            if qdf.shape[1] < 2:
                continue
            c1, c2 = qdf.columns[0], qdf.columns[1]
            restricted = _restrict_gt(dataset, q, c1, c2, union_gt.get(q, set()), lake)
            if not restricted:
                continue
            qw.writerow([written, q, c1, c2, n_neg, seed])
            for cand in sorted(restricted):
                gw.writerow([written, cand])
            written += 1
    mf_path.write_text(json.dumps(
        {"task": "negex", "seed": seed, "n_neg": n_neg, "n_written": written}, indent=2))
    print(f"negex: wrote {written} queries to {out}")
