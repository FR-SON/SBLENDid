"""Persist a per-dataset value-overlap registry (<gt_stem>.overlap.csv) for groundtruth pairs."""
from __future__ import annotations

import argparse
import csv
import sys
import unicodedata
from collections import namedtuple
from functools import lru_cache
from pathlib import Path

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO_ROOT / "datasets"

JOIN_HEADER = {"query_table", "candidate_table", "query_column", "candidate_column"}
UNION_HEADER = {"query_table", "candidate_table"}
_NULL = frozenset({"", "nan", "none"})

csv.field_size_limit(sys.maxsize)


def norm_value(v) -> str:
    s = unicodedata.normalize("NFKC", str(v)).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1].strip()
    s = s.casefold()
    return "" if s in _NULL else s


def norm_name(s) -> str:
    s = str(s).replace("﻿", "").replace("ï»¿", "")
    return s.strip().strip('"').strip("'").strip().casefold()


def resolve_col(colnames, name):
    if name in colnames:
        return name
    t = norm_name(name)
    for c in colnames:
        if norm_name(c) == t:
            return c
    return None


def is_numeric_column(values) -> bool:
    if not values:
        return False
    top = pd.Series(values).value_counts().index[:10].tolist()
    joined = "".join(str(v) for v in top)
    if not joined:
        return False
    return sum(ch.isdigit() for ch in joined) / len(joined) > 0.5


def pair_stats(q_vals, c_vals):
    qs, cs = set(q_vals), set(c_vals)
    nq, nc = len(qs), len(cs)
    ninter = len(qs & cs)
    cont_q2x = ninter / nq if nq else 0.0
    cont_x2q = ninter / nc if nc else 0.0
    denom = nq + nc - ninter
    jac = ninter / denom if denom else 0.0
    return nq, nc, ninter, cont_q2x, cont_x2q, jac


def union_best(q_cols, c_cols):
    best, best_sym = None, -1.0
    for qc, qv in q_cols.items():
        for cc, cv in c_cols.items():
            st = pair_stats(qv, cv)
            sym = max(st[3], st[4])
            if sym > best_sym:
                best_sym, best = sym, (qc, cc, st)
    return best


def norm_distinct(path, row_cap, max_distinct):
    try:
        fh = open(path, newline="", encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return {}
    with fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return {}
        seen = [set() for _ in header]
        cols = [[] for _ in header]
        for i, row in enumerate(reader):
            if row_cap and i >= row_cap:
                break
            for j, cell in enumerate(row[:len(header)]):
                v = norm_value(cell)
                if not v or v in seen[j]:
                    continue
                if max_distinct and len(cols[j]) >= max_distinct:
                    continue
                seen[j].add(v)
                cols[j].append(v)
    out = {}
    for name, lst in zip(header, cols):
        if lst and name not in out:
            out[name] = lst
    return out


def make_loader(row_cap, max_distinct, cache_size):
    @lru_cache(maxsize=cache_size)
    def _load(path_str):
        return norm_distinct(Path(path_str), row_cap, max_distinct)
    return _load


GtFile = namedtuple("GtFile", "path dataset kind csvs_dir")


def discover_gt(datasets_dir, dataset=None, only=()):
    out = []
    for ds_dir in sorted(p for p in datasets_dir.iterdir() if p.is_dir()):
        if dataset and ds_dir.name != dataset:
            continue
        gt_dir = ds_dir / "groundtruth"
        if not gt_dir.is_dir():
            continue
        csvs_dir = ds_dir / "csvs"
        for gt in sorted(gt_dir.glob("*.csv")):
            if gt.name.endswith(".overlap.csv"):
                continue
            if only and not any(s in gt.name for s in only):
                continue
            with gt.open(newline="", encoding="utf-8", errors="replace") as fh:
                header = next(csv.reader(fh))
            cols = set(header)
            if cols >= JOIN_HEADER:
                kind = "join"
            elif cols >= UNION_HEADER:
                kind = "union"
            else:
                print(f"skip {gt}: header {header}", file=sys.stderr)
                continue
            out.append(GtFile(gt, ds_dir.name, kind, csvs_dir))
    return out


FIELDS = [
    "dataset", "gt_file", "kind", "row_idx", "query_table", "candidate_table",
    "query_column", "candidate_column", "n_query", "n_cand", "n_inter",
    "cont_q2x", "cont_x2q", "jaccard", "numeric_q", "numeric_x", "status",
]


def _blank_rec(dataset, gt_name, kind, idx, row):
    return {
        "dataset": dataset, "gt_file": gt_name, "kind": kind, "row_idx": idx,
        "query_table": row["query_table"], "candidate_table": row["candidate_table"],
        "query_column": row.get("query_column", ""),
        "candidate_column": row.get("candidate_column", ""),
        "n_query": "", "n_cand": "", "n_inter": "",
        "cont_q2x": "", "cont_x2q": "", "jaccard": "",
        "numeric_q": "", "numeric_x": "", "status": "ok",
    }


def score_row(kind, dataset, gt_name, idx, row, csvs_dir, loader):
    rec = _blank_rec(dataset, gt_name, kind, idx, row)
    q_cols = loader(str(csvs_dir / row["query_table"]))
    c_cols = loader(str(csvs_dir / row["candidate_table"]))
    if not q_cols or not c_cols:
        rec["status"] = "missing_table"
        return rec
    if kind == "join":
        qa = resolve_col(list(q_cols), row["query_column"])
        ca = resolve_col(list(c_cols), row["candidate_column"])
        if qa is None or ca is None:
            rec["status"] = "missing_column"
            return rec
        st = pair_stats(q_cols[qa], c_cols[ca])
    else:
        qa, ca, st = union_best(q_cols, c_cols)
    nq, nc, ninter, cq, cx, jac = st
    rec.update(
        query_column=qa, candidate_column=ca,
        n_query=nq, n_cand=nc, n_inter=ninter,
        cont_q2x=f"{cq:.6f}", cont_x2q=f"{cx:.6f}", jaccard=f"{jac:.6f}",
        numeric_q=str(is_numeric_column(q_cols[qa])).lower(),
        numeric_x=str(is_numeric_column(c_cols[ca])).lower(),
    )
    return rec


def process_file(gt, loader):
    with gt.path.open(newline="", encoding="utf-8", errors="replace") as fh:
        rows = list(csv.DictReader(fh))
    out_path = gt.path.with_name(gt.path.stem + ".overlap.csv")
    with out_path.open("w", newline="", encoding="utf-8") as ofh:
        w = csv.DictWriter(ofh, fieldnames=FIELDS)
        w.writeheader()
        for idx, r in enumerate(tqdm(rows, desc=gt.path.name, unit="pair", leave=False)):
            w.writerow(score_row(gt.kind, gt.dataset, gt.path.name, idx, r, gt.csvs_dir, loader))
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets-dir", type=Path, default=DATASETS_DIR)
    ap.add_argument("--dataset", help="limit to one datasets/<name> dir")
    ap.add_argument("--only", action="append", default=[],
                    help="substring filter on gt filename (repeatable)")
    ap.add_argument("--row-cap", type=int, default=0,
                    help="max rows scanned per CSV; 0 = unbounded")
    ap.add_argument("--max-distinct", type=int, default=0,
                    help="cap distinct values per column; 0 = unbounded")
    ap.add_argument("--cache-size", type=int, default=4096)
    args = ap.parse_args()

    gts = discover_gt(args.datasets_dir, args.dataset, tuple(args.only))
    if not gts:
        print(f"no groundtruth files under {args.datasets_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"found {len(gts)} groundtruth files", file=sys.stderr)
    loader = make_loader(args.row_cap, args.max_distinct, args.cache_size)
    for gt in tqdm(gts, desc="gt files", unit="file"):
        out = process_file(gt, loader)
        print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
