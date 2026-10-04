"""Build a value-overlap registry scoped to the SHO-indexed (non-numeric) columns.

    uv run python scripts/build_sho_overlap_registry.py --dataset opendata-split_13
"""
from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import duckdb
import pandas as pd
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
csv.field_size_limit(sys.maxsize)

from src.Semantic.config import SemanticConfig  # noqa: E402
from src.Benchmark.nonnumeric_filter import load_sho_nonnumeric  # noqa: E402
from scripts.build_overlap_registry import FIELDS, norm_name  # noqa: E402

JOIN_HEADER = {"query_table", "candidate_table", "query_column", "candidate_column"}
UNION_HEADER = {"query_table", "candidate_table"}
NORM = "lower(trim(trim(trim({c}), '\"''')))"


def build_valset(con, cfg, cols_by_tbl, basename_to_int):
    """Distinct (tid, colid, val) rows over the SHO-indexed columns, one DuckDB read per table."""
    csvs = cfg.dataset.dir() / "csvs"
    con.execute("CREATE TABLE valset(tid INTEGER, colid INTEGER, val VARCHAR)")
    bad = 0
    for bn in tqdm(sorted(cols_by_tbl), desc="reg-scan", unit="tbl"):
        tid = basename_to_int.get(bn)
        if tid is None:
            continue
        src = (f"read_csv('{(csvs / bn).as_posix()}', all_varchar=true, header=false, "
               f"skip=1, ignore_errors=true, null_padding=true)")
        try:
            names = [d[0] for d in con.execute(f"SELECT * FROM {src} LIMIT 0").description]
            cids = [c for c in cols_by_tbl[bn] if c < len(names)]
            if not cids:
                continue
            sel = ", ".join(f'"{names[c]}" AS "c{c}"' for c in cids)
            con.execute("DROP TABLE IF EXISTS _tmp")
            con.execute(f"CREATE TEMP TABLE _tmp AS SELECT {sel} FROM {src}")
            inlist = ", ".join(f'"c{c}"' for c in cids)
            con.execute(
                f"INSERT INTO valset SELECT DISTINCT {tid} AS tid, "
                f"CAST(substr(cn, 2) AS INTEGER) AS colid, {NORM.format(c='val')} AS v "
                f"FROM _tmp UNPIVOT (val FOR cn IN ({inlist})) "
                f"WHERE {NORM.format(c='val')} IS NOT NULL "
                f"AND {NORM.format(c='val')} NOT IN ('', 'nan', 'none')")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"  skip {bn}: {str(e)[:100]}", file=sys.stderr)
    con.execute("DROP TABLE IF EXISTS _tmp")
    con.execute("CREATE INDEX i_tc ON valset(tid, colid)")
    con.execute("CREATE INDEX i_v ON valset(val)")
    return bad


def _blank(kind, dataset, gt_name, idx, r, status):
    return {"dataset": dataset, "gt_file": gt_name, "kind": kind, "row_idx": idx,
            "query_table": r["query_table"], "candidate_table": r["candidate_table"],
            "query_column": r.get("query_column", ""), "candidate_column": r.get("candidate_column", ""),
            "n_query": "", "n_cand": "", "n_inter": "", "cont_q2x": "", "cont_x2q": "",
            "jaccard": "", "numeric_q": "false", "numeric_x": "false", "status": status}


def _fill(rec, nq, nc, ni):
    cq = ni / nq if nq else 0.0
    cx = ni / nc if nc else 0.0
    den = nq + nc - ni
    rec.update(n_query=nq, n_cand=nc, n_inter=ni, cont_q2x=f"{cq:.6f}",
               cont_x2q=f"{cx:.6f}", jaccard=f"{(ni / den if den else 0.0):.6f}", status="ok")
    return rec


def process_join(con, gt_path, rows, dataset, b2i, name_colid, present_tids):
    colcount = {(int(t), int(c)): int(n) for t, c, n in
                con.execute("SELECT tid, colid, COUNT(*) FROM valset GROUP BY tid, colid").fetchall()}
    todo, recs = [], {}
    for i, r in enumerate(rows):
        qt, ct = r["query_table"], r["candidate_table"]
        cq = name_colid.get(qt, {}).get(norm_name(r.get("query_column", "")))
        cc = name_colid.get(ct, {}).get(norm_name(r.get("candidate_column", "")))
        tq, tc = b2i.get(qt), b2i.get(ct)
        if tq is None or tc is None or tq not in present_tids or tc not in present_tids:
            recs[i] = _blank("join", dataset, gt_path.name, i, r, "missing_table")
        elif cq is None or cc is None:
            recs[i] = _blank("join", dataset, gt_path.name, i, r, "col_not_indexed")
        else:
            todo.append((i, tq, cq, tc, cc))
    if todo:
        jdf = pd.DataFrame(todo, columns=["i", "tq", "cq", "tc", "cc"])  # noqa: F841
        inter = con.execute("""
            SELECT j.i, COUNT(*) ni FROM jdf j
            JOIN valset a ON a.tid=j.tq AND a.colid=j.cq
            JOIN valset b ON b.tid=j.tc AND b.colid=j.cc AND b.val=a.val
            GROUP BY j.i""").df()
        nmap = {int(i): int(n) for i, n in zip(inter["i"], inter["ni"])}
        for i, tq, cq, tc, cc in todo:
            recs[i] = _fill(_blank("join", dataset, gt_path.name, i, rows[i], "ok"),
                            colcount.get((tq, cq), 0), colcount.get((tc, cc), 0), nmap.get(i, 0))
    return [recs[i] for i in range(len(rows))]


def process_union(con, gt_path, rows, dataset, b2i, present_tids):
    tabcount = {int(t): int(n) for t, n in
                con.execute("SELECT tid, COUNT(DISTINCT val) FROM valset GROUP BY tid").fetchall()}
    todo, recs = [], {}
    for i, r in enumerate(rows):
        qt, ct = r["query_table"], r["candidate_table"]
        tq, tc = b2i.get(qt), b2i.get(ct)
        if tq is None or tc is None or tq not in present_tids or tc not in present_tids:
            recs[i] = _blank("union", dataset, gt_path.name, i, r, "missing_table")
        else:
            todo.append((i, tq, tc))
    if todo:
        udf = pd.DataFrame(todo, columns=["i", "tq", "tc"])  # noqa: F841
        inter = con.execute("""
            SELECT i, COUNT(*) ni FROM (
              SELECT DISTINCT u.i, a.val FROM udf u
              JOIN valset a ON a.tid=u.tq
              JOIN valset b ON b.tid=u.tc AND b.val=a.val
            ) GROUP BY i""").df()
        nmap = {int(i): int(n) for i, n in zip(inter["i"], inter["ni"])}
        for i, tq, tc in todo:
            recs[i] = _fill(_blank("union", dataset, gt_path.name, i, rows[i], "ok"),
                            tabcount.get(tq, 0), tabcount.get(tc, 0), nmap.get(i, 0))
    return [recs[i] for i in range(len(rows))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="datasets/<name> (reads its config.ini)")
    ap.add_argument("--only", action="append", default=[],
                    help="substring filter on gt filename (repeatable)")
    args = ap.parse_args()

    cfg = SemanticConfig.load(path=REPO / "datasets" / args.dataset / "config.ini")
    u = load_sho_nonnumeric(cfg)
    b2i = {v: k for k, v in u.int_to_basename.items()}
    name_colid: dict[str, dict[str, int]] = defaultdict(dict)
    for (bn, cid), nm in u.table_col_to_name.items():
        name_colid[bn][norm_name(nm)] = cid

    gt_dir = cfg.dataset.dir() / "groundtruth"
    gts = [g for g in sorted(gt_dir.glob("*.csv"))
           if not g.name.endswith(".overlap.csv")
           and (not args.only or any(s in g.name for s in args.only))]
    if not gts:
        print(f"no groundtruth CSVs under {gt_dir}", file=sys.stderr)
        return 1

    cols_by_tbl: dict[str, list[int]] = defaultdict(list)
    for (bn, cid) in u.table_col_to_name:
        cols_by_tbl[bn].append(cid)

    print(f"dataset      : {cfg.dataset.name}", file=sys.stderr)
    print(f"reg universe : {len(u.table_col_to_name)} cols / {len(cols_by_tbl)} tables", file=sys.stderr)
    print(f"gt files     : {[g.name for g in gts]}", file=sys.stderr)

    with tempfile.TemporaryDirectory() as td:
        con = duckdb.connect(str(Path(td) / "valset.duckdb"))
        bad = build_valset(con, cfg, cols_by_tbl, b2i)
        present_tids = {t for (t,) in con.execute("SELECT DISTINCT tid FROM valset").fetchall()}
        print(f"scanned      : {len(present_tids)} tables (skipped {bad})", file=sys.stderr)

        for gt in gts:
            with gt.open(newline="", encoding="utf-8", errors="replace") as fh:
                rows = list(csv.DictReader(fh))
            header = set(rows[0].keys()) if rows else set()
            if header >= JOIN_HEADER:
                out = process_join(con, gt, rows, cfg.dataset.name, b2i, name_colid, present_tids)
            elif header >= UNION_HEADER:
                out = process_union(con, gt, rows, cfg.dataset.name, b2i, present_tids)
            else:
                print(f"skip {gt.name}: header {sorted(header)}", file=sys.stderr)
                continue
            out_path = gt.with_name(gt.stem + ".regscoped.overlap.csv")
            with out_path.open("w", newline="", encoding="utf-8") as ofh:
                w = csv.DictWriter(ofh, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(out)
            ok = sum(1 for r in out if r["status"] == "ok")
            sem = sum(1 for r in out if r["status"] == "ok" and r["n_inter"] == 0)
            print(f"wrote {out_path}  ({len(out)} rows, ok={ok}, semantic={sem})", file=sys.stderr)
        con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
