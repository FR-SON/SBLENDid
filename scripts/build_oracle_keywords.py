"""Rank in-universe queries by seeker quality and emit a worst-N keyword scaffold for semantic_oracle_repair."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src import paths


def propose_tokens(dataset: str, basenames: list[str], top_n: int = 20) -> list[str]:
    from src.Benchmark.datasource import load_query_table
    from src.Index.tokenize import tokenize_cell
    counts: Counter[str] = Counter()
    for b in basenames:
        try:
            df = load_query_table(dataset, b)
        except FileNotFoundError:
            continue
        seen: set[str] = set()
        for col in df.columns:
            for v in df[col]:
                tok = tokenize_cell(v)
                if len(tok) >= 2 and not tok.isdigit():
                    seen.add(tok)
        counts.update(seen)
    return [tok for tok, _ in counts.most_common(top_n)]


def main(argv: list[str] | None = None) -> int:
    import os
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    from tqdm import tqdm
    from src.Benchmark import metrics as M
    from src.Benchmark.datasource import load_sidecar
    from src.Benchmark.db import open_dataset_db
    from src.Benchmark.plans.registry import PlanContext
    from src.Benchmark.plans.semantic_oracle_repair import _plan_units, _seeker_ids
    from src.Semantic.config import SemanticConfig, SemanticOp
    from src.Semantic.retrieve import IndexHandle

    p = argparse.ArgumentParser(prog="build_oracle_keywords")
    p.add_argument("--dataset", required=True)
    p.add_argument("--task", choices=["union", "join"], required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--worst", type=int, default=15)
    p.add_argument("--select", choices=["worst", "band"], default="worst",
                   help="worst: lowest recall/precision; band: a recall/precision window")
    p.add_argument("--recall-min", dest="recall_min", type=float, default=0.0)
    p.add_argument("--recall-max", dest="recall_max", type=float, default=1.0)
    p.add_argument("--precision-min", dest="precision_min", type=float, default=0.0)
    p.add_argument("--precision-max", dest="precision_max", type=float, default=1.0)
    args = p.parse_args(argv)

    i2b, _lake = load_sidecar(args.dataset)
    cfg = SemanticConfig.load(overrides={"dataset": args.dataset})
    op = SemanticOp.SU if args.task == "union" else SemanticOp.SJ
    oc = cfg.operator(op)
    handle = IndexHandle.open(cfg, oc.approach, oc.index_name)
    enrolled_tables = set(handle.table_to_int_id)
    enrolled_cols = set(handle.table_col_to_gid)

    db = open_dataset_db(args.dataset)
    scored: list[dict] = []
    try:
        ctx = PlanContext(dataset=args.dataset, db=db, task=args.task, k=args.k,
                          seed=0, out_dir=Path("."), git_sha="", config_snapshot={})
        units = _plan_units(ctx, enrolled_tables, enrolled_cols, i2b)
        for u in tqdm(units, desc=f"score/{args.task}"):
            R = _seeker_ids(u["seeker"](), db, i2b)
            rel = u["relevant"]
            scored.append({
                "qid": u["qid"],
                "recall": M.recall_at_k(R, rel),
                "precision": M.precision_at_k(R, rel),
                "misses": sorted(rel - set(R)),
                "fps": sorted(set(R) - rel),
            })
    finally:
        db.close()

    scored.sort(key=lambda s: (s["recall"], s["precision"]))
    if args.select == "band":
        band = [s for s in scored
                if args.recall_min <= s["recall"] <= args.recall_max
                and args.precision_min <= s["precision"] <= args.precision_max]
        band.sort(key=lambda s: (not (s["misses"] and s["fps"]), s["recall"]))
        selected = band[: args.worst]
        sel_desc = (f"band recall[{args.recall_min},{args.recall_max}] "
                    f"precision[{args.precision_min},{args.precision_max}], "
                    f"{len(band)} in band")
    else:
        selected = scored[: args.worst]
        sel_desc = "worst"

    fixture = {}
    for s in selected:
        fixture[s["qid"]] = {
            "_recall": round(s["recall"], 4),
            "_precision": round(s["precision"], 4),
            "_misses": s["misses"],
            "_fps": s["fps"],
            "_proposed_recall_tokens": propose_tokens(args.dataset, s["misses"]),
            "_proposed_fp_tokens": propose_tokens(args.dataset, s["fps"]),
            "terms_recall": [],
            "terms_fp": [],
        }

    out_dir = paths.datasets_root() / args.dataset / "plans" / "oracle"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.task}_oracle_keywords.scaffold.json"
    out.write_text(json.dumps(fixture, indent=2))
    print(f"wrote {out} ({len(selected)} queries, {sel_desc})")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
