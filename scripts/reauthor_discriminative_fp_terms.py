"""Re-author oracle `terms_fp` discriminatively: tokens in a query's FP tables but absent from its relevant ones."""
from __future__ import annotations

import argparse
import json
from collections import Counter

from src import paths


def _table_tokens(dataset: str, basenames) -> Counter:
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
    return counts


def discriminative_fp_tokens(dataset: str, fps, relevant, top_n: int = 8) -> list[str]:
    fp_counts = _table_tokens(dataset, fps)
    rel_tokens = set(_table_tokens(dataset, relevant))
    cand = [(tok, c) for tok, c in fp_counts.items() if tok not in rel_tokens]
    cand.sort(key=lambda tc: (-tc[1], -len(tc[0])))
    return [tok for tok, _ in cand[:top_n]]


def main(argv: list[str] | None = None) -> int:
    import os
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    from tqdm import tqdm
    from src.Benchmark.datasource import load_sidecar, load_union_gt, load_join_gt
    from src.Benchmark.db import open_dataset_db
    from src.Benchmark.plans.semantic_oracle_repair import _seeker_ids
    from src.Operators import Seekers

    p = argparse.ArgumentParser(prog="reauthor_discriminative_fp_terms")
    p.add_argument("--dataset", required=True)
    p.add_argument("--task", choices=["union", "join"], required=True)
    p.add_argument("--top-n", type=int, default=8)
    args = p.parse_args(argv)

    orc_dir = paths.datasets_root() / args.dataset / "plans" / "oracle"
    fix_path = orc_dir / f"{args.task}_oracle_keywords.json"
    fixture = json.loads(fix_path.read_text())

    i2b, _ = load_sidecar(args.dataset)
    lake = set(i2b.values())
    gt = load_union_gt(args.dataset) if args.task == "union" else load_join_gt(args.dataset)

    def relevant_of(qid: str) -> set:
        if args.task == "union":
            return gt.get(qid, set()) & lake
        tbl, col = qid.split("::", 1)
        return gt.get((tbl, col), set()) & lake

    for qid, entry in tqdm(fixture.items(), desc=f"reauthor/{args.task}"):
        fps = entry.get("_fps") or []
        entry["terms_fp"] = (
            discriminative_fp_tokens(args.dataset, fps, relevant_of(qid), args.top_n)
            if fps else []
        )

    db = open_dataset_db(args.dataset)
    try:
        print(f"  {'qid':<40}{'flagged/fps':>13}{'leaked/rel':>13}")
        for qid, entry in fixture.items():
            tf = entry.get("terms_fp") or []
            fps = set(entry.get("_fps") or [])
            rel = relevant_of(qid)
            if not tf:
                print(f"  {qid:<40}{'0/' + str(len(fps)):>13}{'(no terms)':>13}")
                continue
            out = set(_seeker_ids(Seekers.Keyword(tf, k=100), db, i2b))
            print(f"  {qid:<40}{str(len(out & fps)) + '/' + str(len(fps)):>13}"
                  f"{str(len(out & rel)) + '/' + str(len(rel)):>13}")
    finally:
        db.close()

    fix_path.write_text(json.dumps(fixture, indent=2))
    (orc_dir / f"{args.task}_oracle_keywords.midband_v2_discriminative.json").write_text(
        json.dumps(fixture, indent=2))
    print(f"wrote {fix_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
