"""Drive `Benchmark.depth_grid` over every lake, encoder and task.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 BLEND_DATASETS_DIR=<datasets-root> .venv/bin/python scripts/run_depth_grid.py --out <out-dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.Benchmark.depth_grid import (  # noqa: E402
    DEFAULT_ARMS, KC_GRID, K_RANGE, VOTE_DEPTHS, best_vote_depth,
    margin_effect, run_depth_grid,
)

LAKES = [
    ("santos", "santos", ["union"]),
    ("opendata-split-11", "opendata", ["union", "join"]),
    ("opendata-split-12", "opendata", ["union", "join"]),
    ("opendata-split_13", "opendata", ["union", "join"]),
    ("tus_large", "tus_large", ["union"]),
]


def encoders_for(datasets_root: Path, lake: str) -> list[tuple[str, str]]:
    """Every approach with a built HNSW index, in a stable order."""
    base = datasets_root / lake / "semantic"
    found = []
    for approach in ("liftus", "snoopy", "deepjoin"):
        idx = base / approach / "default" / "index" / "hnsw.faiss"
        if idx.exists():
            found.append((approach, "default"))
    return found


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    cols: list[str] = []
    for r in rows:
        for c in r:
            if c not in cols:
                cols.append(c)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap queries per (lake, task) — smoke runs only")
    ap.add_argument("--lakes", nargs="*", default=None)
    ap.add_argument("--backend", default="faiss", choices=["faiss", "pgvector"])
    ap.add_argument("--arms", nargs="*", default=None,
                    help="restrict to these arm names")
    ap.add_argument("--encoders", nargs="*", default=None,
                    help="restrict to these approaches (pg holds fewer than disk)")
    args = ap.parse_args()

    from src import paths
    datasets_root = paths.datasets_root()
    args.out.mkdir(parents=True, exist_ok=True)

    all_summary: list[dict] = []
    t0 = time.perf_counter()

    for lake, prefix, tasks in LAKES:
        if args.lakes and lake not in args.lakes:
            continue
        encs = encoders_for(datasets_root, lake)
        if args.encoders:
            encs = [e for e in encs if e[0] in args.encoders]
        if not encs:
            print(f"[{lake}] no built index — skipped", flush=True)
            continue
        print(f"\n=== {lake} (prefix={prefix}) encoders={[e for e, _ in encs]} "
              f"tasks={tasks} ===", flush=True)
        summary = run_depth_grid(
            lake, prefix=prefix, approaches=encs, tasks=tasks,
            arms=tuple(a for a in DEFAULT_ARMS if not args.arms or a.name in args.arms),
            ks=K_RANGE, kc_grid=KC_GRID,
            vote_depths=VOTE_DEPTHS, backend=args.backend, limit=args.limit,
            log=lambda m: print(m, flush=True),
        )
        all_summary.extend(summary)
        write_csv(args.out / "summary.csv", all_summary)

    for k in (10, 25, 50, 100):
        write_csv(args.out / f"best_vote_depth_k{k}.csv",
                  best_vote_depth(all_summary, k, min_margin=25))
    write_csv(args.out / "margin_effect.csv", margin_effect(all_summary))
    (args.out / "run_meta.json").write_text(json.dumps({
        "datasets_root": str(datasets_root),
        "arms": [
            {"name": a.name, "ef_search": a.ef, "max_fetch": a.max_fetch, "role": a.role}
            for a in DEFAULT_ARMS
        ],
        "vote_depths": list(VOTE_DEPTHS),
        "kc_grid": list(KC_GRID),
        "backend": args.backend,
        "ks": list(K_RANGE),
        "lakes": [{"lake": l, "prefix": p, "tasks": t} for l, p, t in LAKES],
        "limit": args.limit,
        "wall_seconds": round(time.perf_counter() - t0, 1),
    }, indent=2))

    print(f"\ncells: {len(all_summary)}", flush=True)
    print(f"done in {time.perf_counter() - t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
