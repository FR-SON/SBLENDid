"""Matrix driver: tasks x methods gated by availability, --repeat medians, run-dir output."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.Benchmark.judit.seekers import base, make_backend
# after judit.seekers, which sets the KMP/OMP env before torch+faiss load
from src.Semantic import depths
from src.Semantic.config import SemanticOp


def median_tree(values):
    """Per-leaf median over a list of same-shaped dicts; non-numeric leaves pass through."""
    first = values[0]
    if isinstance(first, dict):
        return {k: median_tree([v[k] for v in values]) for k in first}
    nums = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if len(nums) == len(values) and nums:
        return statistics.median(nums)
    return first


def resolve_plan(tasks, methods, cfg):
    """(legs, skipped) for the requested tasks x methods."""
    legs, skipped = [], []
    for method in methods:
        be = make_backend(method, cfg)
        if not be.available():
            for t in tasks:
                skipped.append({"task": t, "method": method,
                                "reason": f"unavailable: {method} artifacts absent"})
            continue
        for t in tasks:
            legs.append((t, be))
    return legs, skipped


def _leg_vote_factor(cfg, task, backend, override):
    if not getattr(backend, "uses_rank_rollup", False):
        return depths.DEFAULT_VOTE_FACTOR
    op = SemanticOp.SU if task == "union" else SemanticOp.SJ
    return cfg.vote_factor_for(op, override=override)


def run_matrix(cfg, prefix, *, tasks, methods, repeat, ks, k_coarse, k_vote,
               out_dir, limit=None, vote_factor=None, git_sha="", created=""):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    legs, skipped = resolve_plan(tasks, methods, cfg)
    print(f"plan: {[(t, b.method) for t, b in legs]}  skipped: {skipped}", flush=True)

    agg_legs, per_repeat, per_query, failed = [], {}, [], []
    open_backends: dict[int, object] = {}

    def _flush():
        aggregate = {"dataset": cfg.dataset.name, "prefix": prefix, "git_sha": git_sha,
                     "vector_backend": cfg.vector_backend, "created": created,
                     "k_range": list(ks), "k_coarse": k_coarse, "k_vote": k_vote,
                     "vote_factor": vote_factor,
                     "repeats": repeat, "legs": agg_legs, "skipped": skipped,
                     "failed": failed}
        (out_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2))
        (out_dir / "per_repeat.json").write_text(json.dumps(per_repeat, indent=2))
        pd.DataFrame(per_query).to_parquet(out_dir / "per_query.parquet", index=False)
        return aggregate

    for task, be in tqdm(legs, desc="legs", unit="leg", dynamic_ncols=True):
        open_backends[id(be)] = be
        run_leg = base.run_union_leg if task == "union" else base.run_join_leg
        vf = _leg_vote_factor(cfg, task, be, vote_factor)
        key = f"{task}.{be.method}"
        if be.index_name not in ("", "-", "default"):
            key += f".{be.index_name}"
        reps, rt_samples = [], []
        seg_samples = {tk: [] for tk in base.TIMING_KEYS}
        call_samples = []
        pq_mark = len(per_query)
        try:
            for r in range(repeat):
                leg, rows = run_leg(be, cfg, prefix, ks=ks, k_coarse=k_coarse,
                                    k_vote=k_vote, vote_factor=vf, limit=limit)
                reps.append(leg)
                for row in rows:
                    per_query.append({"task": task, "method": be.method,
                                      "approach": be.approach,
                                      "index_name": be.index_name, "repeat": r, **row})
                    rt_samples.append(row["runtime_ms"])
                    for tk, xs in seg_samples.items():
                        xs.append(row[tk])
                    call_samples.append(row["n_search_calls"])
        except Exception as e:
            del per_query[pq_mark:]
            failed.append({"task": task, "method": be.method,
                           "index_name": be.index_name, "reason": repr(e)})
            print(f"LEG FAILED {task}/{be.method}: {e!r}", flush=True)
            _flush()
            continue
        per_repeat[key] = reps
        med = median_tree(reps)
        med["runtime_ms_per_query"] = statistics.median(rt_samples) if rt_samples else 0.0
        for tk, xs in seg_samples.items():
            med[f"{tk}_per_query"] = statistics.median(xs) if xs else 0.0
        med["n_search_calls_per_query"] = (
            statistics.median(call_samples) if call_samples else 0.0)
        per_call = [t / n for t, n in zip(seg_samples["t_search_ms"], call_samples) if n]
        med["t_search_ms_per_call"] = statistics.median(per_call) if per_call else 0.0
        med["wall_clock_seconds"] = statistics.median(
            [x["wall_clock_seconds"] for x in reps])
        agg_legs.append({"task": task, "method": be.method, "approach": be.approach,
                         "index_name": be.index_name, "vote_factor": vf,
                         "n_queries_kept": reps[0]["n_queries_kept"],
                         "gt_density": reps[0]["gt_density"], "median": med})
        _flush()
    for be in open_backends.values():
        if hasattr(be, "close"):
            be.close()

    aggregate = _flush()
    _print_matrix(aggregate)
    print(f"\nwrote {out_dir}", flush=True)
    return out_dir


def _print_matrix(aggregate) -> None:
    ks = aggregate["k_range"]
    if not aggregate["legs"]:
        print("\nno legs ran (all requested (task,method) pairs unavailable).")
    for leg in aggregate["legs"]:
        med = leg["median"]
        label = f"{leg['task']}/{leg['method']}"
        if leg["index_name"] not in ("-", "default", ""):
            label += f":{leg['index_name']}"
        print(f"\n=== {label}  n={leg['n_queries_kept']}  "
              f"ms/q={med['runtime_ms_per_query']:.2f}  "
              f"wall={med['wall_clock_seconds']:.1f}s ===")
        print(f"  split/q: search={med.get('t_search_ms_per_query', 0.0):.3f}  "
              f"rollup={med.get('t_rollup_ms_per_query', 0.0):.3f}  "
              f"metrics={med.get('t_metrics_ms_per_query', 0.0):.3f} ms   "
              f"({med.get('n_search_calls_per_query', 0.0):.0f} calls/q, "
              f"{med.get('t_search_ms_per_call', 0.0):.3f} ms/call)")
        p, r, nd, m = (med["precision_at_k"], med["recall_at_k"],
                       med["ndcg_at_k"], med["map_at_k"])
        print(f"  {'k':>4}  {'prec':>8}  {'recall':>8}  {'ndcg':>8}  {'map':>8}")
        print(f"  {'-' * 4}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 8}")
        for k in ks:
            sk = str(k)
            print(f"  {k:>4}  {p[sk]:>8.4f}  {r[sk]:>8.4f}  {nd[sk]:>8.4f}  {m[sk]:>8.4f}")
    for s in aggregate["skipped"]:
        print(f"  SKIP {s['task']}/{s['method']}: {s['reason']}")
    for f in aggregate.get("failed", []):
        print(f"  FAILED {f['task']}/{f['method']}: {f['reason']}")
