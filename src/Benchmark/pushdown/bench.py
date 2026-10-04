import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import hashlib
import random
import sys
from statistics import mean, median
from time import perf_counter

from tqdm import tqdm

from src.Operators import Seekers
from src.Benchmark.datasource import load_sidecar, load_query_table
from src.Benchmark.correctness.refs import exact_semantic_ids
from src.Benchmark.correctness.sweep import _semantic_leg
from src.Benchmark import setdiff_metrics as SD


def build_allowed(exact_rank, U, *, k, c, target_size, rng, decoy_band=None):
    """Construct an allowed-id set A and the achievable positives (top-k ∩ A) within it."""
    U = list(U)
    topk = exact_rank[:k]
    candidates = rng.sample(topk, c)
    excluded = set(exact_rank[: 2 * k])
    pool = list(exact_rank[decoy_band[0]:decoy_band[1]]) if decoy_band else U
    decoy_pool = [u for u in pool if u not in excluded]
    n_decoys = max(0, target_size - c)
    decoys = rng.sample(decoy_pool, min(n_decoys, len(decoy_pool)))
    A = set(candidates) | set(decoys)
    achievable = set(candidates)
    return A, achievable


def _arm_seeker(df, *, k, k_coarse, exact_threshold, dataset, mode, ef=None):
    overrides = {"dataset": dataset, "pg_pushdown_mode": mode, "query_encode": "off"}
    if ef is not None:
        overrides["faiss_hnsw_ef_search"] = int(ef)
    return Seekers.SU(df, k=k, k_coarse=k_coarse, exact_threshold=exact_threshold,
                      config_overrides=overrides)


def _timed_arm(make_seeker, additionals, repeats, warmup=True):
    res, times = None, []
    if warmup:
        make_seeker().run(additionals)
    for _ in range(max(1, repeats)):
        seeker = make_seeker()
        t0 = perf_counter()
        res = seeker.run(additionals)
        times.append((perf_counter() - t0) * 1000.0)
    return res, median(times)


def run_pushdown_bench(dataset, *, k, selectivities, comps, modes, exact_threshold,
                       k_coarse, repeats, queries, db, seed, ef=None,
                       decoy_band=None, warmup=True):
    from src.Semantic.config import SemanticConfig, SemanticOp
    from src.Semantic.retrieve import IndexHandle

    i2b, _lake = load_sidecar(dataset)
    cfg = SemanticConfig.load(overrides={"dataset": dataset})
    oc = cfg.operator(SemanticOp.SU)
    handle = IndexHandle.open(cfg, oc.approach, oc.index_name)
    eff_kc = k_coarse if k_coarse is not None else cfg.faiss_k_coarse
    eff_ef = ef if ef is not None else cfg.faiss_hnsw_ef_search
    if cfg.vector_backend == "faiss" and eff_kc and int(eff_ef) != int(eff_kc):
        print(f"[pushdown-bench] WARNING: ef_search={eff_ef} != k_coarse={eff_kc} on "
              f"faiss; pass --ef {eff_kc} to honour the fill floor.", file=sys.stderr)
    U = sorted(handle.table_to_int_id.values())
    all_ids = sorted(U)

    rows: list[dict] = []
    warned_short = False
    skipped = {"missing_csv": 0, "unusable_query": 0}
    evaluated = 0
    for q in tqdm(queries, desc=f"pushdown/{dataset}"):
        try:
            df = load_query_table(dataset, q)
        except FileNotFoundError:
            skipped["missing_csv"] += 1
            continue
        df.attrs["table_id"] = q
        try:
            ex = _semantic_leg("su", df, len(all_ids), exact=True, k_coarse=None, dataset=dataset)
            ex.DB = db
            exact_rank = exact_semantic_ids(ex, all_ids)
        except KeyError:
            skipped["unusable_query"] += 1
            continue
        evaluated += 1

        for s in selectivities:
            for c in comps:
                target_size = max(c, round(s * len(U)))
                tag = f"{seed}|{q}|{s}|{c}".encode()
                rng = random.Random(int.from_bytes(hashlib.sha256(tag).digest()[:8], "big"))
                A, achievable = build_allowed(exact_rank, U, k=k, c=c,
                                              target_size=target_size, rng=rng,
                                              decoy_band=decoy_band)
                if len(A) < target_size and not warned_short:
                    warned_short = True
                    print(f"[pushdown-bench] WARNING: decoy_band={decoy_band} holds too "
                          f"few ids to fill |A|={target_size}; the realised selectivity "
                          f"is below the requested one (see n_allowed per row).",
                          file=sys.stderr)
                add = " AND TableId IN (" + ",".join(str(i) for i in sorted(A)) + ") "
                for mode in modes:
                    def mk(mode=mode):
                        sk = _arm_seeker(df, k=k, k_coarse=k_coarse,
                                         exact_threshold=exact_threshold,
                                         dataset=dataset, mode=mode, ef=ef)
                        sk.DB = db
                        return sk
                    returned, latency_ms = _timed_arm(mk, add, repeats, warmup)
                    rset = set(returned)
                    recall = SD.retention(returned, sorted(achievable)) if c > 0 else None
                    rows.append({
                        "query": q,
                        "mode": mode,
                        "selectivity": s,
                        "c": c,
                        "achievable": len(achievable),
                        "n_allowed": len(A),
                        "recall": recall,
                        "latency_ms": latency_ms,
                        "leakage": len(rset - A),
                        "n_returned": len(returned),
                    })

    summary = _summarize(rows, queries, dataset=dataset, k=k,
                         exact_threshold=exact_threshold, k_coarse=k_coarse, seed=seed,
                         evaluated=evaluated, skipped=skipped, ef=eff_ef,
                         decoy_band=decoy_band, warmup=warmup)
    return rows, summary


def _summarize(rows, queries, *, dataset, k, exact_threshold, k_coarse, seed,
               evaluated, skipped, ef=None, decoy_band=None, warmup=True):
    groups: dict = {}
    for r in rows:
        key = (r["mode"], r["selectivity"], r["c"])
        g = groups.setdefault(key, {"recalls": [], "latencies": []})
        if r["recall"] is not None:
            g["recalls"].append(r["recall"])
        g["latencies"].append(r["latency_ms"])
    by_group = {}
    for (mode, s, c), g in sorted(groups.items()):
        by_group[f"{mode}|s={s}|c={c}"] = {
            "mode": mode, "selectivity": s, "c": c,
            "mean_recall": round(mean(g["recalls"]), 4) if g["recalls"] else None,
            "median_latency_ms": round(median(g["latencies"]), 3) if g["latencies"] else None,
            "n_rows": len(g["latencies"]),
        }
    max_leakage = max((r["leakage"] for r in rows), default=0)
    return {
        "dataset": dataset,
        "params": {"k": k, "exact_threshold": exact_threshold, "k_coarse": k_coarse,
                   "ef_search": ef, "decoy_band": decoy_band, "seed": seed,
                   "warmup": warmup},
        "coverage": {"queries_total": len(queries), "evaluated": evaluated,
                     "rows_total": len(rows)},
        "skipped": skipped,
        "max_leakage": max_leakage,
        "leakage_ok": max_leakage == 0,
        "by_group": by_group,
    }
