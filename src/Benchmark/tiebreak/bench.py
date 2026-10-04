"""SU∩SU / SJ∩SJ tiebreak ablation: k_coarse-aware fitted cost vs naive n_cols ordering."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from statistics import median
from time import perf_counter

from tqdm import tqdm

from src.Operators import Seekers


def _cmp(a: float, b: float) -> str:
    if a == b:
        return "tie"
    return "A" if a < b else "B"


def classify_pair(a: dict, b: dict) -> dict:
    """Classify the naive, fitted and ground-truth run-first orderings of one same-op pair."""
    gt = _cmp(a["gt_ms"], b["gt_ms"])
    placeholder = _cmp(a["n_cols"], b["n_cols"])
    new = _cmp(a["pred_ms"], b["pred_ms"])
    placeholder_correct = placeholder == gt
    new_correct = new == gt
    changed = new != placeholder
    return {
        "gt": gt, "placeholder": placeholder, "new": new,
        "placeholder_correct": placeholder_correct, "new_correct": new_correct,
        "changed": changed,
        "corrected": changed and new_correct and not placeholder_correct,
    }


def _all_pairs(n: int):
    for i in range(n):
        for j in range(i + 1, n):
            yield i, j


def _sample_pairs(n: int, want: int, seed: int):
    import random
    rng = random.Random(seed)
    seen: set[tuple[int, int]] = set()
    while len(seen) < want:
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            continue
        seen.add((i, j) if i < j else (j, i))
    return sorted(seen)


def _measure_ms(make_seeker, db, repeats: int) -> float:
    times = []
    for _ in range(max(1, repeats)):
        sk = make_seeker()
        t0 = perf_counter()
        sk.create_sql_query(db, additionals="")
        times.append((perf_counter() - t0) * 1000.0)
    return median(times)


def _iter_query_items(cfg, dataset, queries, sample_queries, seed, op, skipped):
    """Yield (label, df) one query leg at a time; a generator so the corpus never sits in RAM."""
    from src.Benchmark.datasource import load_query_table
    from src.Semantic.config import SemanticOp
    sop = SemanticOp(op)
    if sample_queries is not None:
        from src.Semantic.retrieve import IndexHandle
        from src.Optimizer.semantic_sweep import (sample_su_queries, sample_sj_queries,
                                                  query_df_from_index)
        op_cfg = cfg.operator(sop)
        handle = IndexHandle.open(cfg, op_cfg.approach, op_cfg.index_name)
        if sop is SemanticOp.SU:
            for qr in sample_su_queries(cfg, sample_queries, seed):
                yield qr.table_id, query_df_from_index(handle, qr.table_id)
        else:
            for qr in sample_sj_queries(cfg, sample_queries, seed):
                df = query_df_from_index(handle, qr.table_id)[[qr.col_name]]
                df.attrs["table_id"] = qr.table_id
                yield f"{qr.table_id}|{qr.col_name}", df
        return
    for q in queries:
        try:
            df = load_query_table(dataset, q)
        except FileNotFoundError:
            skipped["missing_csv"] += 1
            continue
        df.attrs["table_id"] = q
        if sop is SemanticOp.SJ:
            for col in df.columns:
                sub = df[[col]].copy()
                sub.attrs["table_id"] = q
                yield f"{q}|{col}", sub
        else:
            yield q, df


def _arm_overrides(dataset, cfg, kc: int) -> dict:
    """Config for one (n_cols, k_coarse) arm; on faiss the beam follows the fetch."""
    ov = {"dataset": dataset, "query_encode": "off"}
    if cfg.vector_backend == "faiss":
        ov["faiss_hnsw_ef_search"] = int(kc)
    return ov


def run_tiebreak_bench(dataset, *, k, k_coarse_grid, queries, db, repeats, seed,
                       sample_queries=None, max_pairs=200_000, op="SU"):
    from src.Semantic.config import SemanticConfig
    from src.Semantic.cost_predict import cost_unfiltered

    cfg = SemanticConfig.load(overrides={"dataset": dataset})
    dsdir = cfg.dataset.dir()
    _pfn = getattr(db, "optimizer_profile_str", None)
    if callable(_pfn):
        profile = _pfn()
    else:
        from src.Benchmark.db import open_dataset_db
        _pdb = open_dataset_db(dataset)
        try:
            profile = _pdb.optimizer_profile_str()
        finally:
            _pdb.close()

    skipped = {"missing_csv": 0, "unusable_query": 0}
    items = _iter_query_items(cfg, dataset, queries, sample_queries, seed, op, skipped)
    total = len(queries) if (sample_queries is None and op == "SU") else sample_queries

    configs: list[dict] = []
    labels: list[str] = []
    for label, df in tqdm(items, total=total, unit="leg",
                          desc=f"tiebreak/{dataset}/{op} measure"):
        labels.append(label)
        n_cols = df.shape[1]
        for kc in k_coarse_grid:
            def mk(df=df, kc=kc):
                ov = _arm_overrides(dataset, cfg, kc)
                if op == "SJ":
                    sk = Seekers.SJ(df, k=k, k_coarse=kc, config_overrides=ov,
                                    query_col_name=str(df.columns[0]))
                else:
                    sk = Seekers.SU(df, k=k, k_coarse=kc, config_overrides=ov)
                sk.DB = db
                return sk
            try:
                gt_ms = _measure_ms(mk, db, repeats)
            except KeyError:
                skipped["unusable_query"] += 1
                break
            configs.append({
                "query": label, "n_cols": n_cols, "k_coarse": kc,
                "gt_ms": gt_ms,
                "pred_ms": cost_unfiltered(op, n_cols, kc, dataset_dir=dsdir, profile=profile),
            })

    rows = []
    n_cfg = len(configs)
    total_pairs = n_cfg * (n_cfg - 1) // 2
    pair_iter = _all_pairs(n_cfg)
    if max_pairs and total_pairs > max_pairs:
        print(f"[tiebreak] {total_pairs:,} pairs from {n_cfg:,} configs exceeds "
              f"--max-pairs {max_pairs:,}; sampling {max_pairs:,} uniformly "
              f"(seed={seed}). Every reported statistic is over that sample.",
              flush=True)
        pair_iter = _sample_pairs(n_cfg, max_pairs, seed)
    for i, j in pair_iter:
        a, b = configs[i], configs[j]
        v = classify_pair(a, b)
        rows.append({
            **v,
            "a": f"{a['query']}|nc={a['n_cols']}|kc={a['k_coarse']}",
            "b": f"{b['query']}|nc={b['n_cols']}|kc={b['k_coarse']}",
            "a_gt_ms": round(a["gt_ms"], 4), "b_gt_ms": round(b["gt_ms"], 4),
            "a_pred_ms": round(a["pred_ms"], 4), "b_pred_ms": round(b["pred_ms"], 4),
        })
    summary_queries = queries if sample_queries is None else labels
    summary = summarize(rows, dataset, queries=summary_queries, configs=configs,
                        skipped=skipped, k=k, k_coarse_grid=k_coarse_grid,
                        repeats=repeats, seed=seed, pairs_total=total_pairs, op=op,
                        beam_rule=("ef=k_coarse" if cfg.vector_backend == "faiss"
                                   else f"ef={cfg.faiss_hnsw_ef_search}"))
    return rows, summary


def _regret(rows, arm: str, correct_key: str) -> dict:
    total = oracle = 0.0
    wrong_gaps = []
    for r in rows:
        gap = abs(r["a_gt_ms"] - r["b_gt_ms"])
        oracle += min(r["a_gt_ms"], r["b_gt_ms"])
        if r[arm] == "tie":
            total += gap / 2.0
        elif not r[correct_key]:
            total += gap
            wrong_gaps.append(gap)
    wrong_gaps.sort()
    return {
        "regret_ms": round(total, 3),
        "regret_vs_oracle_pct": round(total / oracle * 100, 4) if oracle else None,
        "mean_gap_when_wrong_ms": (round(sum(wrong_gaps) / len(wrong_gaps), 4)
                                   if wrong_gaps else 0.0),
        "median_gap_when_wrong_ms": (round(wrong_gaps[len(wrong_gaps) // 2], 4)
                                     if wrong_gaps else 0.0),
    }


def summarize(rows, dataset, *, queries, configs, skipped, k, k_coarse_grid,
              repeats, seed, beam_rule="", pairs_total=None, op="SU") -> dict:
    inversions = sum(1 for r in rows
                     if r["placeholder"] in ("A", "B") and not r["placeholder_correct"])
    gaps = sorted(abs(r["a_gt_ms"] - r["b_gt_ms"]) for r in rows)
    coin = sum(gaps) / 2.0
    oracle_total = sum(min(r["a_gt_ms"], r["b_gt_ms"]) for r in rows)
    return {
        "dataset": dataset,
        "op": op,
        "params": {"k": k, "k_coarse_grid": list(k_coarse_grid),
                   "beam_rule": beam_rule, "repeats": repeats, "seed": seed},
        "coverage": {"queries_total": len(queries), "configs": len(configs),
                     "pairs": len(rows),
                     "pairs_total": pairs_total if pairs_total is not None else len(rows),
                     "skipped": skipped},
        "placeholder_ties": sum(1 for r in rows if r["placeholder"] == "tie"),
        "placeholder_inversions": inversions,
        "placeholder_correct": sum(1 for r in rows if r["placeholder_correct"]),
        "new_correct": sum(1 for r in rows if r["new_correct"]),
        "decisions_changed": sum(1 for r in rows if r["changed"]),
        "decisions_corrected": sum(1 for r in rows if r["corrected"]),
        "decisions_regressed": sum(1 for r in rows
                                   if r["placeholder_correct"] and not r["new_correct"]),
        "regret": {"placeholder": _regret(rows, "placeholder", "placeholder_correct"),
                   "new": _regret(rows, "new", "new_correct")},
        "median_gap_ms": round(gaps[len(gaps) // 2], 4) if gaps else 0.0,
        "mean_gap_ms": round(sum(gaps) / len(gaps), 4) if gaps else 0.0,
        "coin_flip_regret_ms": round(coin, 3),
        "coin_flip_regret_vs_oracle_pct": (round(coin / oracle_total * 100, 4)
                                           if oracle_total else None),
    }
