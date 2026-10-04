import time
from collections import Counter as _Counter
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from src.DBHandler import DBHandler
from src.Operators.Combiners.Intersection import Intersection
from src.utils import Logger
from src.Optimizer.db import use_lake, load_freqs, lake_csv_dir, lake_dir
from src.optimizer_paths import freqs_path
from src.Optimizer import sampling as smp
from src.Optimizer.training import build_seeker


def aggregate(combiner, result_sets, k):
    if combiner == "Intersection":
        out = set.intersection(*result_sets) if result_sets else set()
        return list(out)[:k]
    if combiner == "Union":
        out = set().union(*result_sets) if result_sets else set()
        return list(out)[:k]
    if combiner == "Difference":
        out = result_sets[0].difference(*result_sets[1:])
        return list(out)[:k]
    if combiner == "Counter":
        counts = _Counter()
        for s in result_sets:
            counts.update(s)
        return [tid for tid, _ in counts.most_common(k)]
    raise ValueError(f"unknown combiner {combiner}")


@dataclass
class PlanSpec:
    seeker_type: str
    csv: str
    col_groups: list
    combiner: str
    k: int


def _make_seekers(plan_spec, lake):
    csvd = lake_csv_dir(lake)
    df = smp._read_csv(csvd / plan_spec.csv)
    seekers = []
    for group in plan_spec.col_groups:
        columns = [df[c].tolist() for c in group]
        seekers.append(build_seeker(plan_spec.seeker_type, columns, plan_spec.k))
    return seekers


def run_naive(plan_spec, lake):
    DBHandler.USE_ML_OPTIMIZER = False
    use_lake(lake)
    seekers = _make_seekers(plan_spec, lake)
    t0 = time.perf_counter()
    result_sets = [set(s.run()) for s in seekers]
    result = aggregate(plan_spec.combiner, result_sets, plan_spec.k)
    return time.perf_counter() - t0, result


def run_optimized(plan_spec, lake, use_ml):
    """Run the plan via Intersection; return (t_total, t_infer, t_query, result)."""
    DBHandler.USE_ML_OPTIMIZER = use_ml
    if use_ml and not DBHandler.frequency_dict:
        load_freqs(freqs_path(lake_dir(lake)))
    db = use_lake(lake)
    seekers = _make_seekers(plan_spec, lake)
    combiner = Intersection(plan_spec.k)
    combiner.set_inputs(seekers)
    t_infer = 0.0
    if use_ml:
        t0 = time.perf_counter()
        for s in seekers:
            s.ml_cost(db)
        t_infer = time.perf_counter() - t0
    t0 = time.perf_counter()
    result = combiner.run()
    t_query = time.perf_counter() - t0
    return t_infer + t_query, t_infer, t_query, result


def _mc_plan_specs(csv_dir, combiner, k, group_size=2):
    csv_dir = Path(csv_dir)
    specs = []
    for name in sorted(p.name for p in csv_dir.glob("*.csv")):
        try:
            df = smp._read_csv(csv_dir / name)
        except Exception:
            continue
        cols = list(df.columns)
        if len(cols) >= 2 * group_size:
            g1, g2 = cols[:group_size], cols[group_size:2 * group_size]
            specs.append(PlanSpec("MC", name, [list(g1), list(g2)], combiner, k))
    return specs


def sample_plan_specs(csv_dir, seeker_type="SC", combiner="Intersection", n=50, seed=0, k=10, max_cat_card=1000):
    if seeker_type == "MC":
        specs = _mc_plan_specs(csv_dir, combiner, k)
    else:
        cands = smp.candidates(csv_dir, seeker_type, max_cat_card=max_cat_card)
        by_csv = {}
        for c in cands:
            by_csv.setdefault(c.csv, []).append(list(c.cols))
        specs = []
        for csv, groups in sorted(by_csv.items()):
            if len(groups) >= 2:
                specs.append(PlanSpec(seeker_type, csv, groups[:2], combiner, k))
    import random
    random.Random(seed).shuffle(specs)
    return specs[:n]


def evaluate(lake, plan_specs, log_dir, clear=True):
    logger = Logger(logging_path=log_dir, clear_logs=clear)
    for i, ps in enumerate(tqdm(plan_specs, desc="eval", unit="plan")):
        t_naive, r_naive = run_naive(ps, lake)
        t_rule, _, _, r_rule = run_optimized(ps, lake, use_ml=False)
        t_ml, t_infer, t_query_ml, r_ml = run_optimized(ps, lake, use_ml=True)
        logger.log("optimizer_eval", {
            "seeker_type": ps.seeker_type, "plan_id": i, "csv": ps.csv, "combiner": ps.combiner,
            "t_naive": t_naive, "t_rule": t_rule, "t_ml": t_ml,
            "t_infer": t_infer, "t_query_ml": t_query_ml,
            "n_naive": len(r_naive), "n_rule": len(r_rule), "n_ml": len(r_ml),
        })


def summarize(log_dir):
    import pandas as pd
    df = pd.read_csv(Path(log_dir) / "optimizer_eval.csv")
    cols = ["t_naive", "t_rule", "t_ml", "t_infer", "t_query_ml"]
    g = df.groupby("seeker_type")[cols].sum().round(3)
    g["plans"] = df.groupby("seeker_type").size()
    g["ml<=naive"] = g["t_ml"] <= g["t_naive"]
    g["ml_order_vs_rule"] = (g["t_query_ml"] <= g["t_rule"])
    return g
