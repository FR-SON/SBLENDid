from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import pandas as pd

from src.Benchmark.datasource import load_query_table, load_sidecar
from src.Benchmark.prepare import guard_outputs


def _key_to_target(df: pd.DataFrame, key: str, target: str) -> pd.Series:
    t = pd.to_numeric(df[target], errors="coerce")
    return t.groupby(df[key]).mean().dropna()


def _spearman_abs(left: pd.Series, right: pd.Series) -> float:
    joined = pd.concat([left, right], axis=1, join="inner").dropna()
    if len(joined) < 3:
        return 0.0
    if joined.iloc[:, 0].nunique() < 2 or joined.iloc[:, 1].nunique() < 2:
        return 0.0
    rho = joined.iloc[:, 0].corr(joined.iloc[:, 1], method="spearman")
    return abs(rho) if pd.notna(rho) else 0.0


def _numeric_cols(df: pd.DataFrame, key: str) -> list[str]:
    out = []
    for c in df.columns:
        if c == key:
            continue
        if pd.to_numeric(df[c], errors="coerce").notna().sum() >= 3:
            out.append(c)
    return out


def prepare_correlation(dataset: str, out: Path, *, n: int, k: int, seed: int, force: bool) -> None:
    _, lake = load_sidecar(dataset)
    rng = random.Random(seed)
    lake_list = sorted(lake)

    triples = []
    for name in rng.sample(lake_list, min(len(lake_list), n * 5)):
        df = load_query_table(dataset, name)
        nums = _numeric_cols(df, df.columns[0])
        if df.shape[1] >= 2 and nums:
            triples.append((name, df.columns[0], nums[0]))
        if len(triples) >= n:
            break

    q_path, gt_path, mf_path = out / "query.csv", out / "groundtruth.csv", out / "manifest.json"
    guard_outputs([q_path, gt_path, mf_path], force)
    with q_path.open("w", newline="") as qf, gt_path.open("w", newline="") as gf:
        qw = csv.writer(qf); qw.writerow(["query_id", "source_table", "key_col", "target_col"])
        gw = csv.writer(gf); gw.writerow(["query_id", "candidate_table"])
        for i, (src, key, tgt) in enumerate(triples):
            left = _key_to_target(load_query_table(dataset, src), key, tgt)
            scored: list[tuple[float, str]] = []
            for cand in lake_list:
                cdf = load_query_table(dataset, cand)
                ckey = cdf.columns[0]
                for col in _numeric_cols(cdf, ckey):
                    right = _key_to_target(cdf, ckey, col)
                    rho = _spearman_abs(left, right)
                    if rho > 0:
                        scored.append((rho, cand))
            scored.sort(reverse=True)
            top_tables = list(dict.fromkeys(c for _, c in scored[:k]))
            qw.writerow([i, src, key, tgt])
            for cand in top_tables:
                gw.writerow([i, cand])
    mf_path.write_text(json.dumps(
        {"task": "correlation", "seed": seed, "k": k, "n_written": len(triples),
         "correlation": "spearman", "note": "synthetic GT; NYC unavailable"}, indent=2))
    print(f"correlation: wrote {len(triples)} queries to {out}")
