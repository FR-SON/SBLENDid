import json
import random
from dataclasses import dataclass, asdict, replace
from pathlib import Path

import pandas as pd
from tqdm import tqdm

SEEKER_TYPES = ("SC", "Keyword", "C", "MC")
CANDIDATE_ALIAS = {"Keyword": "SC"}
_STRATA = [10, 100, 1000, 10000]
_MAX_MC_COLS = 4


@dataclass(frozen=True)
class QuerySpec:
    seeker_type: str
    csv: str
    cols: tuple
    cardinality: int


def _read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, on_bad_lines="skip")


def _is_numeric(series):
    coerced = pd.to_numeric(series, errors="coerce")
    return coerced.notna().mean() >= 0.9


def candidates(csv_dir, seeker_type, max_files=None, max_cat_card=1000):
    csv_dir = Path(csv_dir)
    files = sorted(p.name for p in csv_dir.glob("*.csv"))
    if max_files is not None:
        files = files[:max_files]
    out = []
    for name in tqdm(files, desc=f"scan {seeker_type}", unit="csv"):
        try:
            df = _read_csv(csv_dir / name)
        except Exception:
            continue
        if df.shape[1] == 0 or len(df) == 0:
            continue
        if seeker_type in ("SC", "Keyword"):
            for c in df.columns:
                out.append(QuerySpec(seeker_type, name, (c,), int(df[c].nunique())))
        elif seeker_type == "C":
            numeric = [c for c in df.columns if _is_numeric(df[c])]
            nun = {c: int(df[c].nunique()) for c in df.columns if c not in numeric}
            categorical = [c for c in df.columns
                           if c not in numeric and nun[c] <= max_cat_card]
            for cat in categorical:
                for num in numeric:
                    out.append(QuerySpec(seeker_type, name, (cat, num), nun[cat]))
        elif seeker_type == "MC":
            cols = list(df.columns)
            if len(cols) >= 2:
                pick = tuple(cols[: min(_MAX_MC_COLS, len(cols))])
                card = int(df[list(pick)].drop_duplicates().shape[0])
                out.append(QuerySpec(seeker_type, name, pick, card))
        else:
            raise ValueError(f"unknown seeker_type {seeker_type}")
    return out


def _stratum(card):
    for i, edge in enumerate(_STRATA):
        if card < edge:
            return i
    return len(_STRATA)


def stratified_sample(cands, n, seed):
    rng = random.Random(seed)
    buckets = {}
    for c in cands:
        buckets.setdefault(_stratum(c.cardinality), []).append(c)
    for b in buckets.values():
        b.sort(key=lambda c: (c.csv, c.cols))
        rng.shuffle(b)
    picked, order = [], sorted(buckets)
    while len(picked) < n and any(buckets[b] for b in order):
        for b in order:
            if buckets[b] and len(picked) < n:
                picked.append(buckets[b].pop())
    return picked


def sample_queries(csv_dir, seeker_type, n=1000, seed=0, max_files=None, max_cat_card=1000):
    return stratified_sample(candidates(csv_dir, seeker_type, max_files, max_cat_card), n, seed)


def sample_many(csv_dir, seeker_types, *, n_by_type, n_default, seed=0,
                max_files=None, max_cat_card=1000):
    """Return {seeker_type: specs}, scanning the lake once per distinct candidate construction."""
    pools: dict[str, list] = {}
    out: dict[str, list] = {}
    for t in seeker_types:
        base = CANDIDATE_ALIAS.get(t, t)
        if base not in pools:
            pools[base] = candidates(csv_dir, base, max_files, max_cat_card)
        pool = pools[base]
        if base != t:
            pool = [replace(c, seeker_type=t) for c in pool]
        out[t] = stratified_sample(pool, n_by_type.get(t, n_default), seed)
    return out


def load_columns(csv_dir, spec):
    df = _read_csv(Path(csv_dir) / spec.csv)
    return [df[c].tolist() for c in spec.cols]


def save_specs(specs, path):
    with open(path, "w") as f:
        for s in specs:
            f.write(json.dumps(asdict(s)) + "\n")


def load_specs(path):
    out = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            out.append(QuerySpec(d["seeker_type"], d["csv"], tuple(d["cols"]), d["cardinality"]))
    return out
