from __future__ import annotations

import csv
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.Benchmark.datasource import dataset_dir, load_sidecar
from src.Benchmark.prepare import guard_outputs

_SAMPLE_ROWS = 5000
_MIN_KEY_UNIQUENESS = 0.9
_MIN_KEY_DISTINCT = 10
_MIN_FD_CONFIDENCE = 0.95


def _is_stringy(s: pd.Series) -> bool:
    head = s.head(200)
    if len(head) == 0:
        return False
    return head.str.contains(r"\D", regex=True).mean() >= 0.5


def _fd_confidence(df: pd.DataFrame, key: str, val: str) -> float:
    pair = df.groupby([key, val]).size()
    return pair.groupby(level=0).max().sum() / len(df)


def _select_key_value(table_path, sample: int = _SAMPLE_ROWS, allowed=None):
    try:
        df = pd.read_csv(table_path, dtype=str, nrows=sample, keep_default_na=False)
    except Exception:
        return None
    n = len(df)
    if n < 6 or df.shape[1] < 2:
        return None
    keys = []
    for c in df.columns:
        nun = df[c].nunique()
        if nun >= _MIN_KEY_DISTINCT and nun / n >= _MIN_KEY_UNIQUENESS:
            keys.append((c, nun / n, _is_stringy(df[c])))
    if not keys:
        return None
    keys.sort(key=lambda t: (t[2], t[1]), reverse=True)
    for key, _, _ in keys:
        if allowed is not None and key not in allowed:
            continue
        key_unique = df[key].nunique() == n
        cand = []
        for c in df.columns:
            if c == key or df[c].nunique() < 2:
                continue
            conf = 1.0 if key_unique else _fd_confidence(df, key, c)
            if conf >= _MIN_FD_CONFIDENCE:
                cand.append((c, round(conf, 3), _is_stringy(df[c]), df[c].nunique()))
        if cand:
            cand.sort(key=lambda t: (t[1], t[2], t[3]), reverse=True)
            return key, cand[0][0]
    return None


def _enrolled_columns(dataset: str):
    from collections import defaultdict
    from src.Semantic.config import SemanticConfig, SemanticOp
    from src.Semantic.retrieve import IndexHandle
    cfg = SemanticConfig.load(overrides={"dataset": dataset})
    op = cfg.operator(SemanticOp.SJ)
    handle = IndexHandle.open(cfg, op.approach, op.index_name)
    out: dict[str, set[str]] = defaultdict(set)
    for table, col in handle.table_col_to_gid:
        out[table].add(col)
    return out, f"{op.approach}/{op.index_name}"


def _scan_one(item):
    name, path, allowed = item
    return name, _select_key_value(path, allowed=allowed)


def _reconstruct(table_path: Path, key_col: str, val_col: str):
    df = pd.read_csv(table_path, dtype=str, keep_default_na=False)
    df = df.dropna(subset=[key_col, val_col]) if {key_col, val_col} <= set(df.columns) else df
    examples = df.head(5)[[key_col, val_col]]
    queries = df.iloc[5:][key_col]
    return examples, queries


def prepare_imputation(dataset: str, out: Path, *, n: int, seed: int, force: bool,
                       require_semantic: bool = False) -> None:
    _, lake = load_sidecar(dataset)
    csvs = dataset_dir(dataset) / "csvs"
    enrolled, index_id = (_enrolled_columns(dataset) if require_semantic else (None, None))
    items = [(name, str(csvs / name),
              None if enrolled is None else enrolled.get(name, set()))
             for name in sorted(lake) if (csvs / name).exists()]
    selected: list[tuple[str, str, str]] = []
    with ProcessPoolExecutor() as ex:
        for name, kv in tqdm(ex.map(_scan_one, items, chunksize=8),
                             total=len(items), desc=f"impute-scan/{dataset}"):
            if kv is not None:
                selected.append((name, kv[0], kv[1]))
    rng = random.Random(seed)
    sample = rng.sample(selected, min(n, len(selected)))
    q_path, gt_path, mf_path = out / "query.csv", out / "groundtruth.csv", out / "manifest.json"
    guard_outputs([q_path, gt_path, mf_path], force)
    with q_path.open("w", newline="") as qf, gt_path.open("w", newline="") as gf:
        qw = csv.writer(qf); qw.writerow(["query_id", "source_table", "key_col", "val_col"])
        gw = csv.writer(gf); gw.writerow(["query_id", "candidate_table"])
        for i, (name, key, val) in enumerate(sample):
            qw.writerow([i, name, key, val])
            gw.writerow([i, name])
    mf_path.write_text(json.dumps(
        {"task": "imputation", "seed": seed, "n_requested": n,
         "n_scanned": len(items), "pool_size": len(selected), "n_written": len(sample),
         "require_semantic": require_semantic, "semantic_index": index_id,
         "selection": {"sample_rows": _SAMPLE_ROWS, "min_key_uniqueness": _MIN_KEY_UNIQUENESS,
                       "min_key_distinct": _MIN_KEY_DISTINCT, "min_fd_confidence": _MIN_FD_CONFIDENCE}},
        indent=2))
    gate = f", gated on {index_id}" if require_semantic else ""
    print(f"imputation: wrote {len(sample)} queries "
          f"(pool={len(selected)}/{len(items)} tables have a discriminative key→value{gate}) to {out}")
