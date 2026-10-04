from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src import paths


def dataset_dir(dataset: str) -> Path:
    d = paths.datasets_root() / dataset
    if not d.is_dir():
        raise FileNotFoundError(f"no dataset dir at {d}")
    return d


def _one(dataset: str, subdir: str, suffix: str) -> Path:
    matches = sorted((dataset_dir(dataset) / subdir).glob(f"*{suffix}"))
    if not matches:
        raise FileNotFoundError(
            f"dataset {dataset!r}: no '*{suffix}' under {subdir}/ "
            f"(run `bench prepare` or place the benchmark files there first)")
    return matches[0]


def load_sidecar(dataset: str) -> tuple[dict[int, str], set[str]]:
    sc = pd.read_parquet(dataset_dir(dataset) / "blend_index_basenames.parquet")
    i2b = dict(zip(sc["table_int_id"].astype(int), sc["basename"].astype(str)))
    return i2b, set(i2b.values())


def to_basenames(i2b: dict[int, str], ids, dataset: str) -> list[str]:
    """Blend int TableIds -> basenames; a miss names both datasets."""
    try:
        return [i2b[int(i)] for i in ids]
    except KeyError as e:
        raise KeyError(
            f"TableId {e.args[0]} is not in the {dataset!r} basenames sidecar; "
            "the seeker's index was built for a different lake"
        ) from None


def load_union_queries(dataset: str) -> list[str]:
    path = _one(dataset, "query", "_union_query.csv")
    with path.open(newline="") as f:
        return [row["query_table"].strip() for row in csv.DictReader(f)
                if row.get("query_table", "").strip()]


def load_union_gt(dataset: str) -> dict[str, set[str]]:
    path = _one(dataset, "groundtruth", "_union_ground_truth.csv")
    gt: dict[str, set[str]] = defaultdict(set)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            q, c = row["query_table"].strip(), row["candidate_table"].strip()
            if q and c and c != q:
                gt[q].add(c)
    return dict(gt)


def load_join_queries(dataset: str) -> list[tuple[str, str]]:
    path = _one(dataset, "query", "_join_query.csv")
    with path.open(newline="") as f:
        return [(row["query_table"].strip(), row["query_column"].strip())
                for row in csv.DictReader(f)
                if row.get("query_table", "").strip()]


def load_join_gt(dataset: str) -> dict[tuple[str, str], set[str]]:
    path = _one(dataset, "groundtruth", "_join_ground_truth.csv")
    gt: dict[tuple[str, str], set[str]] = defaultdict(set)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (row["query_table"].strip(), row["query_column"].strip())
            c = row["candidate_table"].strip()
            if key[0] and c and c != key[0]:
                gt[key].add(c)
    return dict(gt)


def load_query_table(dataset: str, basename: str, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(dataset_dir(dataset) / "csvs" / basename,
                       dtype=str, keep_default_na=False, nrows=nrows)
