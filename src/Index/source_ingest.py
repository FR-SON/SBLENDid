"""CSV directory -> tokenized blend_index rows, shared by the DuckDB and Postgres ingesters."""
from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
from typing import Iterator

import pandas as pd

from src.utils import df_to_index

_POOL_CHUNKSIZE = 16

_RENAME = {
    "CellValue": "tokenized",
    "TableId":   "tableid",
    "ColumnId":  "colid",
    "RowId":     "rowid",
    "SuperKey":  "super_key",
    "Quadrant":  "quadrant",
}

def list_csv_paths(csv_dir: Path) -> list[Path]:
    paths = sorted(
        (p for p in Path(csv_dir).iterdir()
         if p.suffix == ".csv" and not p.name.startswith("._")),
        key=lambda p: p.as_posix(),
    )
    if not paths:
        raise SystemExit(f"no csv files under {csv_dir}")
    return paths


def build_basenames_df(csv_paths: list[Path]) -> pd.DataFrame:
    return pd.DataFrame(
        [(tid, p.name) for tid, p in enumerate(csv_paths)],
        columns=["table_int_id", "basename"],
    ).astype({"table_int_id": "int64", "basename": "string"})


def build_jobs(csv_paths: list[Path]) -> list[tuple[int, str]]:
    """Assign TableIds once, in sorted-basename order; --resume filters this list without renumbering."""
    return [(tid, str(p)) for tid, p in enumerate(csv_paths)]


def read_or_write_sidecar(basenames_df: pd.DataFrame, sidecar_path: Path,
                          *, resume: bool) -> None:
    sidecar_path = Path(sidecar_path)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    if resume:
        if not sidecar_path.exists():
            raise SystemExit(
                f"--resume needs sidecar from prior run at {sidecar_path}; "
                f"re-run without --resume to start fresh")
        existing = pd.read_parquet(sidecar_path)
        if not existing.reset_index(drop=True).equals(basenames_df):
            raise SystemExit(
                f"csv_dir basenames drifted vs sidecar at {sidecar_path}; "
                f"TableIds would shift. Re-run without --resume.")
    else:
        basenames_df.to_parquet(sidecar_path, index=False)


def tokenize_one(job: tuple[int, str]) -> pd.DataFrame:
    tid, path_str = job
    df = pd.read_csv(path_str, low_memory=False)
    df.columns.name = str(tid)
    rows = df_to_index(df)
    return rows.rename(columns=_RENAME)


def iter_tokenized_tables(jobs: list[tuple[int, str]], *, workers: int) -> Iterator[pd.DataFrame]:
    """Yield tokenized rows for pre-assigned (tableid, path) jobs from build_jobs()."""
    if workers <= 1:
        for j in jobs:
            yield tokenize_one(j)
        return
    pool = multiprocessing.Pool(processes=workers)
    try:
        for rows in pool.imap_unordered(tokenize_one, jobs, chunksize=_POOL_CHUNKSIZE):
            yield rows
    finally:
        pool.close()
        pool.join()


def default_workers() -> int:
    cpu = os.cpu_count() or 1
    return max(1, cpu // 2)
