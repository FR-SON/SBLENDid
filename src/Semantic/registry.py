"""Column registry: (table_id, col_idx) → global_id mapping."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd


@dataclass(frozen=True)
class ColumnRef:
    global_id: int
    table_id: str
    col_idx: int
    col_name: str | None
    n_rows: int
    source_path: str


@dataclass(frozen=True)
class LakeTableEntry:
    """Input to build_registry: one lake table to enroll."""
    table_path: Path
    source_path: str


_REGISTRY_COLUMNS = [
    "global_id", "table_id", "col_idx", "col_name", "n_rows", "source_path",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_ckpt(path: Path) -> str:
    """Hash a ckpt file, or model.safetensors inside an HF-dir ckpt."""
    if path.is_dir():
        target = path / "model.safetensors"
        if not target.is_file():
            raise FileNotFoundError(
                f"ckpt dir {path} has no model.safetensors to hash"
            )
        return sha256_file(target)
    return sha256_file(path)


def build_registry(entries: list[LakeTableEntry], out_path: Path) -> int:
    """Build a registry.parquet from a list of LakeTableEntry. Returns column count N."""
    import sys

    from tqdm import tqdm

    rows: list[dict] = []
    gid = 0
    for entry in tqdm(entries, desc="registry", unit="table", file=sys.stderr):
        df = pd.read_csv(entry.table_path, dtype=str, keep_default_na=False)
        n_rows = len(df)
        table_id = entry.table_path.name
        for col_idx, col_name in enumerate(df.columns):
            rows.append({
                "global_id": gid,
                "table_id": table_id,
                "col_idx": col_idx,
                "col_name": col_name,
                "n_rows": n_rows,
                "source_path": entry.source_path,
            })
            gid += 1

    out_df = pd.DataFrame(rows, columns=_REGISTRY_COLUMNS)
    out_df = out_df.astype({
        "global_id": "int64",
        "col_idx": "int32",
        "n_rows": "int32",
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    return len(out_df)


def load_registry(path: Path) -> Iterator[ColumnRef]:
    """Yield ColumnRef per row, ordered by global_id ascending."""
    df = pd.read_parquet(path).sort_values("global_id")
    for row in df.itertuples(index=False):
        yield ColumnRef(
            global_id=int(row.global_id),
            table_id=str(row.table_id),
            col_idx=int(row.col_idx),
            col_name=None if pd.isna(row.col_name) else str(row.col_name),
            n_rows=int(row.n_rows),
            source_path=str(row.source_path),
        )


def hash_registry(path: Path) -> str:
    """Return 'sha256:<hex>' of the registry parquet file."""
    return f"sha256:{sha256_file(path)}"
