"""Shared non-numeric column universe (from SHO's codes.parquet) for cross-seeker benches."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.Semantic.config import SemanticConfig, SemanticOp


@dataclass
class NonNumericUniverse:
    by_name: dict[str, set[str]]
    pairs_int: set[tuple[int, int]]
    int_to_basename: dict[int, str]
    table_col_to_name: dict[tuple[str, int], str]


def load_sho_nonnumeric(
    cfg: SemanticConfig, out_dir: Path | None = None
) -> NonNumericUniverse:
    """Read the SHO index's held-column universe from codes.parquet."""
    if out_dir is None:
        op = cfg.operator(SemanticOp.SHO)
        out_dir = cfg.approach_dir(op.approach, op.index_name)
    codes_path = out_dir / "codes.parquet"
    if not codes_path.is_file():
        raise FileNotFoundError(
            f"SHO codes.parquet not found: {codes_path} — build the SHO index "
            "(scripts/create_sho_index.py) before running --nonnumeric-only")

    import duckdb

    sidecar = pd.read_parquet(cfg.blend_basenames_path)
    int_to_basename = dict(
        zip(sidecar["table_int_id"].astype(int).tolist(),
            sidecar["basename"].astype(str).tolist())
    )
    pairs = duckdb.sql(
        f"SELECT DISTINCT tableid, colid FROM read_parquet('{codes_path.as_posix()}')"
    ).df()

    csvs_dir = cfg.dataset.dir() / "csvs"
    headers: dict[str, list[str]] = {}
    by_name: dict[str, set[str]] = {}
    pairs_int: set[tuple[int, int]] = set()
    table_col_to_name: dict[tuple[str, int], str] = {}
    for tid, cid in zip(pairs["tableid"].astype(int).tolist(),
                        pairs["colid"].astype(int).tolist()):
        tid, cid = int(tid), int(cid)
        basename = int_to_basename.get(tid)
        if basename is None:
            continue
        if basename not in headers:
            headers[basename] = pd.read_csv(
                csvs_dir / basename, dtype=str, keep_default_na=False, nrows=0
            ).columns.tolist()
        cols = headers[basename]
        if 0 <= cid < len(cols):
            cname = cols[cid]
            by_name.setdefault(basename, set()).add(cname)
            table_col_to_name[(basename, cid)] = cname
            pairs_int.add((tid, cid))
    return NonNumericUniverse(by_name, pairs_int, int_to_basename, table_col_to_name)


def intersect_reg(
    reg: dict[str, set[str]], allow: dict[str, set[str]]
) -> dict[str, set[str]]:
    """Registry restricted to the non-numeric universe; empty tables dropped."""
    out = {t: (cols & allow.get(t, set())) for t, cols in reg.items()}
    return {t: cols for t, cols in out.items() if cols}
