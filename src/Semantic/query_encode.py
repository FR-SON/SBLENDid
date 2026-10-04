"""Encode query columns that are not in the semantic index from their cells."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .registry import ColumnRef


def _check_unique_labels(df: pd.DataFrame) -> None:
    labels = [str(c) for c in df.columns]
    dupes = sorted({c for c in labels if labels.count(c) > 1})
    if dupes:
        raise ValueError(f"query DataFrame has duplicate column labels: {dupes}")


def cells_for(encoder, series: pd.Series) -> list[str]:
    """Cell strings as the encoder's index build saw them: NaN becomes `encoder.na_cell`."""
    return series.fillna(encoder.na_cell).astype(str).tolist()


def empty_columns(df: pd.DataFrame, cols: list) -> list:
    """Columns of `cols` whose every cell is NaN or whitespace-only."""
    _check_unique_labels(df)
    out = []
    for col in cols:
        s = df[col]
        blank = s.isna() | s.astype(str).str.strip().eq("")
        if bool(blank.all()):
            out.append(col)
    return out


def query_refs(table_id: str, df: pd.DataFrame, cols: list) -> list[ColumnRef]:
    """Synthetic registry rows for query columns; global_id is the position in `cols`."""
    return [
        ColumnRef(
            global_id=i, table_id=table_id, col_idx=int(df.columns.get_loc(col)),
            col_name=str(col), n_rows=int(len(df)), source_path=table_id,
        )
        for i, col in enumerate(cols)
    ]


def encode_query_columns(encoder, table_id: str, df: pd.DataFrame, cols: list) -> dict[str, np.ndarray]:
    """{str(col): fp32 vector} for `cols`; one encode_batch call when the encoder has it."""
    if not cols:
        return {}
    _check_unique_labels(df)
    refs = query_refs(table_id, df, cols)
    cells = {r.global_id: cells_for(encoder, df[cols[r.global_id]]) for r in refs}
    batch = getattr(encoder, "encode_batch", None)
    if batch is not None:
        by_gid = batch(refs, cells_per_ref=cells)
    else:
        by_gid = {r.global_id: encoder.encode_one(r, cells=cells[r.global_id]) for r in refs}
    return {str(cols[g]): np.asarray(v, dtype=np.float32) for g, v in by_gid.items()}


__all__ = ["cells_for", "empty_columns", "query_refs", "encode_query_columns"]
