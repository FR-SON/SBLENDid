import sys
from pathlib import Path

import duckdb

from src.Optimizer.db import resolve_scan_table


def require_duckdb_slice(dbms: str, stage: str) -> None:
    """Hard-fail the slice path on non-duckdb backends."""
    if dbms != "duckdb":
        raise SystemExit(
            f"{stage}: needs [Database] dbms = duckdb (config says {dbms}). "
            "The rowid<256 slice is a duckdb file (ATTACH + CREATE TABLE), and "
            "DBHandler's postgres branch ignores db_filename — the flag would "
            "silently measure the full lake instead.")


def warn_rowid_slice(stage: str) -> None:
    print(
        f"[OPTIMIZER][ROWID-SLICE] {stage}: C is timed against the rowid<256 "
        "slice (~20x smaller, freshly sorted) while every other seeker type is "
        "timed against the full lake, so the trained C model underprices C and "
        "its ml_cost is not comparable to the other types. Smoke runs only — "
        "drop the flag for anything reported; cut --n or --repeats instead.",
        file=sys.stderr, flush=True)


def build_rowid_slice(src_duckdb_path, out_path, index_table="blend_index", max_rowid=256) -> Path:
    """Materialize index rows with rowid < max_rowid into a standalone duckdb file."""
    out_path = Path(out_path)
    src_con = duckdb.connect(str(src_duckdb_path), read_only=True)
    resolved = resolve_scan_table(
        (r[0] for r in src_con.execute("SELECT table_name FROM information_schema.tables").fetchall()),
        index_table,
        prefer="tableid",
    )
    src_con.close()
    if out_path.exists():
        out_path.unlink()
    con = duckdb.connect(str(out_path))
    con.execute("PRAGMA enable_progress_bar")
    con.execute(f"ATTACH '{src_duckdb_path}' AS src (READ_ONLY)")
    con.execute(
        f"CREATE TABLE {index_table} AS "
        f"SELECT * FROM src.{resolved} WHERE rowid < {max_rowid} "
        f"ORDER BY tableid, colid, rowid"
    )
    con.execute("DETACH src")
    con.close()
    return out_path
