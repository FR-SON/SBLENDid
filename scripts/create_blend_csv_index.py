"""Ingest a flat directory of CSVs into a Blend DuckDB index plus the basenames sidecar."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd
from tqdm import tqdm

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

import duckdb

from src.Index.source_ingest import (
    build_basenames_df,
    build_jobs,
    default_workers,
    iter_tokenized_tables,
    list_csv_paths,
    read_or_write_sidecar,
)

_DROP_TABLE_SQL = "DROP TABLE IF EXISTS {table}"
_CREATE_TABLE_SQL = """
CREATE TABLE {table} (
    tokenized VARCHAR,
    tableid   INTEGER,
    colid     INTEGER,
    rowid     INTEGER,
    super_key VARCHAR,
    quadrant  BOOLEAN
)
"""
_CREATE_TABLE_IF_NOT_EXISTS_SQL = _CREATE_TABLE_SQL.replace(
    "CREATE TABLE", "CREATE TABLE IF NOT EXISTS", 1
)

_CACHE_ROWS_LIMIT = 100_000

# DuckDB keeps one physical order per table (Vertica kept both projections); pick one at ingest.
_SORT_KEYS = {
    "tableid": ("tableid", "colid", "rowid"),
    "token": ("tokenized", "tableid", "colid", "rowid"),
}


def sort_index_table(
    con: "duckdb.DuckDBPyConnection",
    table_name: str = "blend_index",
    *,
    sort_by: str = "tableid",
    memory_limit: str = "8GB",
    temp_dir: str | None = None,
    threads: int = 2,
) -> None:
    """Physically sort ``table_name`` by ``_SORT_KEYS[sort_by]`` via CTAS + DROP + RENAME."""
    order_clause = ", ".join(_SORT_KEYS[sort_by])
    tmp_name = f"{table_name}__sorted_tmp"
    con.execute(f"SET threads = {threads}")
    con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute("SET preserve_insertion_order = false")
    if temp_dir:
        td = Path(temp_dir).expanduser().resolve()
        td.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory = '{td}'")
    con.execute("SET enable_progress_bar = true")
    con.execute("SET progress_bar_time = 500")

    con.execute(f"DROP TABLE IF EXISTS {tmp_name}")
    try:
        con.execute(
            f"CREATE TABLE {tmp_name} AS "
            f"SELECT * FROM {table_name} ORDER BY {order_clause}"
        )
        con.execute(f"DROP TABLE {table_name}")
        con.execute(f"ALTER TABLE {tmp_name} RENAME TO {table_name}")
    except BaseException:
        try:
            con.execute(f"DROP TABLE IF EXISTS {tmp_name}")
        except Exception:
            pass
        raise


def materialize_two_table(
    con: "duckdb.DuckDBPyConnection",
    base: str = "blend_index",
    *,
    memory_limit: str = "8GB",
    temp_dir: str | None = None,
    threads: int = 2,
) -> None:
    """Build <base>_token and <base>_tableid from <base>, drop <base>, assert parity."""
    con.execute(f"SET threads = {threads}")
    con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute("SET preserve_insertion_order = false")
    if temp_dir:
        td = Path(temp_dir).expanduser().resolve()
        td.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory = '{td}'")
    orders = {
        "token": "tokenized, tableid, colid, rowid",
        "tableid": "tableid, colid, rowid",
    }
    for suffix, order in orders.items():
        con.execute(f"DROP TABLE IF EXISTS {base}_{suffix}")
        con.execute(
            f"CREATE TABLE {base}_{suffix} AS SELECT * FROM {base} ORDER BY {order}"
        )
    con.execute(f"DROP TABLE {base}")
    cols = "tokenized, tableid, colid, rowid, super_key, quadrant"
    rows = {
        s: con.execute(f"SELECT count(*) FROM {base}_{s}").fetchone()[0] for s in orders
    }
    hashes = {
        s: con.execute(f"SELECT bit_xor(hash({cols})) FROM {base}_{s}").fetchone()[0]
        for s in orders
    }
    if rows["token"] != rows["tableid"] or hashes["token"] != hashes["tableid"]:
        raise RuntimeError(f"two_table parity failed: rows={rows} hashes={hashes}")
    desc = con.execute(
        f"SELECT count(*) FROM (SELECT tableid, lag(tableid) OVER () p FROM {base}_tableid) WHERE tableid < p"
    ).fetchone()[0]
    if desc != 0:
        raise RuntimeError(f"{base}_tableid not tableid-clustered: {desc} descents")


def ingest_csv_dir(
    csv_dir: Path,
    duckdb_path: Path,
    *,
    table_name: str = "blend_index",
    sidecar_path: Path | None = None,
    workers: int = 1,
    resume: bool = False,
    layout: str = "single",
    sort_by: str = "tableid",
    sort_memory_limit: str = "8GB",
    sort_temp_dir: str | None = None,
    sort_threads: int = 2,
) -> tuple[Path, int]:
    """Ingest csv_dir into duckdb_path; return (sidecar_path, n_tables)."""
    csv_paths = list_csv_paths(csv_dir)
    jobs: list[tuple[int, str]] = build_jobs(csv_paths)
    basenames_df = build_basenames_df(csv_paths)

    if sidecar_path is None:
        from src.Semantic.config import SemanticConfig

        sidecar_path = duckdb_path.parent / SemanticConfig().sidecar_filename

    duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)

    read_or_write_sidecar(basenames_df, sidecar_path, resume=resume)

    con = duckdb.connect(str(duckdb_path), read_only=False)
    try:
        if resume:
            con.execute(_CREATE_TABLE_IF_NOT_EXISTS_SQL.format(table=table_name))
            done_ids = {
                row[0]
                for row in con.execute(
                    f"SELECT DISTINCT tableid FROM {table_name}"
                ).fetchall()
            }
            jobs = [(tid, p) for tid, p in jobs if tid not in done_ids]
            print(
                f"resume: {len(done_ids)} tids already ingested, {len(jobs)} remaining"
            )
            if not jobs:
                return sidecar_path, len(basenames_df)
        else:
            con.execute(_DROP_TABLE_SQL.format(table=table_name))
            con.execute(_CREATE_TABLE_SQL.format(table=table_name))

        buffer: list[pd.DataFrame] = []
        buffered_rows = 0

        def flush() -> None:
            nonlocal buffered_rows
            if not buffer:
                return
            merged = pd.concat(buffer, axis=0, ignore_index=True)
            con.register("rows_view", merged)
            try:
                # explicit txn: an interrupted INSERT rolls back, so resume never sees a torn flush.
                con.execute("BEGIN TRANSACTION")
                try:
                    con.execute(f"INSERT INTO {table_name} SELECT * FROM rows_view")
                    con.execute("COMMIT")
                except BaseException:
                    con.execute("ROLLBACK")
                    raise
            finally:
                con.unregister("rows_view")
            buffer.clear()
            buffered_rows = 0

        progress = tqdm(
            iter_tokenized_tables(jobs, workers=workers),
            total=len(jobs),
            unit="tbl",
            desc=f"ingest w={workers}",
            smoothing=0.05,
        )
        for rows in progress:
            buffer.append(rows)
            buffered_rows += len(rows)
            if buffered_rows > _CACHE_ROWS_LIMIT:
                flush()
        flush()

        if layout == "two_table":
            print(
                "materializing two_table (blend_index_token + blend_index_tableid)...",
                flush=True,
            )
            t_sort = perf_counter()
            materialize_two_table(
                con,
                base=table_name,
                memory_limit=sort_memory_limit,
                temp_dir=sort_temp_dir,
                threads=sort_threads,
            )
            print(
                f"  two_table build + parity in {perf_counter() - t_sort:.1f}s",
                flush=True,
            )
        elif sort_by != "none":
            proj = {"tableid": "_to_tableid", "token": "_to_tokenized"}[sort_by]
            print(
                f"sorting {table_name} by {', '.join(_SORT_KEYS[sort_by])} "
                f"(matches Vertica {proj} projection)...",
                flush=True,
            )
            t_sort = perf_counter()
            sort_index_table(
                con,
                table_name=table_name,
                sort_by=sort_by,
                memory_limit=sort_memory_limit,
                temp_dir=sort_temp_dir,
                threads=sort_threads,
            )
            print(f"  sort done in {perf_counter() - t_sort:.1f}s", flush=True)
    finally:
        con.close()

    return sidecar_path, len(basenames_df)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv-dir", required=True, type=Path)
    ap.add_argument(
        "--dataset",
        default=None,
        help="override [Dataset].name from config (resolves duckdb + sidecar paths)",
    )
    ap.add_argument(
        "--duckdb-path",
        type=Path,
        default=None,
        help="explicit duckdb path; overrides --dataset",
    )
    ap.add_argument("--table-name", default="blend_index")
    ap.add_argument(
        "--sidecar-path",
        type=Path,
        default=None,
        help="explicit sidecar path; overrides --dataset-derived default",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=default_workers(),
        help="tokenisation pool size. --workers 1 = in-process serial baseline.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="keep existing rows + skip tids already in blend_index. "
        "Requires sidecar from prior run (written upfront).",
    )
    ap.add_argument(
        "--layout",
        choices=["single", "two_table"],
        default=None,
        help="single (one physical blend_index, --sort-by order) or "
        "two_table (blend_index_token + blend_index_tableid). "
        "Default: [Database].layout from config.",
    )
    ap.add_argument(
        "--sort-by",
        choices=["tableid", "token", "none"],
        default="tableid",
        help="post-ingest physical sort of blend_index. 'tableid' (default) "
        "mirrors Vertica's _to_tableid projection and keeps the "
        "`TableId IN (...)` pushdown prunable for scan-bound seekers (C/MC); "
        "'token' mirrors _to_tokenized (faster standalone token-overlap "
        "seekers); 'none' skips.",
    )
    ap.add_argument(
        "--sort-memory-limit",
        default="8GB",
        help="DuckDB memory cap during sort; overflow spills to --sort-temp-dir",
    )
    ap.add_argument(
        "--sort-temp-dir",
        default=None,
        help="directory for DuckDB sort-spill files; "
        "required on RAM-bound machines, needs ~2x table size free",
    )
    ap.add_argument(
        "--sort-threads",
        type=int,
        default=2,
        help="DuckDB worker threads during sort; lower = less peak memory",
    )
    return ap.parse_args()


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    from src.Semantic.config import SemanticConfig

    overrides = {"dataset": args.dataset} if args.dataset is not None else None
    cfg = SemanticConfig.load(overrides=overrides)
    duckdb_path = args.duckdb_path or cfg.duckdb_path
    sidecar_path = args.sidecar_path or cfg.blend_basenames_path
    return duckdb_path, sidecar_path


def main() -> int:
    args = parse_args()
    duckdb_path, sidecar_path = _resolve_paths(args)
    layout = args.layout
    if layout is None:
        import configparser

        from src import paths

        cp = configparser.ConfigParser()
        cp.read(str(paths.config_path()))
        layout = (
            cp["Database"].get("layout", "single")
            if cp.has_section("Database")
            else "single"
        )
    t0 = perf_counter()
    sidecar, n = ingest_csv_dir(
        csv_dir=args.csv_dir,
        duckdb_path=duckdb_path,
        table_name=args.table_name,
        sidecar_path=sidecar_path,
        workers=args.workers,
        resume=args.resume,
        layout=layout,
        sort_by=args.sort_by,
        sort_memory_limit=args.sort_memory_limit,
        sort_temp_dir=args.sort_temp_dir,
        sort_threads=args.sort_threads,
    )
    secs = perf_counter() - t0
    print(
        f"ingested {n} tables into {duckdb_path} ({args.table_name}) "
        f"with workers={args.workers} in {secs:.1f}s"
    )
    print(f"sidecar: {sidecar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
