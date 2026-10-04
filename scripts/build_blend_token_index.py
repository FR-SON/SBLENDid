"""One-off: add an ART index on ``blend_index.tokenized`` for an existing DuckDB dataset."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

from src.Semantic.config import SemanticConfig

DEFAULT_INDEX_NAME = "idx_blend_index_tokenized"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset",
        default=None,
        help="override [Dataset].name (defaults to whatever config.ini says)",
    )
    ap.add_argument(
        "--db-filename",
        default="blend.duckdb",
        help="DuckDB file name under datasets/<dataset>/",
    )
    ap.add_argument("--table", default="blend_index")
    ap.add_argument("--column", default="tokenized")
    ap.add_argument("--index-name", default=DEFAULT_INDEX_NAME)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="report the plan and existing indexes without modifying the DB",
    )
    ap.add_argument(
        "--threads",
        type=int,
        default=2,
        help="DuckDB worker threads during the build; lower = less peak memory",
    )
    ap.add_argument(
        "--memory-limit",
        default="8GB",
        help="DuckDB memory cap; everything above spills to --temp-dir",
    )
    ap.add_argument(
        "--temp-dir",
        default="",
        help="directory for DuckDB spill files; needs several GB free. "
        "Empty = DuckDB's default location next to the DB.",
    )
    args = ap.parse_args()

    overrides = {"dataset": args.dataset} if args.dataset else None
    cfg = SemanticConfig.load(overrides=overrides)
    db_path = cfg.dataset.dir() / args.db_filename
    if not db_path.exists():
        print(f"ERROR: DuckDB file not found: {db_path}", file=sys.stderr)
        return 2

    pre_size = db_path.stat().st_size
    print(f"dataset      : {cfg.dataset.name}")
    print(f"db           : {db_path}  ({pre_size / 1e9:.2f} GB)")
    print(f"target       : {args.index_name}  ON  {args.table}({args.column})")
    print()
    print("REMINDER: back up the DB before running without --dry-run.")
    print("REMINDER: any other process holding the DB open must be closed first.")
    print()

    import duckdb

    con = duckdb.connect(database=str(db_path), read_only=args.dry_run)

    existing = con.execute(
        "SELECT index_name, sql FROM duckdb_indexes() WHERE table_name = ?",
        [args.table],
    ).fetchall()
    if existing:
        print("existing indexes on", args.table)
        for ix_name, sql in existing:
            print(f"  - {ix_name}: {sql}")
        if any(ix_name == args.index_name for ix_name, _ in existing):
            print(f"\nindex '{args.index_name}' already present. nothing to do.")
            con.close()
            return 0
    else:
        print(f"existing indexes on {args.table}: (none)")

    row_count = con.execute(f"SELECT COUNT(*) FROM {args.table}").fetchone()[0]
    print(f"\nrows in {args.table}: {row_count:,}")

    if args.dry_run:
        print(f"\n--dry-run: would CREATE INDEX {args.index_name} "
              f"ON {args.table}({args.column}).")
        con.close()
        return 0

    con.execute("SET enable_progress_bar = true")
    con.execute("SET progress_bar_time = 500")
    con.execute(f"SET threads = {args.threads}")
    con.execute(f"SET memory_limit = '{args.memory_limit}'")
    con.execute("SET preserve_insertion_order = false")
    if args.temp_dir:
        td = Path(args.temp_dir).expanduser().resolve()
        td.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory = '{td}'")
        print(f"spill dir    : {td}")
    print(
        f"threads={args.threads}  memory_limit={args.memory_limit}  "
        f"preserve_insertion_order=false"
    )

    sql = f"CREATE INDEX {args.index_name} ON {args.table}({args.column})"
    print(f"\nrunning: {sql}", flush=True)
    t0 = time.perf_counter()
    con.execute(sql)
    elapsed = time.perf_counter() - t0
    print(f"  done in {elapsed:.1f}s")
    con.close()

    post_size = db_path.stat().st_size
    delta = post_size - pre_size
    print(
        f"\nDB size: {pre_size / 1e9:.2f} GB -> {post_size / 1e9:.2f} GB  "
        f"(+{delta / 1e9:.2f} GB)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
