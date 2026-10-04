"""Load a CSV directory into Postgres blend_index, mirroring create_blend_csv_index.py's rows.

    uv run python scripts/load_blend_index_pg.py --csv-dir <dir> --dataset <name>
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from time import perf_counter

from tqdm import tqdm

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

from src.Index.source_ingest import (  # noqa: E402
    list_csv_paths, build_basenames_df, build_jobs, read_or_write_sidecar,
    iter_tokenized_tables, default_workers,
)
from src.Semantic.config import SemanticConfig  # noqa: E402
from src.Semantic import pgvector_store as store  # noqa: E402

# FORCE_NULL(quadrant): empty -> NULL (boolean rejects ''). FORCE_NOT_NULL(tokenized): empty -> '' (DuckDB parity).
_COPY_SQL = (
    "COPY blend_index (tokenized, tableid, colid, rowid, super_key, quadrant) "
    "FROM STDIN WITH (FORMAT csv, FORCE_NULL (quadrant), FORCE_NOT_NULL (tokenized))"
)


def load_blend_index_pg(csv_dir: Path, dataset: str, *,
                        workers: int = 1, resume: bool = False) -> int:
    csv_paths = list_csv_paths(Path(csv_dir))
    jobs = build_jobs(csv_paths)
    basenames_df = build_basenames_df(csv_paths)
    cfg = SemanticConfig.load(overrides={"dataset": dataset})
    read_or_write_sidecar(basenames_df, cfg.blend_basenames_path, resume=resume)

    admin = store.connect(dataset, autocommit=True)
    try:
        store.ensure_schema(admin, dataset)
    finally:
        admin.close()

    def build_indexes_and_vacuum() -> None:
        for ddl in tqdm(store.BLEND_INDEX_INDEXES, desc="build indexes", unit="idx"):
            with conn.cursor() as cur:
                cur.execute(ddl)
            conn.commit()
        # Sampled n_distinct badly undershoots on near-unique columns; pin the exact count.
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE blend_index "
                        "ALTER COLUMN tokenized SET STATISTICS 5000")
            print("computing exact tokenized n_distinct ...", flush=True)
            cur.execute("SELECT COUNT(*) FROM "
                        "(SELECT DISTINCT tokenized FROM blend_index) d")
            n_distinct = cur.fetchone()[0]
            cur.execute(f"ALTER TABLE blend_index "
                        f"ALTER COLUMN tokenized SET (n_distinct = {int(n_distinct)})")
        conn.commit()
        print(f"tokenized n_distinct = {n_distinct}")
        prev = conn.autocommit
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("VACUUM ANALYZE blend_index")
        conn.autocommit = prev

    conn = store.connect(dataset)
    try:
        with conn.cursor() as cur:
            if resume:
                cur.execute(store.CREATE_BLEND_INDEX_SQL_IF_NOT_EXISTS)
                cur.execute("SELECT DISTINCT tableid FROM blend_index")
                done = {r[0] for r in cur.fetchall()}
                jobs = [(tid, p) for tid, p in jobs if tid not in done]
                print(f"resume: {len(done)} tids already loaded, {len(jobs)} remaining")
            else:
                cur.execute("DROP TABLE IF EXISTS blend_index CASCADE")
                cur.execute(store.CREATE_BLEND_INDEX_SQL)
        conn.commit()

        if resume and not jobs:
            build_indexes_and_vacuum()
            print(f'nothing to load into "{dataset}".blend_index (all tids present)')
            return len(basenames_df)

        n_rows = 0
        with conn.cursor() as cur:
            with cur.copy(_COPY_SQL) as copy:
                for rows in tqdm(
                    iter_tokenized_tables(jobs, workers=workers),
                    total=len(jobs), unit="tbl", desc="pg load",
                ):
                    buf = io.StringIO()
                    rows.to_csv(buf, index=False, header=False)
                    copy.write(buf.getvalue())
                    n_rows += len(rows)
        conn.commit()

        build_indexes_and_vacuum()

        print(f"loaded {n_rows} rows / {len(jobs)} tables into "
              f'"{dataset}".blend_index')
    finally:
        conn.close()
    return len(basenames_df)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv-dir", required=True, type=Path)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    t0 = perf_counter()
    n = load_blend_index_pg(args.csv_dir, args.dataset,
                            workers=args.workers, resume=args.resume)
    print(f"done: {n} tables in {perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
