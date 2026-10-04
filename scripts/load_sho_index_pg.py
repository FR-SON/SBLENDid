"""Load the SHO codes.parquet into Postgres as blend_index_code (raw SQL, exempt from AllTables routing)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

from tqdm import tqdm

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

from src.Semantic import pgvector_store as store  # noqa: E402

_CREATE_SQL = """
CREATE TABLE {table} (
    simhash_code integer NOT NULL,
    tableid      integer NOT NULL,
    colid        integer NOT NULL,
    rowid        integer NOT NULL
)
"""


def load_codes_into_pg(codes_path: Path, dataset: str, *, table: str = "blend_index_code") -> int:
    import pandas as pd
    codes = pd.read_parquet(codes_path)

    admin = store.connect(dataset, autocommit=True)
    try:
        store.ensure_schema(admin, dataset)
    finally:
        admin.close()

    conn = store.connect(dataset)
    try:
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
            cur.execute(_CREATE_SQL.format(table=table))
            with cur.copy(f"COPY {table} (simhash_code, tableid, colid, rowid) FROM STDIN") as cp:
                for row in tqdm(codes.itertuples(index=False), total=len(codes), unit="row"):
                    cp.write_row((int(row.simhash_code), int(row.tableid),
                                 int(row.colid), int(row.rowid)))
            cur.execute(f"CREATE INDEX {table}_to_code ON {table} (simhash_code, tableid, colid)")
        conn.commit()

        prev = conn.autocommit
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(f"VACUUM ANALYZE {table}")
        conn.autocommit = prev
    finally:
        conn.close()
    return len(codes)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=None, help="override [Dataset].name")
    ap.add_argument("--index-name", default=None, help="default: [Semantic.SHO] index_name")
    args = ap.parse_args()

    from src.Semantic.config import SemanticConfig, SemanticOp
    overrides = {"dataset": args.dataset} if args.dataset else None
    cfg = SemanticConfig.load(overrides=overrides)
    op = cfg.operator(SemanticOp.SHO)
    index_name = args.index_name or op.index_name
    codes_path = cfg.approach_dir(op.approach, index_name) / "codes.parquet"

    t0 = perf_counter()
    n = load_codes_into_pg(codes_path, cfg.dataset.name)
    print(f"done: {n} rows in {perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
