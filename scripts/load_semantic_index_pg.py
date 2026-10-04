"""Load a built semantic index (registry + embeddings) into Postgres semantic_columns.

    uv run python scripts/load_semantic_index_pg.py --dataset <name> --approach liftus --index-name default
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path
from time import perf_counter

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from tqdm import tqdm

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

from src.Semantic.config import SemanticConfig  # noqa: E402
from src.Semantic.indexers.base import l2_normalize_rows  # noqa: E402
from src.Semantic.registry import load_registry  # noqa: E402
from src.Semantic import pgvector_store as store  # noqa: E402
from src.Semantic.pgvector_store import _vec_literal  # noqa: E402


def _apply_index_build_tuning(conn) -> None:
    """Opt-in HNSW build tuning via BLEND_PG_MAINTENANCE_WORK_MEM / BLEND_PG_MAX_PARALLEL_MAINTENANCE_WORKERS."""
    applied = {}
    with conn.cursor() as cur:
        for name, env in (("maintenance_work_mem", "BLEND_PG_MAINTENANCE_WORK_MEM"),
                          ("max_parallel_maintenance_workers",
                           "BLEND_PG_MAX_PARALLEL_MAINTENANCE_WORKERS")):
            val = os.environ.get(env)
            if val:
                cur.execute("SELECT set_config(%s, %s, false)", (name, val))
                applied[name] = val
    conn.commit()
    if applied:
        print(f"[pg] index-build tuning: {applied}")


def load_semantic_index_pg(dataset: str, approach: str, index_name: str) -> int:
    cfg = SemanticConfig.load(overrides={"dataset": dataset})
    index_dir = cfg.index_dir(approach, index_name)
    reg = list(load_registry(index_dir / "registry.parquet"))
    vectors = np.load(index_dir / "embeddings.fp32.npy", mmap_mode="r")
    dim = int(vectors.shape[1])
    table = store.semantic_columns_table(approach, index_name)
    copy_sql = (
        f"COPY {table} (global_id, table_int_id, table_basename, colid, "
        "col_name, n_rows, source_path, embedding) "
        "FROM STDIN WITH (FORMAT csv, FORCE_NULL (col_name))"
    )

    sidecar = pd.read_parquet(cfg.blend_basenames_path)
    basename_to_int = dict(zip(sidecar["basename"].astype(str),
                               sidecar["table_int_id"].astype(int)))

    admin = store.connect(dataset, autocommit=True)
    try:
        store.ensure_schema(admin, dataset)
    finally:
        admin.close()

    conn = store.connect(dataset)
    try:
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
            cur.execute(store.create_semantic_columns_sql(table, dim))
        conn.commit()

        mat = np.asarray(vectors[[r.global_id for r in reg]], dtype=np.float32)
        mat = l2_normalize_rows(mat)
        if not np.isfinite(mat).all():
            bad = np.where(~np.isfinite(mat).all(axis=1))[0][0]
            raise ValueError(
                f"non-finite embedding for global_id {reg[bad].global_id} "
                f"(row {bad}); cannot load into pgvector")

        CHUNK = 2000
        with conn.cursor() as cur, cur.copy(copy_sql) as copy:
            with tqdm(total=len(reg), unit="col", desc="pg vectors") as bar:
                for start in range(0, len(reg), CHUNK):
                    chunk = reg[start:start + CHUNK]
                    df = pd.DataFrame({
                        "global_id": [r.global_id for r in chunk],
                        "table_int_id": [basename_to_int[r.table_id] for r in chunk],
                        "table_basename": [r.table_id for r in chunk],
                        "colid": [r.col_idx for r in chunk],
                        "col_name": [r.col_name for r in chunk],
                        "n_rows": [r.n_rows for r in chunk],
                        "source_path": [r.source_path for r in chunk],
                        "embedding": [_vec_literal(mat[start + i]) for i in range(len(chunk))],
                    })
                    buf = io.StringIO()
                    df.to_csv(buf, index=False, header=False)
                    copy.write(buf.getvalue())
                    bar.update(len(chunk))
        conn.commit()

        print(f"building hnsw index on {table}")
        _apply_index_build_tuning(conn)
        index_ddl = store.semantic_columns_indexes(
            table,
            hnsw_m=cfg.faiss_hnsw_M,
            hnsw_ef_construction=cfg.faiss_hnsw_ef_construction)
        for ddl in tqdm(index_ddl, desc="build indexes", unit="idx"):
            with conn.cursor() as cur:
                cur.execute(ddl)
            conn.commit()
        prev = conn.autocommit
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(f"VACUUM ANALYZE {table}")
        conn.autocommit = prev
        print(f'loaded {len(reg)} columns into "{dataset}".{table}')
    finally:
        conn.close()
    return len(reg)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--approach", required=True)
    ap.add_argument("--index-name", required=True)
    args = ap.parse_args()
    t0 = perf_counter()
    n = load_semantic_index_pg(args.dataset, args.approach, args.index_name)
    print(f"done: {n} columns in {perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
