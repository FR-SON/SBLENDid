"""Semantic seeker runtime sweep (HNSW surface + exact curve); sets the FAISS env guard before heavy imports."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.Operators.Seekers.SemanticUnion import SemanticUnion
from src.Operators.Seekers.SemanticJoin import SemanticJoin
from src.Semantic.config import SemanticConfig, SemanticOp
from src.Semantic.indexers.base import l2_normalize_rows
from src.Semantic.retrieve import IndexHandle


_OP_APPROACH = {"SU": SemanticOp.SU, "SJ": SemanticOp.SJ}


@dataclass(frozen=True)
class QueryRef:
    table_id: str
    col_name: str | None
    n_cols: int


class _FakeDB:
    dbms = "duckdb"


def _handle(cfg: SemanticConfig, op: str, ef_search: int) -> IndexHandle:
    from dataclasses import replace
    op_cfg = cfg.operator(_OP_APPROACH[op])
    cfg_ef = replace(cfg, faiss_hnsw_ef_search=int(ef_search))
    return IndexHandle.open(cfg_ef, op_cfg.approach, op_cfg.index_name)


def sample_su_queries(cfg: SemanticConfig, n: int, seed: int) -> list[QueryRef]:
    h = _handle(cfg, "SU", cfg.faiss_hnsw_ef_search)
    tables = sorted(h.table_to_gids)
    rng = np.random.default_rng(seed)
    pick = rng.permutation(len(tables))[: min(n, len(tables))]
    out = []
    for i in pick:
        tid = tables[int(i)]
        out.append(QueryRef(tid, None, len(h.table_to_gids[tid])))
    return out


def sample_sj_queries(cfg: SemanticConfig, n: int, seed: int) -> list[QueryRef]:
    h = _handle(cfg, "SJ", cfg.faiss_hnsw_ef_search)
    cols = sorted(h.table_col_to_gid)
    rng = np.random.default_rng(seed)
    pick = rng.permutation(len(cols))[: min(n, len(cols))]
    return [QueryRef(cols[int(i)][0], cols[int(i)][1], 1) for i in pick]


def query_df_from_index(handle: IndexHandle, table_id: str) -> pd.DataFrame:
    """Build a one-row query df with the table's registry column names (no CSV read)."""
    cols = []
    for gid in handle.table_to_gids[table_id]:
        name = handle.gid_to_col_name[gid]
        if name is None:
            raise ValueError(
                f"query_df_from_index: table {table_id!r} has a None col_name "
                f"(gid {gid}); cannot build a lookup query df.")
        cols.append(name)
    df = pd.DataFrame([[""] * len(cols)], columns=cols)
    df.attrs["table_id"] = table_id
    return df


def _query_df(cfg: SemanticConfig, q: QueryRef, op: str = "SU") -> pd.DataFrame:
    csv_path = cfg.dataset.dir() / "csvs" / q.table_id
    if csv_path.is_file():
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    else:
        op_cfg = cfg.operator(_OP_APPROACH[op])
        handle = IndexHandle.open(cfg, op_cfg.approach, op_cfg.index_name)
        df = query_df_from_index(handle, q.table_id)
    if q.col_name is not None:
        df = df[[q.col_name]]
    df.attrs["table_id"] = q.table_id
    return df


def _make_seeker(cfg, op, q, ef, kc):
    op_cfg = cfg.operator(_OP_APPROACH[op])
    overrides = {
        "dataset": cfg.dataset.name,
        "dataset_root": str(cfg.dataset.root),
        "faiss_hnsw_ef_search": int(ef),
        "vector_backend": cfg.vector_backend,
        "query_encode": "off",
    }
    df = _query_df(cfg, q, op)
    if op == "SU":
        return SemanticUnion(df, k=10, k_coarse=kc, config_overrides=overrides,
                             approach=op_cfg.approach, index_name=op_cfg.index_name)
    return SemanticJoin(
        df, k=10, query_col_name=q.col_name, k_coarse=kc,
        config_overrides=overrides,
        approach=op_cfg.approach, index_name=op_cfg.index_name,
    )


def _faiss_only_ms(handle: IndexHandle, cfg, q: QueryRef, kc: int) -> float:
    cols = [q.col_name] if q.col_name is not None else None
    if cols is None:
        cols = [c for (t, c) in handle.table_col_to_gid if t == q.table_id]
    total = 0.0
    for c in cols:
        gid = handle.table_col_to_gid[(q.table_id, c)]
        vec = np.asarray(handle.vectors[gid], dtype=np.float32)[None, :]
        if not handle.natively_normalized:
            vec = l2_normalize_rows(vec)
        t0 = time.perf_counter()
        handle.faiss_index.search(vec, k=kc)
        total += (time.perf_counter() - t0) * 1000.0
    return total


def build_gids_filter(handle: IndexHandle, target_gids: int, rng) -> tuple[list[int], int]:
    """Pick random whole tables until their columns reach target_gids; return (int ids, |gids|)."""
    tables = sorted(handle.table_to_gids)
    order = rng.permutation(len(tables))
    chosen, n = [], 0
    for i in order:
        tid = tables[int(i)]
        chosen.append(handle.table_to_int_id[tid])
        n += len(handle.table_to_gids[tid])
        if n >= target_gids:
            break
    return [int(x) for x in chosen], n


def _pushdown(int_ids: list[int]) -> str:
    return f" AND TableId IN ({', '.join(str(i) for i in int_ids)}) "


def run_exact_curve(cfg, op, queries, gids_grid, out_csv, seed: int = 0) -> Path:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    handle = _handle(cfg, op, cfg.faiss_hnsw_ef_search)
    rng = np.random.default_rng(seed)
    rows = []
    total = len(gids_grid) * len(queries)
    with tqdm(total=total, desc=f"exact-curve {op}", unit="q") as bar:
        for target in gids_grid:
            int_ids, n_gids = build_gids_filter(handle, target, rng)
            push = _pushdown(int_ids)
            for q in queries:
                overrides = {"dataset": cfg.dataset.name,
                             "dataset_root": str(cfg.dataset.root),
                             "vector_backend": cfg.vector_backend}
                df = _query_df(cfg, q, op)
                if op == "SU":
                    seeker = SemanticUnion(df, k=10, exact_threshold=n_gids + 1,
                                           config_overrides=overrides)
                else:
                    seeker = SemanticJoin(df, k=10, query_col_name=q.col_name,
                                          exact_threshold=n_gids + 1,
                                          config_overrides=overrides)
                t0 = time.perf_counter()
                seeker.create_sql_query(_FakeDB(), additionals=push)
                exact_ms = (time.perf_counter() - t0) * 1000.0
                rows.append({"seeker": op, "table_id": q.table_id,
                             "col_name": q.col_name or "", "n_query_cols": q.n_cols,
                             "m_tables": len(int_ids), "n_gids": n_gids,
                             "exact_ms": round(exact_ms, 4)})
                bar.update(1)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv


def run_filtered_curve(cfg, op, queries, gids_grid, out_csv, seed: int = 0) -> Path:
    """Time the production filtered query across a low gids_grid (pgvector exact regime)."""
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    handle = _handle(cfg, op, cfg.faiss_hnsw_ef_search)
    rng = np.random.default_rng(seed)
    rows = []
    total = len(gids_grid) * len(queries)
    with tqdm(total=total, desc=f"filtered-curve {op}", unit="q") as bar:
        for target in gids_grid:
            int_ids, n_gids = build_gids_filter(handle, target, rng)
            push = _pushdown(int_ids)
            for q in queries:
                overrides = {"dataset": cfg.dataset.name,
                             "dataset_root": str(cfg.dataset.root),
                             "vector_backend": cfg.vector_backend}
                df = _query_df(cfg, q, op)
                if op == "SU":
                    seeker = SemanticUnion(df, k=10, config_overrides=overrides)
                else:
                    seeker = SemanticJoin(df, k=10, query_col_name=q.col_name,
                                          config_overrides=overrides)
                t0 = time.perf_counter()
                seeker.create_sql_query(_FakeDB(), additionals=push)
                ms = (time.perf_counter() - t0) * 1000.0
                rows.append({"seeker": op, "table_id": q.table_id,
                             "col_name": q.col_name or "", "n_query_cols": q.n_cols,
                             "m_tables": len(int_ids), "n_gids": n_gids,
                             "exact_ms": round(ms, 4)})
                bar.update(1)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv


def _classify(plan_json) -> str:
    s = json.dumps(plan_json)
    if "_hnsw" in s:
        return "hnsw"
    if "Bitmap" in s and "_tid" in s:
        return "bitmap-btree"
    if "_tid" in s:
        return "btree"
    if "Seq Scan" in s:
        return "seqscan"
    return "other"


def _cell_plan_label(cfg, handle, op, queries, kc) -> str:
    if cfg.vector_backend != "pgvector":
        return "faiss"
    q = queries[0]
    cols = ([q.col_name] if q.col_name is not None
            else [c for (t, c) in handle.table_col_to_gid if t == q.table_id])
    gid = handle.table_col_to_gid[(q.table_id, cols[0])]
    return _classify(handle.pg.explain_gid(int(gid), kc))


def run_hnsw_surface(cfg, op, queries, ef_grid, kc_grid, out_csv,
                     diagonal: bool = False, repeats: int = 1) -> Path:
    """Time unfiltered seekers over (efSearch, k_coarse) x queries, one row per replicate."""
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if ef_grid:
        vcount = _handle(cfg, op, ef_grid[0]).vector_count
        kept = [kc for kc in kc_grid if kc <= vcount]
        dropped = [kc for kc in kc_grid if kc > vcount]
        if dropped:
            print(f"[sweep {op}] corpus={vcount} cols; dropping k_coarse > corpus: "
                  f"{dropped}", flush=True)
        kc_grid = kept
    cells = [(ef, kc) for ef in ef_grid for kc in kc_grid
             if not diagonal or ef == kc]
    if diagonal and not cells:
        raise SystemExit(
            f"[sweep {op}] --diagonal but no ef == k_coarse cell in "
            f"ef_grid={list(ef_grid)} kc_grid={list(kc_grid)}; pass matching grids."
        )
    rows = []
    repeats = max(1, int(repeats))
    total = len(cells) * len(queries) * repeats
    handles: dict[int, object] = {}
    plan_labels: dict[tuple[int, int], str | None] = {}
    with tqdm(total=total, desc=f"hnsw-surface {op}", unit="q") as bar:
        for rep in range(repeats):
            for ef, kc in cells:
                if ef not in handles:
                    handles[ef] = _handle(cfg, op, ef)
                handle = handles[ef]
                if (ef, kc) not in plan_labels:
                    plan_labels[(ef, kc)] = (
                        _cell_plan_label(cfg, handle, op, queries, kc) if queries else None)
                plan_label = plan_labels[(ef, kc)]
                for q in queries:
                    seeker = _make_seeker(cfg, op, q, ef, kc)
                    t0 = time.perf_counter()
                    seeker.create_sql_query(_FakeDB(), additionals="")
                    total_ms = (time.perf_counter() - t0) * 1000.0
                    if cfg.vector_backend == "faiss":
                        faiss_ms = _faiss_only_ms(handle, cfg, q, kc)
                        overhead_ms = total_ms - faiss_ms
                    else:
                        faiss_ms = overhead_ms = float("nan")
                    rows.append({
                        "seeker": op,
                        "repeat": rep,
                        "efSearch": ef,
                        "k_coarse": kc,
                        "table_id": q.table_id,
                        "col_name": q.col_name or "",
                        "n_cols": q.n_cols,
                        "faiss_ms": round(faiss_ms, 4),
                        "total_ms": round(total_ms, 4),
                        "overhead_ms": round(overhead_ms, 4),
                        "plan_label": plan_label,
                    })
                    bar.update(1)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv
