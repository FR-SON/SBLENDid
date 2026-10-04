"""Recall-vs-latency frontier for the semantic ANN arms (faiss vs pgvector)."""
from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter, sleep

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.Benchmark.datasource import load_union_queries
from src.Benchmark.runspec import bench_results_root
from src.Semantic.config import SemanticConfig, SemanticOp
from src.Semantic.indexers.base import l2_normalize_rows

PG_VARIANTS = ("baseline", "relaxed", "insql", "binary", "batch", "best", "noscore")
_SCAN = {"baseline": "strict_order"}


@dataclass(frozen=True)
class QueryTable:
    table_id: str
    gids: tuple[int, ...]
    col_names: tuple[str, ...]


def _index_dir(cfg: SemanticConfig, op: SemanticOp) -> Path:
    oc = cfg.operator(op)
    return cfg.index_dir(oc.approach, oc.index_name)


def build_query_set(dataset: str, op: SemanticOp, n: int, seed: int,
                    cfg: SemanticConfig) -> list[QueryTable]:
    """Sample n union query tables whose columns are all present in the registry."""
    reg = pd.read_parquet(_index_dir(cfg, op) / "registry.parquet")
    by_table: dict[str, list[tuple[int, int, str]]] = {}
    for tid, cidx, cname, gid in zip(reg["table_id"].astype(str),
                                     reg["col_idx"].astype(int),
                                     reg["col_name"].astype(str),
                                     reg["global_id"].astype(int)):
        by_table.setdefault(tid, []).append((cidx, gid, cname))
    queries = [q for q in load_union_queries(dataset) if q in by_table]
    rng = np.random.default_rng(seed)
    picked = rng.permutation(len(queries))[:n]
    out = []
    for i in sorted(picked.tolist()):
        cols = sorted(by_table[queries[i]])
        out.append(QueryTable(queries[i],
                              tuple(g for _, g, _ in cols),
                              tuple(c for _, _, c in cols)))
    return out


def exact_reference(qs: list[QueryTable], k: int, cfg: SemanticConfig,
                    op: SemanticOp, *, cache: bool = True) -> dict[int, set[int]]:
    """Exact top-k gids per query gid by cosine, cached on disk."""
    idx = _index_dir(cfg, op)
    oc = cfg.operator(op)
    gids = sorted({g for q in qs for g in q.gids})
    # bump the version tag on any metric change, else a stale cached npz is reused
    sig = (f"v3_{oc.approach}_{oc.index_name}_{len(gids)}_{k}_"
           f"{hash(tuple(gids)) & 0xffffffff:08x}")
    cpath = bench_results_root(cfg.dataset.name) / f"ann_exact_{sig}.npz"
    if cache and cpath.exists():
        z = np.load(cpath)
        return {int(g): set(row.tolist()) for g, row in zip(z["gids"], z["top"])}
    V = np.load(idx / "embeddings.fp32.npy", mmap_mode="r")
    # both engines rank by cosine and liftus vectors are not unit-norm, so normalise
    Vf = l2_normalize_rows(np.asarray(V, dtype=np.float32))
    top = np.empty((len(gids), k), dtype=np.int64)
    CH = 256
    for s in tqdm(range(0, len(gids), CH), desc="exact reference", unit="chunk"):
        block = gids[s:s + CH]
        S = np.ascontiguousarray(Vf[block]) @ Vf.T
        part = np.argpartition(-S, k, axis=1)[:, :k + 1]
        rows = np.arange(len(block))[:, None]
        ordered = part[rows, np.argsort(-S[rows, part], axis=1)]
        for j, g in enumerate(block):
            keep = [int(x) for x in ordered[j] if int(x) != g][:k]
            top[s + j] = keep
    if cache:
        cpath.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cpath, gids=np.array(gids, dtype=np.int64), top=top)
    return {int(g): set(row.tolist()) for g, row in zip(gids, top)}


def _recall(ids, ref: set[int], k: int, self_gid: int) -> tuple[float, int]:
    real = [int(g) for g in ids if int(g) >= 0 and int(g) != int(self_gid)]
    return len(set(real) & ref) / k, len(real)


def run_faiss(qs, efs, k, cfg, op, ref) -> list[dict]:
    import faiss
    idx = _index_dir(cfg, op)
    index = faiss.read_index(str(idx / "hnsw.faiss"))
    hnsw = faiss.downcast_index(index.index) if hasattr(index, "index") else index
    V = np.load(idx / "embeddings.fp32.npy", mmap_mode="r")
    rows = []
    for ef in efs:
        hnsw.hnsw.efSearch = int(ef)
        index.search(np.ascontiguousarray(V[list(qs[0].gids)], dtype=np.float32), k)
        for q in tqdm(qs, desc=f"faiss ef={ef}", unit="tbl", leave=False):
            Q = np.ascontiguousarray(V[list(q.gids)], dtype=np.float32)
            t = perf_counter()
            _, I = index.search(Q, k)
            ms = (perf_counter() - t) * 1000.0
            recs, reals = zip(*(_recall(I[i], ref[g], k, g)
                                for i, g in enumerate(q.gids)))
            rows.append({"arm": "faiss", "variant": "faiss", "ef": ef,
                         "table": q.table_id, "n_cols": len(q.gids),
                         "ms_table": ms, "ms_col": ms / len(q.gids),
                         "recall": float(np.mean(recs)),
                         "real_ids": float(np.mean(reals)), "plan": "faiss_hnsw"})
    return rows


def _register_vector_binary(conn) -> bool:
    from psycopg.adapt import Dumper
    from psycopg.pq import Format
    from psycopg.types import TypeInfo
    info = TypeInfo.fetch(conn, "vector")
    if info is None:
        return False

    class _VecBin(Dumper):
        format = Format.BINARY
        oid = info.oid

        def dump(self, obj):
            a = np.asarray(obj, dtype=">f4")
            return struct.pack(">HH", a.shape[0], 0) + a.tobytes()

    conn.adapters.register_dumper(np.ndarray, _VecBin)
    return True


class _PgArm:
    def __init__(self, dataset: str, variant: str, table: str, k: int):
        from src.Semantic import pgvector_store as store
        self.variant, self.table, self.k = variant, table, k
        self.conn = store.connect(dataset, autocommit=True)
        # must precede cursor creation: a cursor snapshots the adapters map
        wants_binary = variant in ("binary", "best")
        self.binary = _register_vector_binary(self.conn) if wants_binary else False
        if wants_binary and not self.binary:
            raise RuntimeError("could not register a binary vector dumper")
        self.cur = self.conn.cursor()
        self.cur.execute("SET jit = off")
        self.cur.execute(
            f"SET hnsw.iterative_scan = '{_SCAN.get(variant, 'relaxed_order')}'")

    def set_ef(self, ef: int) -> None:
        self.cur.execute(f"SET hnsw.ef_search = {int(ef)}")

    def close(self) -> None:
        self.conn.close()

    def _sql_literal(self):
        return (f"SELECT global_id, 1 - (embedding <=> %(qv)s::vector) AS score "
                f"FROM {self.table} ORDER BY embedding <=> %(qv)s::vector "
                f"LIMIT {self.k}")

    def _sql_binary(self):
        # %(...)b forces the binary dumper; %s would fail to adapt the ndarray
        return (f"SELECT global_id, 1 - (embedding <=> %(qv)b) AS score "
                f"FROM {self.table} ORDER BY embedding <=> %(qv)b LIMIT {self.k}")

    def _sql_noscore(self):
        """Diagnostic only: drops the score so no result row's embedding is fetched."""
        return (f"WITH q AS MATERIALIZED "
                f"(SELECT embedding AS v FROM {self.table} WHERE global_id = %(gid)s) "
                f"SELECT global_id FROM {self.table} "
                f"ORDER BY embedding <=> (SELECT v FROM q) LIMIT {self.k}")

    def _sql_insql(self):
        return (f"WITH q AS MATERIALIZED "
                f"(SELECT embedding AS v FROM {self.table} WHERE global_id = %(gid)s) "
                f"SELECT global_id, 1 - (embedding <=> (SELECT v FROM q)) AS score "
                f"FROM {self.table} ORDER BY embedding <=> (SELECT v FROM q) "
                f"LIMIT {self.k}")

    def _sql_batch(self, n: int):
        """One statement with n independent HNSW scans; a LATERAL join would degrade to seqscans."""
        seed = (lambda i: f"%(v{i})b") if self.variant == "best" else (
            lambda i: f"(SELECT embedding FROM {self.table} WHERE global_id = %(g{i})s)")
        return " UNION ALL ".join(
            f"(SELECT {i} AS q, global_id, 1 - (embedding <=> {seed(i)}) AS score "
            f"FROM {self.table} ORDER BY embedding <=> {seed(i)} LIMIT {self.k})"
            for i in range(n))

    def _fetch_vector(self, gid: int) -> np.ndarray:
        self.cur.execute(f"SELECT embedding::text FROM {self.table} WHERE global_id=%s",
                         (gid,))
        return np.array(json.loads(self.cur.fetchone()[0]), dtype=np.float32)

    def plan(self, gid: int) -> str:
        """Plan label of this variant's own SQL (catches a variant that skips the HNSW index)."""
        if self.variant == "best":
            sql, params = self._sql_batch(1), {"v0": self._fetch_vector(gid)}
        elif self.variant == "batch":
            sql, params = self._sql_batch(1), {"g0": int(gid)}
        elif self.variant == "insql":
            sql, params = self._sql_insql(), {"gid": gid}
        elif self.variant == "noscore":
            sql, params = self._sql_noscore(), {"gid": gid}
        elif self.variant == "binary":
            sql, params = self._sql_binary(), {"qv": self._fetch_vector(gid)}
        else:
            from src.Semantic.pgvector_store import _vec_literal
            sql, params = self._sql_literal(), {"qv": _vec_literal(self._fetch_vector(gid))}
        self.cur.execute("EXPLAIN (FORMAT JSON, COSTS OFF) " + sql, params)
        blob = json.dumps(self.cur.fetchone()[0])
        if "_hnsw" in blob:
            return "hnsw"
        return "seqscan" if "Seq Scan" in blob else "other"

    def search_table(self, q: QueryTable) -> tuple[float, list[list[int]]]:
        """(ms for the whole table, per-column id lists in column order)."""
        if self.variant in ("batch", "best"):
            t = perf_counter()
            if self.variant == "best":
                params = {f"v{i}": self._fetch_vector(g) for i, g in enumerate(q.gids)}
            else:
                params = {f"g{i}": int(g) for i, g in enumerate(q.gids)}
            self.cur.execute(self._sql_batch(len(q.gids)), params)
            rows = self.cur.fetchall()
            ms = (perf_counter() - t) * 1000.0
            per: list[list[int]] = [[] for _ in q.gids]
            for qi, hit, _score in rows:
                per[int(qi)].append(int(hit))
            return ms, per
        out: list[list[int]] = []
        t = perf_counter()
        for g in q.gids:
            if self.variant == "noscore":
                self.cur.execute(self._sql_noscore(), {"gid": int(g)})
            elif self.variant == "insql":
                self.cur.execute(self._sql_insql(), {"gid": int(g)})
            elif self.variant == "binary":
                self.cur.execute(self._sql_binary(), {"qv": self._fetch_vector(g)})
            else:
                from src.Semantic.pgvector_store import _vec_literal
                self.cur.execute(self._sql_literal(),
                                 {"qv": _vec_literal(self._fetch_vector(g))})
            out.append([int(r[0]) for r in self.cur.fetchall()])
        return (perf_counter() - t) * 1000.0, out


def run_pgvector(qs, efs, k, dataset, variants, cfg, op, ref) -> list[dict]:
    from src.Semantic.pgvector_store import semantic_columns_table
    oc = cfg.operator(op)
    table = semantic_columns_table(oc.approach, oc.index_name)
    rows = []
    for variant in variants:
        arm = _PgArm(dataset, variant, table, k)
        try:
            label = arm.plan(qs[0].gids[0])
            for ef in efs:
                arm.set_ef(ef)
                arm.search_table(qs[0])
                for q in tqdm(qs, desc=f"pg {variant} ef={ef}", unit="tbl", leave=False):
                    ms, per_col = arm.search_table(q)
                    recs, reals = zip(*(_recall(ids, ref[g], k, g)
                                        for g, ids in zip(q.gids, per_col)))
                    rows.append({"arm": "pgvector", "variant": variant, "ef": ef,
                                 "table": q.table_id, "n_cols": len(q.gids),
                                 "ms_table": ms, "ms_col": ms / len(q.gids),
                                 "recall": float(np.mean(recs)),
                                 "real_ids": float(np.mean(reals)), "plan": label})
        finally:
            arm.close()
    return rows


_STATS_SETTLE_S = 1.2  # > PGSTAT_MIN_INTERVAL (1s): IO stats flush at most once a second


def _statio(conn, schema: str) -> dict[str, int]:
    sleep(_STATS_SETTLE_S)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT relname, heap_blks_read FROM pg_statio_user_tables"
            " WHERE schemaname = %(s)s"
            " UNION ALL"
            " SELECT indexrelname, idx_blks_read FROM pg_statio_user_indexes"
            " WHERE schemaname = %(s)s", {"s": schema})
        return {r[0]: int(r[1] or 0) for r in cur.fetchall()}


def _pool_state(conn, schema: str) -> dict:
    out: dict = {}
    with conn.cursor() as cur:
        cur.execute("SHOW shared_buffers")
        out["shared_buffers"] = cur.fetchone()[0]
        cur.execute(
            "SELECT pg_size_pretty(sum(pg_total_relation_size(c.oid)))::text,"
            "       sum(pg_total_relation_size(c.oid))"
            " FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace"
            " WHERE n.nspname = %s AND c.relkind = 'r'", (schema,))
        row = cur.fetchone()
        out["working_set"], out["working_set_bytes"] = row[0], int(row[1] or 0)
        try:
            cur.execute(
                "SELECT pg_size_pretty(count(*) FILTER (WHERE relfilenode IS NULL)"
                " * 8192::bigint)::text FROM pg_buffercache")
            out["pool_free"] = cur.fetchone()[0]
        except Exception:  # noqa: BLE001
            conn.rollback() if not conn.autocommit else None
            out["pool_free"] = None
    return out


def _pgvector_with_io_probe(rows, qs, efs, k, dataset, variants, cfg, sop, ref):
    from src.Semantic import pgvector_store as store
    conn = store.connect(dataset, autocommit=True)
    try:
        before = _statio(conn, dataset)
        first = run_pgvector(qs, efs, k, dataset, variants, cfg, sop, ref)
        mid = _statio(conn, dataset)
        run_pgvector(qs, efs, k, dataset, variants, cfg, sop, ref)
        after = _statio(conn, dataset)
        pool = _pool_state(conn, dataset)
    finally:
        conn.close()
    names = set(before) | set(mid) | set(after)
    per_rel = {
        n: {"pass1_reads": mid.get(n, 0) - before.get(n, 0),
            "pass2_reads": after.get(n, 0) - mid.get(n, 0)}
        for n in sorted(names)
    }
    p1 = sum(v["pass1_reads"] for v in per_rel.values())
    p2 = sum(v["pass2_reads"] for v in per_rel.values())
    fits = pool.get("working_set_bytes", 0) and pool.get("shared_buffers")
    io = {**pool, "pass1_reads": p1, "pass2_reads": p2,
          "per_relation": per_rel,
          "inconclusive": bool(p1 == 0 and p2 == 0 and fits),
          "ok": p2 <= max(64, int(0.01 * p1))}
    return rows + first, io


def run_frontier(dataset: str, *, op: str = "SU", k_coarse: int | None = None,
                 efs: list[int], n_queries: int, seed: int,
                 variants: list[str], with_faiss: bool = True,
                 io_stats: bool = False, approach: str | None = None,
                 index_name: str | None = None) -> tuple[list[dict], dict]:
    ops = None
    if approach or index_name:
        from src.Semantic.config import OperatorConfig
        base = SemanticConfig.load(overrides={"dataset": dataset})
        try:
            cur = base.operator(SemanticOp(op))
            a, i = cur.approach, cur.index_name
        except KeyError:
            a, i = approach, index_name or "default"
        ops = {SemanticOp(op): OperatorConfig(approach=approach or a,
                                             index_name=index_name or i)}
    cfg = SemanticConfig.load(overrides={"dataset": dataset}, operators=ops)
    sop = SemanticOp(op)
    k = int(k_coarse if k_coarse is not None else cfg.faiss_k_coarse)
    qs = build_query_set(dataset, sop, n_queries, seed, cfg)
    if not qs:
        raise SystemExit(f"no union query tables of {dataset!r} are in the registry")
    ref = exact_reference(qs, k, cfg, sop)
    rows: list[dict] = []
    io: dict | None = None
    if with_faiss:
        rows += run_faiss(qs, efs, k, cfg, sop, ref)
    if variants and io_stats:
        rows, io = _pgvector_with_io_probe(
            rows, qs, efs, k, dataset, variants, cfg, sop, ref)
    elif variants:
        rows += run_pgvector(qs, efs, k, dataset, variants, cfg, sop, ref)
    cells: dict[str, dict] = {}
    for r in rows:
        cells.setdefault(f"{r['variant']}@ef{r['ef']}", {"rows": []})["rows"].append(r)
    summary = {
        "dataset": dataset, "op": op, "k_coarse": k, "seed": seed,
        "n_query_tables": len(qs),
        "n_col_searches": sum(len(q.gids) for q in qs),
        "mean_cols_per_table": sum(len(q.gids) for q in qs) / len(qs),
        "cells": {
            name: {
                "arm": c["rows"][0]["arm"], "variant": c["rows"][0]["variant"],
                "ef": c["rows"][0]["ef"], "plan": c["rows"][0]["plan"],
                "median_ms_table": median(r["ms_table"] for r in c["rows"]),
                "mean_ms_col": float(np.mean([r["ms_col"] for r in c["rows"]])),
                "mean_recall": float(np.mean([r["recall"] for r in c["rows"]])),
                "mean_real_ids": float(np.mean([r["real_ids"] for r in c["rows"]])),
            }
            for name, c in cells.items()
        },
    }
    if io is not None:
        summary["io"] = io
    return rows, summary
