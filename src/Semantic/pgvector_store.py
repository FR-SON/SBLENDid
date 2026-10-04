"""Postgres + pgvector store: connection, schema DDL and vector search."""
from __future__ import annotations

import configparser
import json
import os
import re
from pathlib import Path

import numpy as np
import psycopg

from ._result_shape import pad_to_k as _pad_to_k
from src import paths

_DB_KEYS = ("host", "port", "user", "password", "dbname")

_IDENT_RE = re.compile(r"^[A-Za-z0-9_]+$")
_SCHEMA_RE = re.compile(r"^[A-Za-z0-9_-]+$")

_PG_ENV = ("PGHOST", "PGHOSTADDR", "PGDATABASE", "PGUSER", "PGSERVICE")


def _has_pg_env() -> bool:
    return any(os.environ.get(k) for k in _PG_ENV)


def _vec_literal(v) -> str:
    return "[" + ",".join(repr(float(x)) for x in v) + "]"


def _build_search_sql(table: str, *, with_sql: str, qv_sql: str, where: str, exact: bool) -> str:
    if exact:
        return (f"{with_sql}SELECT global_id, -(embedding <#> {qv_sql}) AS score "
                f"FROM {table} {where} "
                f"ORDER BY embedding <#> {qv_sql}, global_id LIMIT %(k)s")
    # A secondary ORDER BY key defeats hnsw, so ties are broken in Python by the caller.
    return (f"{with_sql}SELECT global_id, 1 - (embedding <=> {qv_sql}) AS score "
            f"FROM {table} {where} "
            f"ORDER BY embedding <=> {qv_sql} LIMIT %(k)s")


def _check_ident(name: str, kind: str = "identifier", *, allow_dash: bool = False) -> str:
    rx = _SCHEMA_RE if allow_dash else _IDENT_RE
    if not rx.match(name or ""):
        raise ValueError(f"unsafe {kind}: {name!r}")
    return name


def semantic_columns_table(approach: str, index_name: str) -> str:
    """Per-(approach, index) vector table name."""
    _check_ident(approach, "approach")
    _check_ident(index_name, "index_name")
    return f"semantic_columns__{approach}__{index_name}"


def create_semantic_columns_sql(table: str, dim: int) -> str:
    return f"""
CREATE TABLE {table} (
    global_id     int PRIMARY KEY,
    table_int_id  int NOT NULL,
    table_basename text,
    colid         int,
    col_name      text,
    n_rows        int,
    source_path   text,
    -- STORAGE main: the type declares `external`, so a 768-dim vector (3080 B)
    -- exceeds the 2032 B toast threshold and every embedding lands out of line.
    -- Each returned row then costs a toast-index probe + chunk reads on top of
    -- the heap fetch. `main` keeps it inline while it fits the page.
    embedding     vector({int(dim)}) STORAGE main NOT NULL
)
"""


def semantic_columns_indexes(table: str, *, hnsw_m: int = 64,
                             hnsw_ef_construction: int = 200) -> list[str]:
    return [
        f"CREATE INDEX {table}_table_int_id ON {table} (table_int_id)",
        f"CREATE INDEX {table}_hnsw ON {table} "
        f"USING hnsw (embedding vector_cosine_ops) "
        f"WITH (m = {int(hnsw_m)}, ef_construction = {int(hnsw_ef_construction)})",
    ]


def read_db_params(ini_path: Path | None = None) -> dict:
    parser = configparser.ConfigParser()
    parser.read(Path(ini_path) if ini_path else paths.config_path())
    sect = parser["Database"]
    return {k: sect.get(k) for k in _DB_KEYS if sect.get(k) is not None}


def connect(dataset: str, *, ini_path: Path | None = None,
            autocommit: bool = False) -> psycopg.Connection:
    _check_ident(dataset, "dataset", allow_dash=True)
    params = read_db_params(ini_path)
    if not params and not _has_pg_env():
        raise RuntimeError(
            "postgres selected but config.ini [Database] has no host/user/dbname and "
            "no PG* env vars are set — add connection params or export PGHOST/PGDATABASE/PGUSER.")
    conn = psycopg.connect(**params, autocommit=autocommit)
    with conn.cursor() as cur:
        cur.execute(f'SET search_path TO "{dataset}", public')
        cur.execute("SET jit = off")
    if not autocommit:
        conn.commit()
    return conn


def ensure_schema(conn: psycopg.Connection, dataset: str) -> None:
    _check_ident(dataset, "dataset", allow_dash=True)
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{dataset}"')
    if not conn.autocommit:
        conn.commit()


CREATE_BLEND_INDEX_SQL = """
CREATE TABLE blend_index (
    tokenized text,
    tableid   int,
    colid     int,
    rowid     int,
    super_key text,
    quadrant  boolean
)
"""

CREATE_BLEND_INDEX_SQL_IF_NOT_EXISTS = CREATE_BLEND_INDEX_SQL.replace(
    "CREATE TABLE blend_index", "CREATE TABLE IF NOT EXISTS blend_index")

BLEND_INDEX_INDEXES = [
    "CREATE INDEX IF NOT EXISTS blend_index_to_tokenized ON blend_index "
    "(tokenized, tableid, colid, rowid) INCLUDE (super_key, quadrant)",
    "CREATE INDEX IF NOT EXISTS blend_index_to_tableid ON blend_index "
    "(tableid, colid, rowid) INCLUDE (tokenized, super_key, quadrant)",
]

class PgVectorStore:
    """Per-IndexHandle long-lived read connection for pgvector search."""

    def __init__(self, dataset: str, approach: str, index_name: str, *,
                 hnsw_ef_search: int = 64, hnsw_iterative_scan: str = "relaxed_order"):
        self.table = semantic_columns_table(approach, index_name)
        self.conn = connect(dataset, autocommit=True)
        with self.conn.cursor() as cur:
            # SET rejects bind params, so int-cast values are inlined.
            cur.execute(f"SET hnsw.ef_search = {int(hnsw_ef_search)}")
            scan = hnsw_iterative_scan if hnsw_iterative_scan in (
                "off", "strict_order", "relaxed_order") else "relaxed_order"
            try:
                cur.execute(f"SET hnsw.iterative_scan = '{scan}'")
            except Exception:   # noqa: BLE001 — pgvector <0.8 lacks this GUC; ef_search still set
                pass
            cur.execute(f"SELECT count(*) FROM {self.table}")
            self._count = cur.fetchone()[0]

    @property
    def vector_count(self) -> int:
        return self._count

    def close(self) -> None:
        self.conn.close()

    def get_vector(self, gid: int) -> np.ndarray:
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT embedding::text FROM {self.table} WHERE global_id=%s", (gid,))
            row = cur.fetchone()
        if row is None:
            raise KeyError(f"gid {gid} not in {self.table}")
        return np.array(json.loads(row[0]), dtype=np.float32)

    def explain_gid(self, gid: int, k: int, *, allowed_int_ids=None) -> dict:
        """EXPLAIN (FORMAT JSON) of the ANN query search_gid issues."""
        params = {"gid": gid, "k": k}
        clauses = []
        if allowed_int_ids is not None:
            clauses.append("table_int_id = ANY(%(ids)s)")
            params["ids"] = list(allowed_int_ids)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        qv = "(SELECT v FROM q)"
        sql = (f"WITH q AS MATERIALIZED "
               f"(SELECT embedding AS v FROM {self.table} WHERE global_id = %(gid)s) "
               f"SELECT global_id, 1 - (embedding <=> {qv}) AS score "
               f"FROM {self.table} {where} "
               f"ORDER BY embedding <=> {qv} LIMIT %(k)s")
        with self.conn.cursor() as cur:
            cur.execute("EXPLAIN (FORMAT JSON) " + sql, params)
            return cur.fetchone()[0]

    def _search(self, *, with_sql: str, qv_sql: str, params: dict, k: int,
                allowed_int_ids, excluded_int_ids, exact: bool) -> tuple[np.ndarray, np.ndarray]:
        params = dict(params, k=k)
        clauses = []
        if allowed_int_ids is not None:
            clauses.append("table_int_id = ANY(%(ids)s)")
            params["ids"] = list(allowed_int_ids)
        if excluded_int_ids:
            clauses.append("NOT (table_int_id = ANY(%(ex)s))")
            params["ex"] = list(excluded_int_ids)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = _build_search_sql(self.table, with_sql=with_sql, qv_sql=qv_sql, where=where, exact=exact)
        if exact:
            with self.conn.transaction(), self.conn.cursor() as cur:
                cur.execute("SET LOCAL enable_indexscan = off")
                cur.execute("SET LOCAL enable_indexonlyscan = off")
                cur.execute("SET LOCAL enable_seqscan = on")
                cur.execute(sql, params)
                rows = cur.fetchall()
        else:
            with self.conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        if not rows:
            return _pad_to_k(np.empty(0, np.float32), np.empty(0, np.int64), k)
        gids = np.array([r[0] for r in rows], dtype=np.int64)
        scores = np.array([r[1] for r in rows], dtype=np.float32)
        if not exact:
            order = np.lexsort((gids, -scores))
            gids, scores = gids[order], scores[order]
        return _pad_to_k(scores, gids, k)

    def search_gid(self, gid: int, k: int, *, allowed_int_ids=None,
                   excluded_int_ids=None, exact: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Top-k (scores, gids), each shaped (1, k) and padded with (-inf, -1)."""
        if exact:
            with_sql = ""
            qv_sql = f"(SELECT embedding FROM {self.table} WHERE global_id=%(gid)s)"
        else:
            with_sql = (f"WITH q AS MATERIALIZED "
                        f"(SELECT embedding AS v FROM {self.table} WHERE global_id = %(gid)s) ")
            qv_sql = "(SELECT v FROM q)"
        return self._search(with_sql=with_sql, qv_sql=qv_sql, params={"gid": gid}, k=k,
                            allowed_int_ids=allowed_int_ids,
                            excluded_int_ids=excluded_int_ids, exact=exact)

    def search_vec(self, vec: np.ndarray, k: int, *, allowed_int_ids=None,
                   excluded_int_ids=None, exact: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """search_gid for a vector that is not in the table; L2-normalised like the loaded rows."""
        q = np.asarray(vec, dtype=np.float32).reshape(-1)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        return self._search(with_sql="", qv_sql="%(qv)s::vector",
                            params={"qv": _vec_literal(q)}, k=k,
                            allowed_int_ids=allowed_int_ids,
                            excluded_int_ids=excluded_int_ids, exact=exact)
