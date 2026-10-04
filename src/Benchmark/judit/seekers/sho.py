"""SHO (simhash-overlap) backend over the simhash_code projection."""

from __future__ import annotations

import json


class ShoBackend:
    uses_rank_rollup = False
    method = "sho"
    approach = "simhash"

    def __init__(self, cfg, index_name="default"):
        self.cfg = cfg
        self.index_name = index_name
        self._adir = cfg.approach_dir(self.approach, index_name)
        self._codes_path = (self._adir / "codes.parquet").as_posix()
        self._db = None
        self._reg = None
        self._int_to_basename = None
        self._table_col_to_name = None
        self._basename_to_int = None
        self._name_colid = None
        self._code_cache: dict = {}
        self._tables_fetched: set[int] = set()

    def available(self):
        return (self._adir / "codes.parquet").is_file()

    def _open_db(self):
        if self._db is None:
            from src.Benchmark.db import open_dataset_db
            self._db = open_dataset_db(self.cfg.dataset.name)
        return self._db

    def _build_maps(self):
        if self._reg is not None:
            return
        from src.Benchmark.nonnumeric_filter import load_sho_nonnumeric
        u = load_sho_nonnumeric(self.cfg, self._adir)
        self._reg = u.by_name
        self._int_to_basename = u.int_to_basename
        self._table_col_to_name = u.table_col_to_name
        self._basename_to_int = {v: k for k, v in u.int_to_basename.items()}
        name_colid: dict[str, dict[str, int]] = {}
        for (bn, cid), nm in u.table_col_to_name.items():
            name_colid.setdefault(bn, {})[nm] = cid
        self._name_colid = name_colid
        json.loads((self._adir / "manifest.json").read_text())

    def reg(self):
        self._build_maps()
        return self._reg

    def _query_codes(self, qt, col):
        self._build_maps()
        tid = self._basename_to_int.get(qt)
        cid = self._name_colid.get(qt, {}).get(col)
        if tid is None or cid is None:
            return []
        if tid not in self._tables_fetched:
            import duckdb
            rel = duckdb.sql(
                f"SELECT DISTINCT colid, simhash_code "
                f"FROM read_parquet('{self._codes_path}') "
                f"WHERE tableid = {tid}").df()
            by_col: dict[int, set[int]] = {}
            for c, code in zip(rel["colid"], rel["simhash_code"]):
                by_col.setdefault(int(c), set()).add(int(code))
            for c, codes in by_col.items():
                self._code_cache[(tid, c)] = sorted(codes)
            self._tables_fetched.add(tid)
        return self._code_cache.get((tid, cid), [])

    def _sho_search(self, codes, k_coarse):
        if not codes:
            return [], False
        db = self._open_db()
        codes_sql = db.create_sql_list_numeric(codes)
        sql = f"""
        SELECT TableId, ColumnId, COUNT(DISTINCT simhash_code) AS overlap_cnt
        FROM AllTables
        WHERE simhash_code IN ({codes_sql})
        GROUP BY TableId, ColumnId
        ORDER BY overlap_cnt DESC
        LIMIT {k_coarse}
        """
        rows = db.execute_and_fetchall(sql)
        sat = len(rows) == k_coarse
        return [(int(r[0]), int(r[1]), int(r[2])) for r in rows], sat

    def search_union_column(self, qt, col, kc, kf=None, *, ef=None):
        sho_rows, sat = self._sho_search(self._query_codes(qt, col), kc)
        tables, dists = [], []
        for tid, _cid, cnt in sho_rows[:kf]:
            bn = self._int_to_basename.get(tid)
            if bn is None:
                continue
            tables.append(bn)
            dists.append(-float(cnt))
        return tables, dists, sat

    def search_join_column(self, qt, qcol, kc, kf=None, *, ef=None):
        sho_rows, sat = self._sho_search(self._query_codes(qt, qcol), kc)
        cols, dists = [], []
        for tid, cid, cnt in sho_rows[:kf]:
            bn = self._int_to_basename.get(tid)
            if bn is None:
                continue
            name = self._table_col_to_name.get((bn, cid))
            if name is None:
                continue
            cols.append((bn, name))
            dists.append(-float(cnt))
        return cols, dists, sat

    def close(self):
        if self._db is not None:
            self._db.close()
            self._db = None
