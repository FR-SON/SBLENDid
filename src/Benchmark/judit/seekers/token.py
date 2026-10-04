"""Token-overlap (SC) backend over the full lake."""

from __future__ import annotations

import pandas as pd
from tqdm import tqdm


class TokenBackend:
    uses_rank_rollup = False
    method = "sc"
    approach = "token"
    index_name = "-"

    def __init__(self, cfg):
        self.cfg = cfg
        self._csvs = cfg.dataset.dir() / "csvs"
        self._db = None
        self._reg = None
        self._int_to_basename = None
        self._table_col_to_name = None
        self._qcache = (None, None)

    def available(self):
        return self.cfg.blend_basenames_path.is_file()

    def _open_db(self):
        if self._db is None:
            from src.Benchmark.db import open_dataset_db
            self._db = open_dataset_db(self.cfg.dataset.name)
        return self._db

    def _build_maps(self):
        if self._reg is not None:
            return
        sidecar = pd.read_parquet(self.cfg.blend_basenames_path)
        self._int_to_basename = dict(zip(
            sidecar["table_int_id"].astype(int).tolist(),
            sidecar["basename"].astype(str).tolist()))
        reg: dict[str, set[str]] = {}
        tcn: dict[tuple[str, int], str] = {}
        for bn in tqdm(self._int_to_basename.values(), desc="sc/reg",
                       unit="tbl", dynamic_ncols=True, leave=False):
            header = pd.read_csv(self._csvs / bn, dtype=str, keep_default_na=False,
                                 nrows=0).columns.tolist()
            reg[bn] = set(header)
            for cid, name in enumerate(header):
                tcn[(bn, cid)] = name
        self._reg, self._table_col_to_name = reg, tcn

    def reg(self):
        self._build_maps()
        return self._reg

    def _query_values(self, qt, col):
        cqt, cdf = self._qcache
        if cqt != qt:
            cdf = pd.read_csv(self._csvs / qt, dtype=str, keep_default_na=False)
            self._qcache = (qt, cdf)
        return cdf[col].tolist()

    def _sc_search(self, values, k_coarse):
        db = self._open_db()
        cleaned = db.clean_value_collection(values)
        if not cleaned:
            return [], False
        tokens_sql = db.create_sql_list_str(cleaned)
        sql = f"""
        SELECT TableId, ColumnId, COUNT(DISTINCT CellValue) AS overlap_cnt
        FROM AllTables
        WHERE CellValue IN ({tokens_sql})
        GROUP BY TableId, ColumnId
        ORDER BY overlap_cnt DESC
        LIMIT {k_coarse}
        """
        rows = db.execute_and_fetchall(sql)
        sat = len(rows) == k_coarse
        return [(int(r[0]), int(r[1]), int(r[2])) for r in rows], sat

    def search_union_column(self, qt, col, kc, kf=None, *, ef=None):
        self._build_maps()
        sc_rows, sat = self._sc_search(self._query_values(qt, col), kc)
        tables, dists = [], []
        for tid, _cid, cnt in sc_rows[:kf]:
            bn = self._int_to_basename.get(tid)
            if bn is None:
                continue
            tables.append(bn)
            dists.append(-float(cnt))
        return tables, dists, sat

    def search_join_column(self, qt, qcol, kc, kf=None, *, ef=None):
        self._build_maps()
        sc_rows, sat = self._sc_search(self._query_values(qt, qcol), kc)
        cols, dists = [], []
        for tid, cid, cnt in sc_rows[:kf]:
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
