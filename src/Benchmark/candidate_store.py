from __future__ import annotations

from collections import OrderedDict

import pandas as pd

class _Cand:
    __slots__ = ("df", "_col_sets")

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._col_sets: dict | None = None

    @property
    def col_sets(self) -> dict:
        if self._col_sets is None:
            self._col_sets = {c: set(self.df[c]) for c in self.df.columns}
        return self._col_sets


_COLS = ("colid", "rowid", "tokenized")


class CandidateStore:
    """Blend int TableId -> wide DataFrame of already-tokenized cells (integer colid cols).

    `sql` must select `_COLS` in order and carry a `{tableid}` slot.
    """

    def __init__(self, db, *, sql: str, max_tables: int | None = None):
        self._db = db
        self._sql = sql
        self._max = max_tables
        self._cache: "OrderedDict[int, _Cand | None]" = OrderedDict()
        self.misses = 0

    def _entry(self, tableid: int) -> "_Cand | None":
        tid = int(tableid)
        if tid in self._cache:
            self._cache.move_to_end(tid)
            return self._cache[tid]
        self.misses += 1
        rows = pd.DataFrame(
            self._db.execute_and_fetchall(self._sql.format(tableid=tid)),
            columns=list(_COLS))
        cand = None
        if not rows.empty:
            df = (rows.pivot(index="rowid", columns="colid", values="tokenized")
                      .sort_index().fillna(""))
            cand = _Cand(df)
        self._cache[tid] = cand
        if self._max is not None and len(self._cache) > self._max:
            self._cache.popitem(last=False)
        return cand

    def get(self, tableid: int) -> pd.DataFrame | None:
        cand = self._entry(tableid)
        return cand.df if cand is not None else None

    def col_sets(self, tableid: int) -> dict | None:
        cand = self._entry(tableid)
        return cand.col_sets if cand is not None else None
