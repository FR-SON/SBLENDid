"""Encoder-agnostic semantic backend driving union and join legs via search_gid."""

from __future__ import annotations


class SemanticBackend:
    uses_rank_rollup = True

    def __init__(self, cfg, approach, index_name="default"):
        self.cfg = cfg
        self.approach = approach
        self.index_name = index_name
        self.method = approach
        self._h = None
        self._g2tc = None

    def _handle(self):
        if self._h is None:
            from src.Semantic.retrieve import IndexHandle
            self._h = IndexHandle.open(self.cfg, self.approach, self.index_name)
        return self._h

    def available(self):
        try:
            self._handle()
            return True
        except (FileNotFoundError, KeyError, ValueError):
            return False

    def reg(self):
        reg: dict[str, set[str]] = {}
        for t, c in self._handle().table_col_to_gid.keys():
            reg.setdefault(t, set()).add(c)
        return reg

    def _gid_to_table_col(self):
        if self._g2tc is None:
            h = self._handle()
            self._g2tc = {g: (t, c) for (t, c), g in h.table_col_to_gid.items()}
        return self._g2tc

    def _search(self, qt, col, kc, ef=None):
        """Return the FULL k_coarse list with self-hits; truncation belongs at the rollup."""
        h = self._handle()
        gid = h.table_col_to_gid[(qt, col)]
        scores, gids = h.search_gid(gid, kc, ef=ef)
        sat = bool(gids.shape[1] == kc)
        out_g, out_d = [], []
        for g, s in zip(gids[0].tolist(), scores[0].tolist()):
            if g < 0:
                continue
            out_g.append(int(g))
            out_d.append(-float(s))
        return out_g, out_d, sat

    def search_union_column(self, qt, col, kc, kf=None, *, ef=None):
        g2t = self._handle().gid_to_table
        gids, dists, sat = self._search(qt, col, kc, ef=ef)
        tables = [g2t[g] for g in gids if g in g2t]
        return tables, dists[: len(tables)], sat

    def search_join_column(self, qt, qcol, kc, kf=None, *, ef=None):
        g2tc = self._gid_to_table_col()
        gids, dists, sat = self._search(qt, qcol, kc, ef=ef)
        cols = [g2tc[g] for g in gids if g in g2tc]
        return cols, dists[: len(cols)], sat
