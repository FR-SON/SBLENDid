"""SeekerBackend protocol + approach-agnostic union/join leg templates."""

from __future__ import annotations

import time
from typing import Protocol

from tqdm import tqdm

from src.Benchmark.judit import loading, metrics
from src.Benchmark.self_filter import rollup_without_self, without_self_scored
from src.Semantic import depths
from src.Semantic.rollup import rollup_columns_to_tables


class SeekerBackend(Protocol):
    method: str
    approach: str
    index_name: str

    def available(self) -> bool: ...

    def reg(self) -> dict[str, set[str]]: ...

    def search_union_column(
        self, qt: str, col: str, k_coarse: int, *, ef: int | None = None
    ) -> tuple[list[str], list[float], bool]: ...

    def search_join_column(
        self, qt: str, qcol: str, k_coarse: int, *, ef: int | None = None
    ) -> tuple[list[tuple[str, str]], list[float], bool]: ...


def _plan(k: int, k_coarse, k_vote, vote_factor: float, cfg=None):
    """(k_coarse, k_vote, ef) for ONE evaluation depth; explicit or config pins win."""
    cfg_pin = getattr(cfg, "faiss_k_coarse", None) if cfg is not None else None
    cfg_ef = getattr(cfg, "faiss_hnsw_ef_search", None) if cfg is not None else None
    d = depths.derive(
        k=k, vote_factor=vote_factor,
        pinned_k_coarse=(k_coarse if k_coarse is not None else cfg_pin),
        pinned_ef_search=cfg_ef,
    )
    if d.regime == "pinned":
        return d.k_coarse, (int(k_vote) if k_vote is not None else d.k_vote), \
            d.ef_search
    kc = d.k_coarse if k_coarse is None else int(k_coarse)
    kv = d.k_vote if k_vote is None else int(k_vote)
    # ef follows the fetch on faiss (fill floor); None keeps the configured beam
    ef = d.ef_search if (k_coarse is None and d.ef_search is not None) else None
    return kc, kv, ef


TIMING_KEYS = ("t_search_ms", "t_rollup_ms", "t_metrics_ms")


def _timing_row(t_search, t_rollup, t_metrics, n_calls):
    return {"t_search_ms": t_search * 1000.0,
            "t_rollup_ms": t_rollup * 1000.0,
            "t_metrics_ms": t_metrics * 1000.0,
            "n_search_calls": n_calls}


def run_union_leg(backend, cfg, prefix, *, ks, k_coarse, k_vote=None,
                  vote_factor=depths.DEFAULT_VOTE_FACTOR, limit=None):
    reg = backend.reg()
    qgt = loading.load_union_qgt(cfg, prefix, reg)
    queries = qgt.queries_kept[:limit] if limit else qgt.queries_kept
    rows: list[dict] = []
    gt_counts: list[int] = []
    n_saturated = n_specs = 0
    plans = {k: _plan(k, k_coarse, k_vote, vote_factor, cfg) for k in ks}
    t_wall = time.perf_counter()
    for qt in tqdm(queries, desc=f"union/{backend.method}", unit="q", dynamic_ncols=True):
        t0 = time.perf_counter()
        relevant = qgt.relevant_by_q.get(qt, set())
        row = {"query_table": qt, "query_column": ""}
        fetched: dict[tuple, list] = {}
        t_search = t_rollup = t_metrics = 0.0
        n_calls = 0
        for _k in ks:
            kc, kv, ef = plans[_k]
            per_col = fetched.get((kc, ef))
            if per_col is None:
                per_col = []
                for col in qgt.header_cols[qt]:
                    if col not in reg[qt]:
                        continue
                    _t = time.perf_counter()
                    tables, dists, sat = backend.search_union_column(
                        qt, col, kc, ef=ef)
                    t_search += time.perf_counter() - _t
                    n_calls += 1
                    per_col.append((tables, dists))
                    n_specs += 1
                    n_saturated += int(sat)
                fetched[(kc, ef)] = per_col
            _t = time.perf_counter()
            ranked = [t for t, _ in rollup_without_self(per_col, qt, k=kv)]
            t_rollup += time.perf_counter() - _t
            _t = time.perf_counter()
            row.update(metrics.per_query_metrics(ranked, relevant, (_k,)))
            t_metrics += time.perf_counter() - _t
        runtime_ms = (time.perf_counter() - t0) * 1000.0
        row["runtime_ms"] = runtime_ms
        row.update(_timing_row(t_search, t_rollup, t_metrics, n_calls))
        rows.append(row)
        gt_counts.append(len(relevant))
    leg = metrics.assemble_union_metrics(
        rows, gt_counts, ks=ks,
        k_coarse={k: p[0] for k, p in plans.items()},
        k_vote={k: p[1] for k, p in plans.items()},
        ef_used={k: p[2] for k, p in plans.items()},
        depths_derived=(k_coarse is None and k_vote is None),
        n_saturated=n_saturated, n_specs_total=n_specs,
        n_queries_input=qgt.n_queries_input, n_gt_rows_input=qgt.n_gt_rows_input,
        n_gt_rows_kept=qgt.n_gt_rows_kept, n_dropped_q=qgt.n_dropped_q,
        n_dropped_c=qgt.n_dropped_c,
        wall_clock_seconds=round(time.perf_counter() - t_wall, 6))
    return leg, rows


def run_join_leg(backend, cfg, prefix, *, ks, k_coarse, k_vote=None,
                 vote_factor=depths.DEFAULT_VOTE_FACTOR, limit=None):
    ks = tuple(sorted(ks))
    reg = backend.reg()
    qgt = loading.load_join_qgt(cfg, prefix, reg)
    pairs = qgt.pairs_kept[:limit] if limit else qgt.pairs_kept
    rows: list[dict] = []
    gt_counts: list[int] = []
    gt_counts_col: list[int] = []
    n_saturated = 0
    plans = {k: _plan(k, k_coarse, k_vote, vote_factor, cfg) for k in ks}
    t_wall = time.perf_counter()
    for qt, qc in tqdm(pairs, desc=f"join/{backend.method}", unit="pair", dynamic_ncols=True):
        t0 = time.perf_counter()
        relevant = qgt.relevant_by_q.get((qt, qc), set())
        relevant_cols = qgt.relevant_cols_by_q.get((qt, qc), set())
        row = {"query_table": qt, "query_column": qc}
        fetched: dict[tuple, tuple] = {}
        cols = []
        t_search = t_rollup = t_metrics = 0.0
        n_calls = 0
        for _k in ks:
            kc, kv, ef = plans[_k]
            hit = fetched.get((kc, ef))
            if hit is None:
                _t = time.perf_counter()
                _cols, _dists, sat = backend.search_join_column(qt, qc, kc, ef=ef)
                _cols, _dists = without_self_scored(_cols, _dists, qt)
                t_search += time.perf_counter() - _t
                n_calls += 1
                n_saturated += int(sat)
                hit = (_cols, _dists)
                fetched[(kc, ef)] = hit
            _cols, _dists = hit
            cols = _cols
            _t = time.perf_counter()
            ranked = [t for t, _ in rollup_columns_to_tables(
                [([t for t, _c in _cols], _dists)], k=kv)]
            t_rollup += time.perf_counter() - _t
            _t = time.perf_counter()
            row.update(metrics.per_query_metrics(ranked, relevant, (_k,)))
            t_metrics += time.perf_counter() - _t
        runtime_ms = (time.perf_counter() - t0) * 1000.0
        row.update(metrics.per_query_col_metrics(cols, relevant_cols, ks))
        row["runtime_ms"] = runtime_ms
        row.update(_timing_row(t_search, t_rollup, t_metrics, n_calls))
        rows.append(row)
        gt_counts.append(len(relevant))
        gt_counts_col.append(len(relevant_cols))
    leg = metrics.assemble_join_metrics(
        rows, gt_counts, gt_counts_col, ks=ks,
        k_coarse={k: p[0] for k, p in plans.items()},
        k_vote={k: p[1] for k, p in plans.items()},
        ef_used={k: p[2] for k, p in plans.items()},
        depths_derived=(k_coarse is None and k_vote is None),
        n_saturated=n_saturated,
        n_queries_input=qgt.n_queries_input, n_gt_rows_input=qgt.n_gt_rows_input,
        n_gt_rows_kept=qgt.n_gt_rows_kept, n_dropped_q=qgt.n_dropped_q,
        n_dropped_c=qgt.n_dropped_c, n_gt_col_kept=qgt.n_gt_col_kept,
        n_gt_col_dropped_missing=qgt.n_gt_col_dropped_missing,
        wall_clock_seconds=round(time.perf_counter() - t_wall, 6))
    return leg, rows
