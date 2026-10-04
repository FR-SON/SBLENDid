"""Depth grid: ef_search (beam) x k_coarse (fetch) x vote depth, at table level."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from src.Benchmark.judit.loading import load_join_qgt, load_union_qgt
from src.Semantic.config import SemanticConfig
from src.Semantic.indexers.base import l2_normalize_rows
from src.Semantic.retrieve import IndexHandle
from src.Semantic.rollup import rollup_columns_to_tables

KC_GRID = (25, 50, 75, 100, 150, 250, 500)
VOTE_DEPTHS = (10, 25, 50, 75, 100, 150, 250, 500)
K_RANGE = (1, 5, 10, 25, 50, 100, 150)


@dataclass(frozen=True)
class Arm:
    """One retrieval pass: a beam, and the deepest fetch it can honestly fill."""

    name: str
    ef: int | None
    max_fetch: int
    role: str

    @property
    def is_exact(self) -> bool:
        return self.ef is None


# FAISS needs ef >= k_coarse to fill; ef64_kc500 deliberately measures the under-filled state
DEFAULT_ARMS: tuple[Arm, ...] = (
    Arm("ef25", 25, 25, "candidate: narrowest beam the fill rule allows at k=10"),
    Arm("ef50", 50, 50, "candidate: the derived beam at k=10 (k_vote 20 + margin)"),
    Arm("ef100", 100, 100, "candidate"),
    Arm("ef250", 250, 250, "candidate"),
    Arm("ef500", 500, 500, "candidate: current server FAISS setting"),
    Arm("ef64_kc500", 64, 500, "diagnostic: the ef < k_coarse under-filled state"),
    Arm("exact", None, 500, "reference: brute-force cosine, no graph"),
)


def open_handle(dataset: str, approach: str, index_name: str, arm: Arm,
                backend: str = "faiss"):
    """A handle whose beam is exactly this arm's."""
    overrides = {
        "dataset": dataset,
        "vector_backend": backend,
        "faiss_k_coarse": arm.max_fetch,
        # the exact arm never walks the graph; ef = width keeps _warn_if_ef_below_k quiet
        "faiss_hnsw_ef_search": arm.max_fetch if arm.ef is None else arm.ef,
    }
    cfg = SemanticConfig.load(overrides=overrides)
    return cfg, IndexHandle.open(cfg, approach, index_name)


def _normalized_matrix(handle) -> np.ndarray:
    mat = np.asarray(handle.vectors, dtype=np.float32)
    if not handle.natively_normalized:
        mat = l2_normalize_rows(mat)
    return mat


def _exact_search_gid(mat: np.ndarray, gid: int, k: int):
    q = mat[gid]
    scores = mat @ q
    k = min(k, scores.shape[0])
    part = np.argpartition(-scores, k - 1)[:k]
    order = part[np.argsort(-scores[part], kind="stable")]
    return scores[order], order.astype(np.int64)


def _retrieve_raw(handle, mat, arm: Arm, gid: int):
    """The raw list, self-hits and padding still in it."""
    if arm.is_exact:
        scores, gids = _exact_search_gid(mat, gid, arm.max_fetch)
    else:
        s, g = handle.search_gid(gid, arm.max_fetch)
        scores, gids = np.asarray(s[0]), np.asarray(g[0])
    return gids, scores


def _clean(gids, scores, gid_to_table, qtid: str, k_coarse: int):
    """Truncate the raw list to `k_coarse`, then drop padding and self-hits."""
    tables: list[str] = []
    dists: list[float] = []
    n_real = n_self = 0
    for g, s in zip(gids[:k_coarse].tolist(), scores[:k_coarse].tolist()):
        if g < 0:
            continue
        n_real += 1
        tid = gid_to_table.get(int(g))
        if tid is None:
            continue
        if tid == qtid:
            n_self += 1
            continue
        tables.append(tid)
        dists.append(-float(s))
    return tables, dists, n_real, n_self


def _recall_precision(ranked: list[str], relevant: set, ks) -> dict:
    out = {}
    seen = 0
    hits_at = {}
    rel = relevant
    for i, t in enumerate(ranked):
        if t in rel:
            seen += 1
        hits_at[i + 1] = seen
    n_rel = len(rel) or 1
    for k in ks:
        h = hits_at.get(min(k, len(ranked)), 0) if ranked else 0
        out[f"recall_at_{k}"] = h / n_rel
        out[f"precision_at_{k}"] = h / k
    return out


def _load_queries(cfg, prefix, reg, task, limit):
    if task == "union":
        qgt = load_union_qgt(cfg, prefix, reg)
        queries = qgt.queries_kept[:limit] if limit else qgt.queries_kept
        units = [(qt, [c for c in qgt.header_cols[qt] if c in reg.get(qt, ())])
                 for qt in queries]
        return units, qgt.relevant_by_q, lambda qt: qt
    qgt = load_join_qgt(cfg, prefix, reg)
    pairs = qgt.pairs_kept[:limit] if limit else qgt.pairs_kept
    units = [(qt, [qc]) for qt, qc in pairs]
    return units, qgt.relevant_by_q, None


def run_depth_grid(
    dataset: str,
    *,
    prefix: str,
    approaches: list[tuple[str, str]],
    tasks: list[str],
    arms: tuple[Arm, ...] = DEFAULT_ARMS,
    ks: tuple[int, ...] = K_RANGE,
    kc_grid: tuple[int, ...] = KC_GRID,
    vote_depths: tuple[int, ...] = VOTE_DEPTHS,
    backend: str = "faiss",
    limit: int | None = None,
    log=print,
) -> list[dict]:
    """One summary row per (lake, task, encoder, arm, k_coarse, k_vote)."""
    summary: list[dict] = []

    for approach, index_name in approaches:
        for arm in arms:
            cfg, handle = open_handle(dataset, approach, index_name, arm, backend)
            mat = _normalized_matrix(handle) if arm.is_exact else None
            reg: dict[str, set[str]] = {}
            for t, c in handle.table_col_to_gid.keys():
                reg.setdefault(t, set()).add(c)

            for task in tasks:
                try:
                    units, relevant_by_q, _ = _load_queries(
                        cfg, prefix, reg, task, limit)
                except FileNotFoundError:
                    log(f"[{dataset}/{approach}/{task}] no query or GT file — skipped")
                    continue

                t0 = time.perf_counter()
                raw: list[tuple] = []
                for qt, cols in units:
                    per_col = []
                    for col in cols:
                        gid = handle.table_col_to_gid.get((qt, str(col)))
                        if gid is not None:
                            per_col.append(_retrieve_raw(handle, mat, arm, gid))
                    if per_col:
                        key = qt if task == "union" else (qt, str(cols[0]))
                        raw.append((key, qt, per_col))
                secs = time.perf_counter() - t0
                log(f"[{dataset}/{approach}/{task}/{arm.name}] "
                    f"{len(raw)} queries retrieved in {secs:.1f}s")

                for kc in [c for c in kc_grid if c <= arm.max_fetch]:
                    cleaned = {}
                    stat_real, stat_self, stat_len = [], [], []
                    for key, qt, per_col in raw:
                        cols_clean = []
                        for gids, scores in per_col:
                            t, d, n_real, n_self = _clean(
                                gids, scores, handle.gid_to_table, qt, kc)
                            cols_clean.append((t, d))
                            stat_real.append(n_real)
                            stat_self.append(n_self)
                            stat_len.append(len(t))
                        cleaned[key] = cols_clean

                    for kv in [v for v in vote_depths if v <= kc]:
                        per_q, n_ranked, n_distinct = [], [], []
                        for key, cols_clean in cleaned.items():
                            rel = relevant_by_q.get(key, set())
                            if not rel:
                                continue
                            ranked = [t for t, _ in
                                      rollup_columns_to_tables(cols_clean, k=kv)]
                            per_q.append(_recall_precision(ranked, rel, ks))
                            n_ranked.append(len(ranked))
                            n_distinct.append(
                                float(np.mean([len(set(t[:kv]))
                                               for t, _ in cols_clean])))
                        if not per_q:
                            continue
                        cell = {
                            "dataset": dataset, "task": task, "encoder": approach,
                            "backend": backend, "arm": arm.name, "role": arm.role,
                            "ef_search": arm.ef, "k_coarse": kc, "k_vote": kv,
                            "margin": kc - kv,
                            "n_queries": len(per_q),
                            "retrieval_seconds": round(secs, 3),
                            "mean_real_ids": round(float(np.mean(stat_real)), 1),
                            "mean_self_dropped": round(float(np.mean(stat_self)), 2),
                            "mean_voted_len": round(float(np.mean(stat_len)), 1),
                            "mean_ranked": round(float(np.mean(n_ranked)), 1),
                            "mean_distinct_tables": round(float(np.mean(n_distinct)), 1),
                        }
                        for k in ks:
                            for m in ("recall", "precision"):
                                col = f"{m}_at_{k}"
                                cell[col] = round(
                                    float(np.mean([r[col] for r in per_q])), 6)
                        summary.append(cell)
    return summary


def best_vote_depth(summary: list[dict], k: int, *, arm="ef500",
                    min_margin: int = 0) -> list[dict]:
    """Per (lake, task, encoder), the k_vote maximizing recall@k."""
    groups: dict[tuple, list[dict]] = {}
    for c in summary:
        if c["arm"] != arm or c["margin"] < min_margin:
            continue
        groups.setdefault((c["dataset"], c["task"], c["encoder"]), []).append(c)
    out = []
    for key, cells in sorted(groups.items(), key=lambda kv: str(kv[0])):
        best = max(cells, key=lambda c: c[f"recall_at_{k}"])
        out.append({
            "dataset": key[0], "task": key[1], "encoder": key[2], "k": k,
            "best_k_vote": best["k_vote"], "best_k_coarse": best["k_coarse"],
            "ratio_kv_over_k": round(best["k_vote"] / k, 2),
            f"recall_at_{k}": best[f"recall_at_{k}"],
        })
    return out


def margin_effect(summary: list[dict], *, arm="ef500", ks=(10, 50)) -> list[dict]:
    """At a fixed vote depth, what a deeper fetch buys."""
    idx: dict[tuple, dict] = {}
    for c in summary:
        if c["arm"] != arm:
            continue
        idx[(c["dataset"], c["task"], c["encoder"], c["k_coarse"], c["k_vote"])] = c
    out = []
    for (ds, task, enc, kc, kv), cell in sorted(idx.items(), key=lambda kv_: str(kv_[0])):
        if kc != kv:
            continue
        for kc2 in sorted({c["k_coarse"] for c in summary if c["arm"] == arm}):
            if kc2 <= kv:
                continue
            ref = idx.get((ds, task, enc, kc2, kv))
            if ref is None:
                continue
            row = {"dataset": ds, "task": task, "encoder": enc, "k_vote": kv,
                   "k_coarse": kc2, "margin": kc2 - kv,
                   "mean_self_dropped": ref["mean_self_dropped"]}
            for k in ks:
                row[f"d_recall_at_{k}"] = round(
                    ref[f"recall_at_{k}"] - cell[f"recall_at_{k}"], 6)
            out.append(row)
    return out
