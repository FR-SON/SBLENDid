"""Hungarian-assignment table aggregation (DeepJoin-paper scoring)."""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from .base import AggregationContext


class MunkresAggregator:
    name = "munkres"
    requires_ctx = True

    def __init__(self, *, threshold: float) -> None:
        self.threshold = float(threshold)
        self.n_candidates_scored_total = 0
        self.n_assigned_pairs_total = 0
        self.hungarian_seconds_total = 0.0

    def aggregate(
        self,
        per_col_hits: list[tuple[list[str], list[float]]],
        k: int,
        *,
        ctx: AggregationContext | None = None,
    ) -> list[tuple[str, float]]:
        if ctx is None:
            raise ValueError(
                "MunkresAggregator requires an AggregationContext (ctx=); "
                "it scores full query x candidate cosine matrices"
            )
        candidates: dict[str, None] = {}
        for tables, _dists in per_col_hits:
            for t in tables:
                candidates.setdefault(t, None)
        qmat = np.asarray(ctx.qmat, dtype=np.float32)
        ranked: list[tuple[str, float]] = []
        for ct in candidates:
            cmat = np.asarray(ctx.candidate_vectors_fn(ct), dtype=np.float32)
            if qmat.size == 0 or cmat.size == 0:
                continue
            self.n_candidates_scored_total += 1
            cos = qmat @ cmat.T
            above = cos > self.threshold
            if not above.any():
                continue
            graph = np.where(above, cos, 0.0)
            t0 = time.perf_counter()
            row_ind, col_ind = linear_sum_assignment(graph, maximize=True)
            self.hungarian_seconds_total += time.perf_counter() - t0
            vals = cos[row_ind, col_ind]
            valid = vals > self.threshold
            if not valid.any():
                continue
            self.n_assigned_pairs_total += int(valid.sum())
            ranked.append((ct, float(vals[valid].sum())))
        ranked.sort(key=lambda r: r[1], reverse=True)
        return ranked[:k]
