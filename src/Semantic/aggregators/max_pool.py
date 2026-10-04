"""Max-pool column -> table aggregation (best single column hit wins)."""

from __future__ import annotations


class MaxPoolAggregator:
    """Score each table by its best column hit; input dists are negated similarities (lower = better)."""

    name = "max_pool"
    requires_ctx = False

    def aggregate(
        self,
        per_col_hits: list[tuple[list[str], list[float]]],
        k: int,
        *,
        ctx=None,
    ) -> list[tuple[str, float]]:
        best: dict[str, float] = {}
        for tables, dists in per_col_hits:
            for tid, d in zip(tables, dists):
                d = float(d)
                if d < best.get(tid, float("inf")):
                    best[tid] = d
        ranked = sorted(
            ((tid, -d) for tid, d in best.items()),
            key=lambda r: (-r[1], r[0]),
        )
        return ranked[:k]
