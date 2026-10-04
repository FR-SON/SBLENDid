from __future__ import annotations

from ..rollup import rollup_columns_to_tables


class DecayVoteAggregator:
    name = "decay_vote"
    requires_ctx = False

    def aggregate(
        self,
        per_col_hits: list[tuple[list[str], list[float]]],
        k: int,
        *,
        ctx=None,
    ) -> list[tuple[str, float]]:
        ranked = rollup_columns_to_tables(per_col_hits, k=k)
        return [(tid, float(score)) for tid, score in ranked]
