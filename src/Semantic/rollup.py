"""Column → table rollup via decay voting (mirrors LIFTus's result.solve())."""

from __future__ import annotations

from collections import Counter
from typing import Iterable


def vote_solve(
    M: dict[str, int],
    target_tables: list[str],
    distances: list[float],
    *,
    eps: float = 1e-6,
) -> None:
    """Decay-vote one query column's ranked hits into M (LIFTus result.solve convention)."""
    n = len(target_tables)
    if n == 0:
        return
    score = n
    for i in range(n):
        M[target_tables[i]] = M.get(target_tables[i], 0) + score
        if i < n - 1 and abs(distances[i] - distances[i + 1]) > eps:
            score -= 1


def rollup_columns_to_tables(
    per_col_retrieved: Iterable[tuple[list[str], list[float]]],
    k: int,
) -> list[tuple[str, int]]:
    """Aggregate decay votes across query columns, each truncated to its top k."""
    M: dict[str, int] = {}
    for tables, dists in per_col_retrieved:
        vote_solve(M, tables[:k], dists[:k])
    return Counter(M).most_common()
