from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


@dataclass(frozen=True)
class AggregationContext:
    """L2-normalized query and candidate column vectors for cosine-matrix aggregators."""
    qmat: np.ndarray
    candidate_vectors_fn: Callable[[str], np.ndarray]
    query_table_id: str


class Aggregator(Protocol):
    name: str
    requires_ctx: bool

    def aggregate(
        self,
        per_col_hits: list[tuple[list[str], list[float]]],
        k: int,
        *,
        ctx: AggregationContext | None = None,
    ) -> list[tuple[str, float]]:
        """Roll per-column FAISS hits up to a ranked [(table_id, score)] list."""
        ...
