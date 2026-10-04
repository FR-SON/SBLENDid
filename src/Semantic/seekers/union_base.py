from __future__ import annotations

import re

import pandas as pd

from src.DBHandler import DBHandler
from src.Operators.Seekers.SeekerBase import Seeker
from .cost import measured_cost

from ..config import SemanticConfig, SemanticOp
from ..retrieve import IndexHandle, search
from ..sql_builder import values_select_with_rank
from .tiebreak import tiebreak_cost


# The positive parser must not match Difference's `TableId NOT IN (...)`.
_PUSHDOWN_IN_RE = re.compile(
    r"(?<!NOT\s)TableId\s+IN\s*\(([^)]*)\)", re.IGNORECASE
)
_PUSHDOWN_NOT_IN_RE = re.compile(
    r"TableId\s+NOT\s+IN\s*\(([^)]*)\)", re.IGNORECASE
)


def _parse_pushdown_ids(additionals: str) -> set[int] | None:
    """Intersect all `TableId IN (...)` sets; None if absent, empty set if nothing survives."""
    matches = _PUSHDOWN_IN_RE.findall(additionals or "")
    if not matches:
        return None
    out: set[int] | None = None
    for body in matches:
        ids: set[int] = set()
        for tok in body.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                ids.add(int(tok))
            except ValueError:
                continue
        out = ids if out is None else (out & ids)
    return out if out is not None else set()


def _parse_pushdown_exclude_ids(additionals: str) -> set[int]:
    out: set[int] = set()
    for body in _PUSHDOWN_NOT_IN_RE.findall(additionals or ""):
        for tok in body.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                out.add(int(tok))
            except ValueError:
                continue
    return out


class SemanticUnionSeekerBase(Seeker):
    """Vector-based union-discovery seeker returning a `VALUES` fragment of ranked tables."""

    is_approximate = True
    HAS_ML_COST_MODEL = False
    SEMANTIC_OP = "SU"

    def __init__(
        self,
        input_df: pd.DataFrame,
        k: int = 10,
        *,
        approach: str | None = None,
        index_name: str | None = None,
        k_coarse: int | None = None,
        query_table_id: str | None = None,
        exact_threshold: int | None = None,
        config_overrides: dict | None = None,
        order_cost: int | None = None,
    ) -> None:
        super().__init__(k)
        # Ordering hint for Intersection's sort, kept out of cost() so it never aggregates upward.
        self.order_cost = None if order_cost is None else int(order_cost)
        self.input = input_df.copy()
        self._cfg = SemanticConfig.load(overrides=config_overrides)
        op_cfg = self._cfg.operator(SemanticOp.SU)
        self._op_cfg = op_cfg
        self._approach = approach or op_cfg.approach
        self._index_name = index_name or op_cfg.index_name
        self._k_coarse = self._cfg.faiss_k_coarse if k_coarse is None else int(k_coarse)
        self._vote_factor = self._cfg.vote_factor_for(SemanticOp.SU)
        self._exact_threshold = (
            self._cfg.exact_threshold if exact_threshold is None else int(exact_threshold)
        )
        self._query_table_id = query_table_id or input_df.attrs.get("table_id")
        if not self._query_table_id and self._cfg.query_encode != "auto":
            raise ValueError(
                "SemanticUnion requires a query_table_id (kwarg or "
                "input_df.attrs['table_id']) when query_encode = off. Query columns "
                "are looked up in the semantic index by (table_id, col_name)."
            )
        self._cached: tuple[tuple, list[int]] | None = None

    def create_sql_query(self, db: DBHandler, additionals: str = "") -> str:
        cache_key = (
            self._query_table_id, tuple(str(c) for c in self.input.columns),
            self.k, self._approach,
            self._index_name, self._k_coarse, self._exact_threshold,
            self._op_cfg.aggregator, self._op_cfg.munkres_threshold,
            additionals,
        )
        if self._cached is None or self._cached[0] != cache_key:
            handle = IndexHandle.open(self._cfg, self._approach, self._index_name)
            allowed_int_ids = _parse_pushdown_ids(additionals)
            table_filter: set[str] | None = None
            if allowed_int_ids is not None:
                if not allowed_int_ids:
                    self._cached = (cache_key, [])
                    return values_select_with_rank(db, [], k=self.k)
                table_filter = {handle.int_to_table[i] for i in allowed_int_ids
                                if i in handle.int_to_table}
                if not table_filter:
                    self._cached = (cache_key, [])
                    return values_select_with_rank(db, [], k=self.k)
            exclude_int_ids = _parse_pushdown_exclude_ids(additionals)
            exclude_filter = {handle.int_to_table[i] for i in exclude_int_ids
                              if i in handle.int_to_table} or None
            from ..aggregators import build_op_aggregator
            aggregator = build_op_aggregator(self._op_cfg, handle.aggregator)
            ids = search(
                handle, self.input, k=self.k, k_coarse=self._k_coarse,
                query_table_id=self._query_table_id,
                table_filter=table_filter,
                exact_threshold=self._exact_threshold,
                exclude_filter=exclude_filter,
                aggregator=aggregator,
                vote_factor=self._vote_factor,
                query_encode=self._cfg.query_encode,
            )
            self._cached = (cache_key, ids)
        _, ids = self._cached
        return values_select_with_rank(db, ids, k=self.k)

    def cost(self) -> int:
        return measured_cost(self, "SU", 2)

    def ml_cost(self, db: DBHandler) -> float:
        return tiebreak_cost(self)

    def predicted_filtered_ms(self, a_cols: int) -> float:
        from ..cost_predict import predicted_filtered_ms_for
        return predicted_filtered_ms_for(self, a_cols)
