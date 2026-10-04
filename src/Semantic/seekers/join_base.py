"""SemanticJoin seeker base: single-column join-key lookup."""

from __future__ import annotations

import pandas as pd

from src.DBHandler import DBHandler
from src.Operators.Seekers.SeekerBase import Seeker
from .cost import measured_cost

from ..config import SemanticConfig, SemanticOp
from ..retrieve import IndexHandle, search
from ..sql_builder import values_select_with_rank
from .tiebreak import tiebreak_cost
from .union_base import _parse_pushdown_ids, _parse_pushdown_exclude_ids


class SemanticJoinSeekerBase(Seeker):
    """Vector-based join-key seeker (one column -> ranked candidate tables)."""

    is_approximate = True
    HAS_ML_COST_MODEL = False
    SEMANTIC_OP = "SJ"

    def __init__(
        self,
        input_df: pd.DataFrame,
        k: int = 10,
        *,
        query_col_name: str | None = None,
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
        if input_df.shape[1] == 0:
            raise ValueError("SemanticJoin requires a non-empty DataFrame")
        if query_col_name is None:
            if input_df.shape[1] != 1:
                raise ValueError(
                    "SemanticJoin requires query_col_name when the input "
                    f"DataFrame has multiple columns (got {input_df.shape[1]})"
                )
            query_col_name = str(input_df.columns[0])
        if query_col_name not in input_df.columns:
            raise KeyError(f"query_col_name {query_col_name!r} not in df.columns")
        self.input = input_df[[query_col_name]].copy()
        self._query_col_name = query_col_name

        self._cfg = SemanticConfig.load(overrides=config_overrides)
        op_cfg = self._cfg.operator(SemanticOp.SJ)
        self._op_cfg = op_cfg
        self._approach = approach or op_cfg.approach
        self._index_name = index_name or op_cfg.index_name
        self._k_coarse = self._cfg.faiss_k_coarse if k_coarse is None else int(k_coarse)
        self._vote_factor = self._cfg.vote_factor_for(SemanticOp.SJ)
        self._exact_threshold = (
            self._cfg.exact_threshold if exact_threshold is None else int(exact_threshold)
        )
        self._query_table_id = query_table_id or input_df.attrs.get("table_id")
        if not self._query_table_id and self._cfg.query_encode != "auto":
            raise ValueError(
                "SemanticJoin requires a query_table_id (kwarg or "
                "input_df.attrs['table_id']) when query_encode = off. The stored "
                "vector is looked up by (table_id, col_name)."
            )
        self._cached: tuple[tuple, list[int]] | None = None

    def create_sql_query(self, db: DBHandler, additionals: str = "") -> str:
        cache_key = (
            self._query_table_id, self._query_col_name,
            self.k, self._approach, self._index_name,
            self._k_coarse, self._exact_threshold,
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
                table_filter = {
                    handle.int_to_table[i] for i in allowed_int_ids
                    if i in handle.int_to_table
                }
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
        return measured_cost(self, "SJ", 1)

    def ml_cost(self, db: DBHandler) -> float:
        return tiebreak_cost(self)

    def predicted_filtered_ms(self, a_cols: int) -> float:
        from ..cost_predict import predicted_filtered_ms_for
        return predicted_filtered_ms_for(self, a_cols)
