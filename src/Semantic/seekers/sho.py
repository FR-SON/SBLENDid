from __future__ import annotations

from typing import Iterable

from src.DBHandler import DBHandler
from src.Operators.Seekers.SeekerBase import Seeker
from src.cost_model import resolve_cost

from ..config import SemanticConfig
from ..sho_artifacts import ShoHandle
from ..simhash import clean_cell, is_numeric_column, sample_row_indices
from ..sql_builder import values_select_with_rank


class SimHashOverlapSeekerBase(Seeker):
    """SimHash-Overlap seeker: SC-shaped SQL over the simhash_code column."""

    is_approximate = True
    HAS_ML_COST_MODEL = False
    SEMANTIC_OP = "SHO"

    BASE_SQL = """
    SELECT TableId FROM AllTables
    WHERE simhash_code IN ($CODES$) $ADDITIONALS$
    GROUP BY TableId, ColumnId
    ORDER BY COUNT(DISTINCT simhash_code) DESC
    LIMIT $TOPK$
    """

    def __init__(
        self,
        input_query_values: Iterable,
        k: int = 10,
        *,
        query_table_id: str | int | None = None,
        query_col_id: int | None = None,
        config_overrides: dict | None = None,
        order_cost: int | None = None,
    ) -> None:
        super().__init__(k)
        self.order_cost = None if order_cost is None else int(order_cost)
        self.input = [str(v) for v in input_query_values]
        self._cfg = SemanticConfig.load(overrides=config_overrides)
        self._qtid = query_table_id
        self._qcid = None if query_col_id is None else int(query_col_id)
        self._codes: list[int] | None = None

    def _resolve_codes(self, db: DBHandler) -> list[int]:
        if self._codes is not None:
            return self._codes
        if self._qtid is not None and self._qcid is not None:
            self._codes = self._lookup_codes(db)
        else:
            self._codes = self._embed_codes()
        return self._codes

    def _lookup_codes(self, db: DBHandler) -> list[int]:
        handle = ShoHandle.open(self._cfg)
        tid = (self._qtid if isinstance(self._qtid, int)
               else handle.basename_to_int.get(str(self._qtid)))
        if tid is None:
            return []
        rows = db.execute_and_fetchall(
            f"SELECT DISTINCT simhash_code FROM AllTables "
            f"WHERE TableId = {int(tid)} AND ColumnId = {int(self._qcid)}")
        return sorted(int(r[0]) for r in rows)

    def _embed_codes(self) -> list[int]:
        """External-query fallback: seeded differently from the corpus, so not comparable to the lookup."""
        import sys
        print(
            "[SHO][FALLBACK-ENCODE] live-encoding query column "
            f"(no in-lake identity: query_table_id={self._qtid!r}, "
            f"query_col_id={self._qcid!r}); results follow the EXTERNAL-query "
            "path (re-sample + re-hash), NOT the in-lake index lookup — not "
            "comparable to lookup runs.",
            file=sys.stderr, flush=True)
        handle = ShoHandle.open(self._cfg)
        man = handle.manifest
        cleaned = [c for c in (clean_cell(v) for v in self.input) if c is not None]
        if not cleaned or (man["skip_numeric"] and is_numeric_column(cleaned)):
            return []
        idx = sample_row_indices(len(cleaned), man["row_cap"], man["sample_seed"])
        sampled = list(dict.fromkeys(cleaned[i] for i in idx))
        return sorted(set(handle.codes_for_values(sampled)))

    def create_sql_query(self, db: DBHandler, additionals: str = "") -> str:
        codes = self._resolve_codes(db)
        if not codes:
            return values_select_with_rank(db, [], k=self.k)
        sql = self.BASE_SQL.replace("$TOPK$", str(self.k))
        sql = sql.replace("$ADDITIONALS$", additionals)
        return sql.replace("$CODES$", db.create_sql_list_numeric(codes))

    def cost(self) -> int:
        return resolve_cost("SHO", 5, db=self.DB)

    def ml_cost(self, db: DBHandler) -> float:
        return self._predict_runtime([list(self.input)], db)
