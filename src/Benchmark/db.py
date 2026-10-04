from __future__ import annotations

from src.DBHandler import DBHandler
from src.Plan import Plan


def open_dataset_db(dataset: str, index_table: str = "blend_index") -> DBHandler:
    """Read DBHandler for `dataset` (DuckDB file or Postgres schema)."""
    return DBHandler.for_dataset(dataset, index_table)


def bind_plan(plan: Plan, db: DBHandler) -> None:
    """Re-point plan + its operators at the shared db; closes the plan's own connection."""
    plan.DB.close()
    plan.DB = db
    for op in plan._operators.values():
        op.DB = db
