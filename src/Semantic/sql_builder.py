"""SQL wrapper exposing ranked semantic TableId lists to Blend combiners."""

from __future__ import annotations

from typing import Sequence

from src.DBHandler import DBHandler


_OUTER_TEMPLATE = """
SELECT TableId
FROM ({inner}) AS {alias}
ORDER BY rank ASC
LIMIT $TOPK$
"""

_EMPTY_TEMPLATE = (
    "SELECT TableId FROM (SELECT 0 AS TableId, 0 AS rank WHERE 1=0) AS {alias}"
)


def _values_inner(dbms: str, rows: Sequence[tuple[int, int]], alias: str) -> str:
    if dbms == "postgres":
        body = ", ".join(f"({tid}, {rank})" for tid, rank in rows)
        return f"SELECT * FROM (VALUES {body}) AS {alias}_v(TableId, rank)"
    return " UNION ALL ".join(
        f"SELECT {int(tid)} AS TableId, {int(rank)} AS rank" for tid, rank in rows
    )


def values_select_with_rank(
    db: DBHandler,
    table_ids: Sequence[int],
    k: int,
) -> str:
    """Wrap a ranked int TableId list as a SELECT for Blend combiners."""
    seen: dict[int, int] = {}
    for i, tid in enumerate(table_ids):
        if tid not in seen:
            seen[int(tid)] = i
    rows = sorted(seen.items(), key=lambda kv: kv[1])

    alias = DBHandler.random_subquery_name()
    if not rows:
        sql = _EMPTY_TEMPLATE.format(alias=alias)
    else:
        inner = _values_inner(db.dbms, rows, alias)
        sql = _OUTER_TEMPLATE.format(inner=inner, alias=alias)

    return sql.replace("$TOPK$", str(int(k)))


__all__ = ["values_select_with_rank"]
