"""Untruncated (TableId, score) rankings by rewriting a seeker's own base_sql."""
from __future__ import annotations

import re

_ORDER_RE = re.compile(r"ORDER\s+BY\s+(.+?)\s+DESC", re.IGNORECASE | re.DOTALL)
_SELECT = "SELECT TableId FROM AllTables"


def scored_sql_template(base_sql: str) -> str:
    m = _ORDER_RE.search(base_sql)
    if m is None or _SELECT not in base_sql or "LIMIT $TOPK$" not in base_sql:
        raise ValueError(
            "seeker base_sql is not scoreable: need 'SELECT TableId FROM "
            "AllTables ... ORDER BY <expr> DESC LIMIT $TOPK$'")
    expr = " ".join(m.group(1).split())
    out = base_sql.replace("LIMIT $TOPK$", "")
    return out.replace(_SELECT, f"SELECT TableId, {expr} AS score FROM AllTables", 1)


def scored_ranking(leg, db, additionals: str = "") -> list[tuple[int, float]]:
    template = scored_sql_template(leg.base_sql)
    saved = leg.base_sql
    leg.base_sql = template
    try:
        sql = leg.create_sql_query(db, additionals=additionals)
    finally:
        leg.base_sql = saved
    return [(int(r[0]), float(r[1])) for r in db.execute_and_fetchall(sql)]
