"""Central self-match exclusion for benchmarks."""

from __future__ import annotations


def drop_self_gt(gt):
    """GT DataFrame -> rows whose candidate_table != query_table."""
    return gt[gt["candidate_table"].astype(str) != gt["query_table"].astype(str)]


def _table_of(item):
    return item[0] if isinstance(item, tuple) else item


def without_self(items, query_table):
    """Drop items (table names or (table, ...) tuples) whose table == query_table."""
    qt = str(query_table)
    return [x for x in items if str(_table_of(x)) != qt]


def without_self_scored(items, scores, query_table):
    """Filter parallel `items` and `scores` lists on the item's table."""
    qt = str(query_table)
    keep = [i for i, x in enumerate(items) if str(_table_of(x)) != qt]
    return [items[i] for i in keep], [scores[i] for i in keep]


def rollup_without_self(per_col, query_table, *, k):
    """Drop self from each column-hit list before the rollup's top-k cut, then roll up."""
    from src.Semantic.rollup import rollup_columns_to_tables

    cleaned = [without_self_scored(tables, scores, query_table)
               for tables, scores in per_col]
    return rollup_columns_to_tables(cleaned, k=k)
