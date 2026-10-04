"""Shape -> fresh operator instances, untruncated leg output, and B-No."""
from __future__ import annotations

from src.Benchmark.correctness.theorem1.shapes import (
    SEM_LAST_ORDER_COST, SEMANTIC_KINDS, LegSpec, Shape)

TOKEN_NONBINDING_K = 10 ** 7  # rows; semantic legs use n_tables, never 10**9


def cut(lst, k, k_semantics: str = "faithful"):
    if k is None:
        return list(lst)
    if k_semantics == "faithful":
        return list(lst)[:k]
    out: list = []
    for t in lst:
        if t not in out:
            out.append(t)
            if len(out) == k:
                break
    return out


def build_leg(spec: LegSpec, df, c0, c1, k, *, db, dataset,
              query_table_id=None, order_cost=None,
              k_coarse=None, ef_search=None, exact_threshold=None):
    from src.Operators import Seekers
    leg_k = k * spec.k_mult
    if spec.kind in ("SC", "KW"):
        col = c0 if spec.columns == "c0" else c1
        cls = Seekers.SC if spec.kind == "SC" else Seekers.Keyword
        leg = cls(df[col].astype(str).tolist(), k=leg_k)
    elif spec.kind == "MC":
        leg = Seekers.MC(df[[c0, c1]].astype(str), k=leg_k)
    elif spec.kind in SEMANTIC_KINDS:
        overrides = {"dataset": dataset, "query_encode": "off"}
        if ef_search is not None:
            overrides["faiss_hnsw_ef_search"] = int(ef_search)
        kwargs = dict(k=leg_k, query_table_id=query_table_id,
                      config_overrides=overrides, order_cost=order_cost)
        if k_coarse is not None:
            kwargs["k_coarse"] = int(k_coarse)
        if exact_threshold is not None:
            kwargs["exact_threshold"] = int(exact_threshold)
        cls = Seekers.SU if spec.kind == "SU" else Seekers.SJ
        leg = cls(df if spec.kind == "SU" else df[[c0]], **kwargs)
    else:
        raise ValueError(spec.kind)
    leg.DB = db
    return leg


def build_legs(shape: Shape, df, c0, c1, k, *, db, dataset,
               query_table_id=None, sem_position: str = "first",
               sem_knobs: dict | None = None):
    sem_knobs = sem_knobs or {}
    legs = []
    for spec in shape.legs:
        extra = {}
        if spec.kind in SEMANTIC_KINDS:
            extra = dict(sem_knobs)
            extra["order_cost"] = (SEM_LAST_ORDER_COST
                                   if sem_position == "last" else None)
        legs.append(build_leg(spec, df, c0, c1, k, db=db, dataset=dataset,
                              query_table_id=query_table_id, **extra))
    return legs


def raw_rows(leg, db, additionals: str = "") -> list[int]:
    """Every row's TableId from the leg's SQL, no [:k] slice."""
    return [r[0] for r in db.execute_and_fetchall(
        leg.create_sql_query(db, additionals=additionals))]


def full_rows(leg, db, nonbinding_k: int = TOKEN_NONBINDING_K,
              additionals: str = "") -> list[int]:
    """Untruncated, duplicate-preserving row list; fails if the row cap was hit."""
    saved = leg.k
    leg.k = nonbinding_k
    try:
        out = raw_rows(leg, db, additionals)
    finally:
        leg.k = saved
    if len(out) >= nonbinding_k:
        raise RuntimeError(
            f"untruncated reference hit its k cap ({nonbinding_k}); raise it")
    return out


def leg_lists(legs, shape: Shape, *, db, construction: str,
              k_semantics: str = "faithful", full_fns: dict | None = None):
    """Per-leg B-No lists at each leg's own k, plus how each was built ('slice' | 'direct')."""
    full_fns = full_fns or {}
    lists, used = [], []
    for i, (spec, leg) in enumerate(zip(shape.legs, legs)):
        leg_k = leg.k
        if construction == "slice" and spec.kind != "MC":
            fn = full_fns.get(i)
            full = fn() if fn is not None else full_rows(leg, db)
            lists.append(cut(full, leg_k, k_semantics))
            used.append("slice")
        else:
            lists.append(cut(raw_rows(leg, db), leg_k, k_semantics))
            used.append("direct")
    return lists, used


def combine_bno(lists, k, *, order: str = "lega",
                k_semantics: str = "faithful", terminal_ranking=None):
    keep = set(lists[0])
    for other in lists[1:]:
        keep &= set(other)
    if order == "lega":
        agg = [t for t in lists[0] if t in keep]
    elif order == "terminal":
        if terminal_ranking is None:
            raise ValueError("bno order 'terminal' needs a scored terminal ranking")
        pos: dict = {}
        for idx, (tid, _score) in enumerate(terminal_ranking):
            pos.setdefault(tid, idx)
        agg = sorted(keep, key=lambda t: (pos.get(t, len(terminal_ranking)), t))
    else:
        raise ValueError(order)
    return cut(agg, k, k_semantics)


def bno_run(legs, shape: Shape, k, *, db, construction: str,
            bno_order: str = "lega", k_semantics: str = "faithful",
            full_fns: dict | None = None, terminal_ranking=None):
    if construction == "direct" and k_semantics == "faithful" and bno_order == "lega":
        # imported lazily: refs pulls OperatorBase, which binds a DBHandler at import
        from src.Benchmark.correctness.refs import reference
        return cut(reference(legs, "intersection"), k), ["direct"] * len(legs)
    lists, used = leg_lists(legs, shape, db=db, construction=construction,
                            k_semantics=k_semantics, full_fns=full_fns)
    out = combine_bno(lists, k, order=bno_order, k_semantics=k_semantics,
                      terminal_ranking=terminal_ranking)
    return out, used
