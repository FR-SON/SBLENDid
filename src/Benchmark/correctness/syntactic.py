import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from dataclasses import dataclass
import pandas as pd
from src.Operators import Seekers
from src.Benchmark.coltype import classify_columns

SYNTACTIC_KIND = {"su_sc": "sc", "sj_sc": "sc", "su_c": "c", "sj_mc": "mc"}
_NON_NUMERIC = {"categorical", "id", "text"}
_HUGE_K = 10 ** 6


@dataclass
class SyntacticInputs:
    kind: str
    roles: dict


def _query_col(df, qcol):
    return qcol if qcol is not None else str(df.columns[0])


def select_inputs(kind, df, qcol):
    if kind == "sc":
        return SyntacticInputs("sc", {"col": _query_col(df, qcol)})
    types = classify_columns(df)
    if kind == "c":
        key = next((c for c in df.columns if types[c] == "categorical"), None)
        target = next((c for c in df.columns if types[c] == "numeric"), None)
        if key is None or target is None:
            return None
        return SyntacticInputs("c", {"key": key, "target": target})
    if kind == "mc":
        join_col = _query_col(df, qcol)
        others = [c for c in df.columns
                  if c != join_col and types.get(c) in _NON_NUMERIC]
        if not others:
            return None
        return SyntacticInputs("mc", {"cols": [join_col, *others]})
    raise ValueError(f"unknown syntactic kind {kind!r}")


def build_seeker(inp, df, k, db):
    if inp.kind == "sc":
        s = Seekers.SC(df[inp.roles["col"]].astype(str).tolist(), k=k)
    elif inp.kind == "c":
        src = df[inp.roles["key"]].tolist()
        tgt = pd.to_numeric(df[inp.roles["target"]], errors="coerce").tolist()
        s = Seekers.C(src, tgt, k=k)
    elif inp.kind == "mc":
        s = Seekers.MC(df[inp.roles["cols"]], k=k)
    else:
        raise ValueError(f"unknown syntactic kind {inp.kind!r}")
    s.DB = db
    return s


def full(inp, df, db):
    order = build_seeker(inp, df, _HUGE_K, db).run("")
    if inp.kind == "mc":
        return set(order), None
    return set(order), order


def topk_set(inp, df, k, db, *, full_order=None):
    if full_order is not None:
        return set(full_order[:k])
    if inp.kind == "mc":
        return set(build_seeker(inp, df, k, db).run(""))
    return set(build_seeker(inp, df, _HUGE_K, db).run("")[:k])


def derive(full_set, full_order, sem_list, k):
    sem_set = sem_list if isinstance(sem_list, (set, frozenset)) else set(sem_list)
    if full_order is not None:
        return [t for t in full_order if t in sem_set][:k]
    return [t for t in sem_list if t in full_set][:k]
