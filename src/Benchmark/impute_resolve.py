from __future__ import annotations

import pandas as pd

from src.Index.tokenize import tokenize_cell

_norm = tokenize_cell


def align_columns(cand: pd.DataFrame, examples: pd.DataFrame, *,
                  cand_normalized: bool = False, col_sets: dict | None = None):
    """The candidate's (key_col, val_col) best containing the example pairs, or None."""
    if cand.shape[1] < 2:
        return None
    cn = cand if cand_normalized else cand.apply(lambda s: s.map(_norm))
    if col_sets is None:
        col_sets = {c: set(cn[c]) for c in cn.columns}
    ex = [(_norm(k), _norm(v)) for k, v in examples.itertuples(index=False)]
    ex = [(k, v) for k, v in ex if k and v]
    if not ex:
        return None
    ex_keys = {k for k, _ in ex}
    best, best_score = None, 0
    cols = list(cn.columns)
    for kc in cols:
        if not (col_sets[kc] & ex_keys):
            continue
        sub = cn[cn[kc].isin(ex_keys)]
        for vc in cols:
            if vc == kc:
                continue
            pairs = set(zip(sub[kc], sub[vc]))
            score = sum(1 for kv in ex if kv in pairs)
            if score > best_score:
                best, best_score = (kc, vc), score
    return best if best_score >= 1 else None


def build_mapping(cand: pd.DataFrame, kc, vc, *, normalized: bool = False) -> dict[str, str]:
    ks = cand[kc] if normalized else cand[kc].map(_norm)
    vs = cand[vc] if normalized else cand[vc].map(_norm)
    m: dict[str, str] = {}
    for k, v in zip(ks, vs):
        if k and v and k not in m:
            m[k] = v
    return m


from collections import Counter, defaultdict


def predict(ranked_cands, examples: pd.DataFrame, masked_keys: set[str],
            *, cand_normalized: bool = False, col_sets_by_key=None) -> dict[str, str]:
    votes: dict[str, Counter] = defaultdict(Counter)
    order: dict[str, list[str]] = defaultdict(list)
    for key, cand in ranked_cands:
        cs = col_sets_by_key.get(key) if col_sets_by_key else None
        al = align_columns(cand, examples, cand_normalized=cand_normalized, col_sets=cs)
        if not al:
            continue
        m = build_mapping(cand, *al, normalized=cand_normalized)
        for k in masked_keys:
            if k in m:
                votes[k][m[k]] += 1
                order[k].append(m[k])
    preds: dict[str, str] = {}
    for k, c in votes.items():
        top = max(c.values())
        preds[k] = next(v for v in order[k] if c[v] == top)
    return preds


def score(preds: dict[str, str], truth: dict[str, str]) -> dict[str, int]:
    keys = set(truth)
    covered = set(preds) & keys
    correct = sum(1 for k in covered if preds[k] == truth[k])
    return {"n_keys": len(keys), "n_covered": len(covered), "n_correct": correct}
