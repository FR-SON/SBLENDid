"""Strict-gap audit: can tie resolution alone explain a table opt added vs B-No?"""
from __future__ import annotations

NOT_AUDITABLE_KINDS = frozenset({"MC", "SU", "SJ"})
_NEG_INF = float("-inf")


def audit_witness(added_ids, leg_rankings, leg_ks) -> dict:
    """Classify added tables against each truncating leg's score boundary."""
    if not added_ids:
        return {"verdict": "not_auditable", "reason": "no_added_tables",
                "records": []}
    truncating = {n: len(r) > leg_ks[n] for n, r in leg_rankings.items()}
    records: list = []
    any_strict = False
    above_flags: list = []
    for t in sorted(added_ids):
        t_above_all = True
        for name, ranking in leg_rankings.items():
            scores = [s for tid, s in ranking if tid == t]
            best = max(scores) if scores else _NEG_INF
            bnd = ranking[leg_ks[name] - 1][1] if truncating[name] else None
            strict = bool(truncating[name] and best < bnd)
            records.append({"table": t, "leg": name,
                            "best_score": None if best == _NEG_INF else best,
                            "boundary": bnd, "strict": strict})
            any_strict |= strict
            if truncating[name] and not best > bnd:
                t_above_all = False
        above_flags.append(t_above_all)
    if any_strict:
        verdict = "genuine"
    elif not any(truncating.values()):
        verdict = "not_truncation"
    elif all(above_flags):
        verdict = "not_truncation"
    else:
        verdict = "tie_fragile"
    return {"verdict": verdict, "records": records}
