"""Repeats -> stability flags, witnesses, and the nondeterminism floor."""
from __future__ import annotations


def witness_flags(opt_runs, bno_runs) -> dict:
    o_sets = {frozenset(x) for x in opt_runs}
    b_sets = {frozenset(x) for x in bno_runs}
    stable = len(o_sets) == 1 and len(b_sets) == 1
    differ = stable and next(iter(o_sets)) != next(iter(b_sets))
    o = next(iter(o_sets)) if stable else None
    b = next(iter(b_sets)) if stable else None
    return dict(
        stable_opt=len(o_sets) == 1,
        stable_bno=len(b_sets) == 1,
        both_stable=stable,
        witness=differ,
        witness_bno_empty=bool(differ and not b),
        witness_added=(len(o - b) if differ else None),
        witness_dropped=(len(b - o) if differ else None),
    )


def floor_stats(rows) -> dict:
    """Nondeterminism-floor statistics for opt and B-No, at set and list level."""
    out: dict = {}
    for level, keyf in (("set", frozenset), ("list", tuple)):
        unstable_opt = unstable_bno = 0
        nn_o = nd_o = nn_b = nd_b = en = ed = 0
        for r in rows:
            o = [keyf(x) for x in r["opt_runs"]]
            b = [keyf(x) for x in r["bno_runs"]]
            unstable_opt += len(set(o)) > 1
            unstable_bno += len(set(b)) > 1
            for i in range(len(o)):
                for j in range(i + 1, len(o)):
                    nd_o += 1
                    nn_o += o[i] != o[j]
            for i in range(len(b)):
                for j in range(i + 1, len(b)):
                    nd_b += 1
                    nn_b += b[i] != b[j]
            for x in o:
                for y in b:
                    ed += 1
                    en += x != y
        p_null_opt = nn_o / max(nd_o, 1)
        p_null_bno = nn_b / max(nd_b, 1)
        p_eff = en / max(ed, 1)
        out[level] = dict(
            n=len(rows), unstable_opt=unstable_opt, unstable_bno=unstable_bno,
            p_null_opt=p_null_opt, p_null_bno=p_null_bno, p_eff=p_eff,
            excess_vs_opt=p_eff - p_null_opt,
            excess_vs_max=p_eff - max(p_null_opt, p_null_bno),
            raw_rate_not_evidence=True,
        )
    return out
