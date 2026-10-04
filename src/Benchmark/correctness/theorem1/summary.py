"""Summary tables from rows.jsonl rows."""
from __future__ import annotations

from statistics import median

from src.Benchmark.correctness.theorem1 import witness as witness_mod

_SC_KW_CAVEAT = ("optimizer_reachable=False: the optimizer never emits this "
                 "order (Rule 1 pins KW first) — arm-1 here compares against "
                 "a plan the optimizer never runs; excluded from aggregates")


def _ok(rows):
    return [r for r in rows if "error" not in r]


def _by_shape(rows, shape_names):
    return {s: [r for r in rows if r["shape"] == s] for s in shape_names
            if any(r["shape"] == s for r in rows)}


def _med(vals):
    vals = [v for v in vals if v is not None]
    return round(median(vals), 1) if vals else None


def build_summary(rows, *, dataset, k, repeats, shape_names, skipped,
                  queries_total, arm_set) -> dict:
    ok = _ok(rows)
    n_err = len(rows) - len(ok)
    groups = _by_shape(ok, shape_names)

    witnesses_by_shape = {}
    floor_by_shape = {}
    for shape, g in groups.items():
        w = [r for r in g if r.get("witness")]
        witnesses_by_shape[shape] = {
            "n": len(g),
            "both_stable": sum(bool(r.get("both_stable")) for r in g),
            "witnesses": len(w),
            "witness_bno_empty": sum(bool(r.get("witness_bno_empty")) for r in w),
            "med_added": _med([r.get("witness_added") for r in w]),
            "med_dropped": _med([r.get("witness_dropped") for r in w]),
            "shape_source": g[0]["shape_source"],
            "operators_upstream_faithful": g[0]["operators_upstream_faithful"],
        }
        if repeats > 1:
            floor_by_shape[shape] = witness_mod.floor_stats(g)

    arm3 = {"by_shape": {}}
    for shape, g in groups.items():
        empty = [r for r in g if not r["bno_runs"][0]]
        nonempty = [r for r in g if r["bno_runs"][0]]
        have_i = all(r.get("n_I") is not None for r in g)
        if have_i:
            nonempty = [r for r in nonempty if r["n_I"] > 0]
            note = "|I|=0 rows excluded from the non-empty split"
        else:
            note = ("n_I unavailable (width arm not run); |I|=0 exclusion "
                    "not applied")

        def _cell(rs):
            return {
                "n": len(rs),
                "set_differs": sum(set(r["opt_runs"][0]) != set(r["bno_runs"][0])
                                   for r in rs),
                "med_added": _med([len(set(r["opt_runs"][0]) - set(r["bno_runs"][0]))
                                   for r in rs]),
                "med_dropped": _med([len(set(r["bno_runs"][0]) - set(r["opt_runs"][0]))
                                     for r in rs]),
            }

        arm3["by_shape"][shape] = {"bno_empty": _cell(empty),
                                   "bno_nonempty": _cell(nonempty),
                                   "note_n_I": note}

    arm1 = {"by_shape": {}, "aggregate_set_differs": 0, "aggregate_list_differs": 0,
            "aggregate_n": 0}
    for shape, g in groups.items():
        ga = [r for r in g if "ab" in r]
        if not ga:
            continue
        cell = {"n": len(ga),
                "set_differs": sum(bool(r["a1_set_differs"]) for r in ga),
                "list_differs": sum(bool(r["a1_list_differs"]) for r in ga),
                "optimizer_reachable": bool(ga[0]["optimizer_reachable"])}
        if not cell["optimizer_reachable"]:
            cell["caveat"] = _SC_KW_CAVEAT
        else:
            arm1["aggregate_set_differs"] += cell["set_differs"]
            arm1["aggregate_list_differs"] += cell["list_differs"]
            arm1["aggregate_n"] += cell["n"]
        arm1["by_shape"][shape] = cell

    width = {}
    for shape, g in groups.items():
        gw = [r for r in g if r.get("width")]
        if not gw:
            continue
        per_w: dict = {}
        for wkey in gw[0]["width"]:
            cells = [r["width"][wkey] for r in gw]
            recs = [c["rec_ab"] for c in cells] + [c["rec_ba"] for c in cells]
            per_w[wkey] = {
                "n": len(cells),
                "set_differs": sum(c["set_differs"] for c in cells),
                "list_differs": sum(c["list_differs"] for c in cells),
                "rec_med": round(median(recs), 3),
                "rec_min": round(min(recs), 3),
                "frac_rec_1": round(sum(v >= 1.0 for v in recs) / len(recs), 3),
            }
        width[shape] = {"n": len(gw),
                        "med_n_I": _med([r["n_I"] for r in gw]),
                        "by_width": per_w}

    skipped = dict(skipped)
    n_skipped = sum(skipped.values()) + n_err
    return {
        "dataset": dataset, "k": k, "repeats": repeats, "arms": sorted(arm_set),
        "cohort": {"queries_total": queries_total, "errors": n_err,
                   "skipped": skipped,
                   "cohort_complete": n_skipped == 0},
        "primary_evidence": {"witnesses_by_shape": witnesses_by_shape},
        "supporting_evidence_floor_choice_dependent":
            {"floor_by_shape": floor_by_shape},
        "arm3": arm3, "arm1": arm1, "width": width,
        "omitted": {
            "gt_overlay": ("Theorem 1 is a claim about optimized vs unoptimized "
                           "output, measured GT-free against B-No; a relevance "
                           "overlay answers a different question and token-only "
                           "shapes have no lead semantic seeker to supply one"),
            "timing": "wall_s recorded per query for scheduling only; no "
                      "latency claims",
        },
    }


def flatten_rows(rows) -> list[dict]:
    """One scalar row per (query, shape, arm, repeat) for results.csv."""
    flat: list[dict] = []
    for r in _ok(rows):
        base = {"query": r["query"], "shape": r["shape"], "k": r["k"],
                "shape_source": r["shape_source"],
                "operators_upstream_faithful": r["operators_upstream_faithful"],
                "arm3_only": r["arm3_only"], "wall_s": r["wall_s"]}
        for i, (opt, bno) in enumerate(zip(r["opt_runs"], r["bno_runs"])):
            flat.append({**base, "arm": "a3", "repeat": i,
                         "n_opt": len(opt), "n_bno": len(bno),
                         "set_differs": set(opt) != set(bno),
                         "list_differs": opt != bno,
                         "added": len(set(opt) - set(bno)),
                         "dropped": len(set(bno) - set(opt)),
                         "bno_empty": not bno})
        if "ab" in r:
            flat.append({**base, "arm": "a1", "repeat": 0,
                         "n_opt": len(r["ab"]), "n_bno": len(r["ba"]),
                         "set_differs": r["a1_set_differs"],
                         "list_differs": r["a1_list_differs"],
                         "added": len(set(r["ab"]) - set(r["ba"])),
                         "dropped": len(set(r["ba"]) - set(r["ab"])),
                         "bno_empty": not r["ba"]})
        for wkey, cell in (r.get("width") or {}).items():
            flat.append({**base, "arm": "width", "repeat": 0, "width": wkey,
                         "set_differs": cell["set_differs"],
                         "list_differs": cell["list_differs"],
                         "rec_ab": cell["rec_ab"], "rec_ba": cell["rec_ba"]})
    return flat


def print_summary(s) -> None:
    print(f"\n=== correctness-theorem1 {s['dataset']} (k={s['k']}, "
          f"repeats={s['repeats']}, arms={','.join(s['arms'])}) ===")
    c = s["cohort"]
    print(f"cohort: {c['queries_total']} queries, skipped={c['skipped']}, "
          f"errors={c['errors']}, complete={c['cohort_complete']}")
    print("\n--- WITNESSES (primary evidence — no noise floor involved) ---")
    print(f"{'shape':>13} {'n':>4} {'both stable':>12} {'WITNESSES':>10} "
          f"{'w/ B-No empty':>14} {'med +add':>9} {'med -drop':>10}  source")
    for shape, w in s["primary_evidence"]["witnesses_by_shape"].items():
        print(f"{shape:>13} {w['n']:>4} {w['both_stable']:>12} "
              f"{w['witnesses']:>10} {w['witness_bno_empty']:>14} "
              f"{str(w['med_added']):>9} {str(w['med_dropped']):>10}  "
              f"{w['shape_source']}"
              + ("" if w["operators_upstream_faithful"] in (True, None)
                 else "  [ops NOT upstream-faithful]"))
    fl = s["supporting_evidence_floor_choice_dependent"]["floor_by_shape"]
    if fl:
        print("\n--- FLOOR (supporting — floor-choice dependent; raw rates are "
              "not evidence) ---")
        print(f"{'shape':>13} {'lvl':>5} {'exc_vs_max':>11} {'exc_vs_opt':>11} "
              f"{'p_eff':>7} {'p_null_opt':>11} {'p_null_bno':>11} "
              f"{'uns_opt':>8} {'uns_bno':>8}")
        for shape, f in fl.items():
            for lvl in ("set", "list"):
                x = f[lvl]
                print(f"{shape:>13} {lvl:>5} {x['excess_vs_max']:>11.3f} "
                      f"{x['excess_vs_opt']:>11.3f} {x['p_eff']:>7.3f} "
                      f"{x['p_null_opt']:>11.3f} {x['p_null_bno']:>11.3f} "
                      f"{x['unstable_opt']:>8} {x['unstable_bno']:>8}")
    print("\n--- ARM 3 (opt vs B-No, split by B-No empty/non-empty) ---")
    for shape, a in s["arm3"]["by_shape"].items():
        e, ne = a["bno_empty"], a["bno_nonempty"]
        print(f"{shape:>13}  empty: {e['set_differs']}/{e['n']}  "
              f"non-empty: {ne['set_differs']}/{ne['n']} "
              f"(+{ne['med_added']}/-{ne['med_dropped']})  [{a['note_n_I']}]")
    if s["arm1"]["by_shape"]:
        print("\n--- ARM 1 (order swap; counterfactual) ---")
        for shape, a in s["arm1"]["by_shape"].items():
            extra = f"  <-- {a['caveat']}" if "caveat" in a else ""
            print(f"{shape:>13}  set {a['set_differs']}/{a['n']}  "
                  f"list {a['list_differs']}/{a['n']}{extra}")
        print(f"{'aggregate':>13}  set {s['arm1']['aggregate_set_differs']}"
              f"/{s['arm1']['aggregate_n']}  list "
              f"{s['arm1']['aggregate_list_differs']}/{s['arm1']['aggregate_n']} "
              f"(optimizer-reachable shapes only)")
    for shape, wd in s["width"].items():
        print(f"\n--- WIDTH SWEEP [{shape}] n={wd['n']} med|I|={wd['med_n_I']} ---")
        print(f"{'width':>7} {'set-differs':>12} {'list-differs':>13} "
              f"{'rec med':>8} {'rec min':>8} {'frac rec==1':>12}")
        for wkey, cell in wd["by_width"].items():
            print(f"{wkey:>7} {cell['set_differs']:>5}/{cell['n']:<6} "
                  f"{cell['list_differs']:>6}/{cell['n']:<6} "
                  f"{cell['rec_med']:>8.3f} {cell['rec_min']:>8.3f} "
                  f"{cell['frac_rec_1']:>12.3f}")
