"""Driver for correctness-theorem1: arms a3 / a1 / width, streaming rows.jsonl."""
from __future__ import annotations

import contextlib
import json
import sys
import time

from tqdm import tqdm

from src.Benchmark.correctness.theorem1 import audit as audit_mod
from src.Benchmark.correctness.theorem1 import fidelity
from src.Benchmark.correctness.theorem1 import legs as legs_mod
from src.Benchmark.correctness.theorem1 import scored as scored_mod
from src.Benchmark.correctness.theorem1 import summary as summary_mod
from src.Benchmark.correctness.theorem1 import witness as witness_mod
from src.Benchmark.correctness.theorem1.shapes import (
    SCOREABLE_KINDS, SEMANTIC_KINDS, SHAPES, is_semantic, leg_order_key,
    select_columns)
from src.Benchmark.runspec import Field

DEFAULT_SHAPES = ["sc_sc_flat", "sc_sc_prov", "sc_kw"]
DEFAULT_WIDTHS = [10, 25, 50, 100, 250, 500, 1000, 2500, None]

FIELDS = (
    Field("shapes", default=DEFAULT_SHAPES),
    Field("k", prefix="k", default=10),
    Field("repeats", prefix="rep", default=1),
    Field("rows", prefix="rows", default=5000),
    Field("arms", default=["a3"]),
    Field("bno_construction", prefix="bno", default="slice"),
    Field("bno_order", prefix="ord", default="lega"),
    Field("k_semantics", prefix="ksem", default="faithful"),
    Field("sem_position", prefix="sem", default="first"),
    Field("k_coarse", prefix="kc"),
    Field("ef", prefix="ef"),
    Field("exact_threshold", prefix="xt"),
    Field("force_exact", prefix="fx", default=False),
    Field("limit", prefix="lim"),
    Field("query_timeout", prefix="qt"),
    Field("width_grid", prefix="w", default=DEFAULT_WIDTHS),
    Field("verify_soundness", in_slug=False, default=False),
    Field("allow_measured_costs", in_slug=False, default=False),
    Field("resume_from", in_slug=False),
    Field("backend"),
)


def field_values(args, backend: str) -> dict:
    ef_eff = args.ef if args.ef is not None else args.k_coarse
    return {"shapes": list(args.shapes), "k": args.k, "repeats": args.repeats,
            "rows": args.rows, "arms": list(args.arms),
            "bno_construction": args.bno_construction,
            "bno_order": args.bno_order, "k_semantics": args.k_semantics,
            "sem_position": args.sem_position, "k_coarse": args.k_coarse,
            "ef": ef_eff, "exact_threshold": args.exact_threshold,
            "force_exact": args.force_exact, "limit": args.limit,
            "query_timeout": args.query_timeout,
            "width_grid": list(args.width_grid),
            "verify_soundness": args.verify_soundness,
            "allow_measured_costs": args.allow_measured_costs,
            "resume_from": args.resume_from, "backend": backend}


def present_queries(dataset: str, queries) -> tuple[list, int]:
    """Filter to query tables present in the lake's csvs/; returns (present, n_absent)."""
    from src.Benchmark.datasource import dataset_dir
    csv_root = dataset_dir(dataset) / "csvs"
    present = [q for q in queries if (csv_root / q).is_file()]
    return present, len(queries) - len(present)


def _semantic_ctx(dataset: str, shape_names) -> dict:
    from src.Semantic.config import SemanticConfig, SemanticOp
    from src.Semantic.retrieve import IndexHandle
    ctx: dict = {}
    kinds = {l.kind for n in shape_names for l in SHAPES[n].legs
             if l.kind in SEMANTIC_KINDS}
    for kind in sorted(kinds):
        cfg = SemanticConfig.load(overrides={"dataset": dataset})
        oc = cfg.operator(SemanticOp.SU if kind == "SU" else SemanticOp.SJ)
        handle = IndexHandle.open(cfg, oc.approach, oc.index_name)
        ints = sorted(handle.table_to_int_id.values())
        ctx[kind] = {"cfg": cfg, "handle": handle,
                     "n_gids": int(handle.vector_count),
                     "n_tables": len(ints), "all_ids": ints}
    return ctx


def _sem_ctor_knobs(sem_knobs: dict | None) -> dict:
    if not sem_knobs:
        return {}
    out: dict = {}
    if sem_knobs["k_coarse_source"] == "cli":
        out["k_coarse"] = sem_knobs["k_coarse"]
    if sem_knobs["ef_search_source"] in ("cli", "derived"):
        out["ef_search"] = sem_knobs["ef_search"]
    if sem_knobs["exact_threshold_source"] == "cli":
        out["exact_threshold"] = sem_knobs["exact_threshold"]
    return out


def _semantic_full_fn(spec, shape, df, c0, c1, *, db, dataset, qtid, sctx):
    """Closure producing the exact untruncated semantic reference ranking."""
    info = sctx[spec.kind]
    ref = fidelity.reference_knobs(info["n_gids"])

    def fn():
        leg = legs_mod.build_leg(spec, df, c0, c1, 1, db=db, dataset=dataset,
                                 query_table_id=qtid,
                                 k_coarse=ref["k_coarse"],
                                 exact_threshold=ref["exact_threshold"])
        leg.k = info["n_tables"]
        add = (" AND TableId IN "
               f"({db.create_sql_list_numeric(info['all_ids'])}) ")
        with fidelity.forced_exact():
            import src.Semantic.retrieve as retrieve
            if not retrieve.should_use_exact(info["n_gids"],
                                             ref["exact_threshold"],
                                             vector_backend=info["cfg"].vector_backend):
                raise fidelity.FidelityError(
                    f"{shape.name}: semantic reference leg did not take the "
                    "exact path (exact_fires gate)")
            out = legs_mod.raw_rows(leg, db, add)
        fidelity.assert_clean_ids(out, where=f"{shape.name} semantic reference")
        return out

    return fn


def _full_fns(shape, df, c0, c1, *, db, dataset, qtid, sctx) -> dict:
    return {i: _semantic_full_fn(spec, shape, df, c0, c1, db=db,
                                 dataset=dataset, qtid=qtid, sctx=sctx)
            for i, spec in enumerate(shape.legs) if spec.kind in SEMANTIC_KINDS}


def _prov(shape, legs, *, sem_position, db, sctx, force_exact):
    names = ["a", "b"]
    prov: dict = {"leg_costs": {n: leg.cost() for n, leg in zip(names, legs)}}
    keys = [leg_order_key(s, sem_position=sem_position) for s in shape.legs]
    order = sorted(range(len(keys)), key=lambda i: keys[i])
    sem_i = next((i for i, s in enumerate(shape.legs)
                  if s.kind in SEMANTIC_KINDS), None)
    if sem_i is None:
        return prov
    spec = shape.legs[sem_i]
    info = sctx[spec.kind]
    prov["order_cost"] = getattr(legs[sem_i], "order_cost", None)
    prov["semantic_runs_last"] = order[-1] == sem_i
    tok_i = next((i for i, s in enumerate(shape.legs)
                  if s.kind in SCOREABLE_KINDS), None)
    if tok_i is None or not prov["semantic_runs_last"]:
        return prov
    handle = info["handle"]
    push = set(legs_mod.raw_rows(legs[tok_i], db))
    tables = {t for t, i in handle.table_to_int_id.items() if i in push}
    n_gids = sum(len(handle.table_to_gids.get(t, ())) for t in tables)
    et = legs[sem_i]._exact_threshold
    import src.Semantic.retrieve as retrieve
    fn = (fidelity._always_exact_when_filtered if force_exact
          else retrieve.should_use_exact)
    prov.update(pushdown_tables=len(tables), pushdown_gids=n_gids,
                exact_threshold=et,
                exact_fires=bool(fn(n_gids, et,
                                    vector_backend=info["cfg"].vector_backend)),
                exact_forced=bool(force_exact))
    probe = fidelity.depth_probe(
        info["cfg"], k=legs[sem_i].k, n_gids=n_gids, n_total=info["n_gids"],
        exact_used=prov["exact_fires"],
        k_coarse_pin=legs[sem_i]._k_coarse,
        ef_pin=info["cfg"].faiss_hnsw_ef_search)
    prov["regime"] = probe.regime
    return prov


def _eval_one(shape, df, qt, *, k, repeats, db, dataset, sctx, sem_knobs,
              arm_set, width_grid, bno_construction, bno_order, k_semantics,
              sem_position, force_exact, verify_soundness):
    from src.Benchmark.correctness import arms
    c0, c1 = select_columns(df)
    df.attrs["table_id"] = qt
    ctor_knobs = _sem_ctor_knobs(sem_knobs)

    def mk():
        return legs_mod.build_legs(shape, df, c0, c1, k, db=db, dataset=dataset,
                                   query_table_id=qt, sem_position=sem_position,
                                   sem_knobs=ctor_knobs)

    forced = fidelity.forced_exact if force_exact else contextlib.nullcontext
    full_fns = _full_fns(shape, df, c0, c1, db=db, dataset=dataset, qtid=qt,
                         sctx=sctx)

    terminal_ranking = None
    if bno_order == "terminal":
        keys = [leg_order_key(s, sem_position=sem_position) for s in shape.legs]
        t_i = max(range(len(keys)), key=lambda i: (keys[i], i))
        terminal_ranking = scored_mod.scored_ranking(mk()[t_i], db)

    opt_runs, bno_runs, used = [], [], None
    for _ in range(repeats):
        with forced():
            opt = arms.run_arm(mk(), "intersection", "cost", db, k, guard="none")
        fidelity.assert_clean_ids(opt, where=f"{shape.name}/{qt} opt")
        with forced():
            bno, used = legs_mod.bno_run(
                mk(), shape, k, db=db, construction=bno_construction,
                bno_order=bno_order, k_semantics=k_semantics,
                full_fns=full_fns, terminal_ranking=terminal_ranking)
        fidelity.assert_clean_ids(bno, where=f"{shape.name}/{qt} bno")
        opt_runs.append(opt)
        bno_runs.append(bno)

    row = dict(query=qt, shape=shape.name, k=k, repeats=repeats,
               opt_runs=opt_runs, bno_runs=bno_runs,
               shape_source=shape.shape_source,
               operators_upstream_faithful=shape.operators_upstream_faithful,
               optimizer_reachable=shape.optimizer_reachable,
               arm3_only=shape.arm3_only, bno_construction_used=used,
               n_full_a=None, n_full_b=None, n_I=None)
    row.update(witness_mod.witness_flags(opt_runs, bno_runs))
    row["prov"] = _prov(shape, mk(), sem_position=sem_position, db=db,
                        sctx=sctx, force_exact=force_exact)

    if shape.arm3_only:
        return row

    if "a1" in arm_set:
        with forced():
            ab = arms.run_arm(mk(), "intersection", [0, 1], db, k, guard="none")
            ba = arms.run_arm(mk(), "intersection", [1, 0], db, k, guard="none")
        row.update(ab=ab, ba=ba, a1_set_differs=set(ab) != set(ba),
                   a1_list_differs=ab != ba)

    if "width" in arm_set or verify_soundness:
        fa = (full_fns[0]() if 0 in full_fns
              else legs_mod.full_rows(mk()[0], db))
        fb = (full_fns[1]() if 1 in full_fns
              else legs_mod.full_rows(mk()[1], db))
        I = set(fa) & set(fb)
        row.update(n_full_a=len(set(fa)), n_full_b=len(set(fb)), n_I=len(I))
        if "width" in arm_set and I:
            nb = [sctx[s.kind]["n_tables"] if s.kind in SEMANTIC_KINDS
                  else legs_mod.TOKEN_NONBINDING_K for s in shape.legs]
            w_block: dict = {}
            for w in width_grid:
                src_a = legs_mod.cut(fa, w)
                src_b = legs_mod.cut(fb, w)
                add_a = (" AND TableId IN "
                         f"({db.create_sql_list_numeric(src_a)}) ") if src_a else None
                add_b = (" AND TableId IN "
                         f"({db.create_sql_list_numeric(src_b)}) ") if src_b else None
                x = (legs_mod.full_rows(mk()[1], db, nb[1], add_a)
                     if add_a else [])
                y = (legs_mod.full_rows(mk()[0], db, nb[0], add_b)
                     if add_b else [])
                w_block[str(w)] = dict(
                    set_differs=set(x) != set(y), list_differs=x != y,
                    rec_ab=len(set(x) & I) / len(I),
                    rec_ba=len(set(y) & I) / len(I))
            row["width"] = w_block
        if verify_soundness:
            viol = []
            for arm_name, res in (("opt", opt_runs), ("bno", bno_runs)):
                for rep, lst in enumerate(res):
                    bad = [t for t in lst if t not in I]
                    if bad:
                        viol.append({"arm": arm_name, "repeat": rep, "ids": bad})
            for arm_name in ("ab", "ba"):
                if arm_name in row:
                    bad = [t for t in row[arm_name] if t not in I]
                    if bad:
                        viol.append({"arm": arm_name, "repeat": 0, "ids": bad})
            row["soundness_violations"] = viol
    return row


def _audit_stage(rows, *, db, dataset, k, rows_cap, sem_position) -> list[dict]:
    from src.Benchmark.datasource import load_query_table
    out: list[dict] = []
    for r in rows:
        if "error" in r or not r.get("witness"):
            continue
        shape = SHAPES[r["shape"]]
        added = sorted(set(r["opt_runs"][0]) - set(r["bno_runs"][0]))
        if any(spec.kind in audit_mod.NOT_AUDITABLE_KINDS for spec in shape.legs):
            out.append({"query": r["query"], "shape": r["shape"],
                        "verdict": "not_auditable",
                        "reason": "leg kind not scoreable (MC/SU/SJ)",
                        "records": []})
            continue
        df = load_query_table(dataset, r["query"], nrows=rows_cap)
        c0, c1 = select_columns(df)
        df.attrs["table_id"] = r["query"]
        legs = legs_mod.build_legs(shape, df, c0, c1, k, db=db, dataset=dataset,
                                   query_table_id=r["query"],
                                   sem_position=sem_position)
        names = ["a", "b"]
        rankings = {n: scored_mod.scored_ranking(leg, db)
                    for n, leg in zip(names, legs)}
        leg_ks = {n: leg.k for n, leg in zip(names, legs)}
        verdict = audit_mod.audit_witness(set(added), rankings, leg_ks)
        out.append({"query": r["query"], "shape": r["shape"],
                    "added": added, **verdict})
    return out


def _deadline(db, seconds):
    if not seconds:
        return contextlib.nullcontext()
    from src.Optimizer.training import query_deadline
    return query_deadline(db, seconds)


def run_theorem1(dataset, *, shape_names, k, repeats, rows_cap, arm_set,
                 width_grid, bno_construction, bno_order, k_semantics,
                 sem_position, sem_knobs, force_exact, verify_soundness,
                 resume_rows, queries, db, run, limit=None,
                 query_timeout=None) -> dict:
    from src.Benchmark.datasource import load_query_table, load_sidecar
    sctx = (_semantic_ctx(dataset, shape_names)
            if any(is_semantic(SHAPES[s]) for s in shape_names) else {})
    done = {(r["query"], r["shape"]) for r in resume_rows}
    rows: list[dict] = list(resume_rows)
    out_path = run.file("rows.jsonl")
    with out_path.open("w") as f:
        for r in resume_rows:
            f.write(json.dumps(r) + "\n")
    skipped = {"missing_csv": 0, "too_few_usable_columns": 0, "error": 0,
               "timeout": 0}
    target = limit if limit else len(queries)
    completed = attempted = 0
    if query_timeout:
        from src.Optimizer.training import apply_statement_timeout
        apply_statement_timeout(db, query_timeout)
    bar = tqdm(total=target * len(shape_names), desc=f"theorem1/{dataset}",
               unit="leg")
    with out_path.open("a") as f:
        for qt in queries:
            if completed >= target:
                break
            attempted += 1
            timed_out = skipped_query = False
            for sname in shape_names:
                if (qt, sname) in done:
                    bar.update(1)
                    continue
                shape = SHAPES[sname]
                try:
                    df = load_query_table(dataset, qt, nrows=rows_cap)
                except FileNotFoundError:
                    skipped["missing_csv"] += 1
                    skipped_query = True
                    break
                if select_columns(df) is None:
                    skipped["too_few_usable_columns"] += 1
                    skipped_query = True
                    break
                t0 = time.time()
                try:
                    with _deadline(db, query_timeout):
                        row = _eval_one(
                            shape, df, qt, k=k, repeats=repeats, db=db,
                            dataset=dataset, sctx=sctx, sem_knobs=sem_knobs,
                            arm_set=arm_set, width_grid=width_grid,
                            bno_construction=bno_construction,
                            bno_order=bno_order, k_semantics=k_semantics,
                            sem_position=sem_position,
                            force_exact=force_exact,
                            verify_soundness=verify_soundness)
                except fidelity.FidelityError:
                    raise
                except Exception as e:  # noqa: BLE001
                    if type(e).__name__ == "QueryTimeout":
                        skipped["timeout"] += 1
                        timed_out = True
                    else:
                        skipped["error"] += 1
                    row = {"query": qt, "shape": sname,
                           "error": f"{type(e).__name__}: {e}"}
                row["wall_s"] = round(time.time() - t0, 2)
                rows.append(row)
                f.write(json.dumps(row) + "\n")
                f.flush()
                bar.update(1)
                if timed_out:
                    break
            if not (timed_out or skipped_query):
                completed += 1
    bar.close()

    i2b, _ = load_sidecar(dataset)
    per_shape: dict = {}
    for sname in shape_names:
        g = [r for r in rows if r.get("shape") == sname and "error" not in r]
        w = [r for r in g if r.get("witness")]

        def _names(ids):
            return [i2b.get(int(t), str(t)) for t in ids]

        per_shape[sname] = {
            "both_stable": sum(bool(r.get("both_stable")) for r in g),
            "witnesses": len(w),
            "witness_bno_empty": sum(bool(r["witness_bno_empty"]) for r in w),
            "per_witness": [{
                "query": r["query"],
                "added_ids": sorted(set(r["opt_runs"][0]) - set(r["bno_runs"][0])),
                "dropped_ids": sorted(set(r["bno_runs"][0]) - set(r["opt_runs"][0])),
                "added_basenames": _names(
                    sorted(set(r["opt_runs"][0]) - set(r["bno_runs"][0]))),
                "dropped_basenames": _names(
                    sorted(set(r["bno_runs"][0]) - set(r["opt_runs"][0]))),
            } for r in w],
        }
    run.write_json("witnesses.json", per_shape)

    audit_rows = _audit_stage(rows, db=db, dataset=dataset, k=k,
                              rows_cap=rows_cap, sem_position=sem_position)
    run.write_json("audit.json", audit_rows)

    if verify_soundness:
        checked = [r for r in rows if "soundness_violations" in r]
        run.write_json("soundness.json", {
            "checked": len(checked),
            "violations": [
                {"query": r["query"], "shape": r["shape"], **v}
                for r in checked for v in r["soundness_violations"]],
        })

    summary = summary_mod.build_summary(
        rows, dataset=dataset, k=k, repeats=repeats, shape_names=shape_names,
        skipped=skipped, queries_total=attempted, arm_set=arm_set)
    summary["cohort"].update(completed_queries=completed, target_queries=target,
                             query_timeout_s=query_timeout)
    summary["audit_verdicts"] = {
        v: sum(a["verdict"] == v for a in audit_rows)
        for v in ("genuine", "tie_fragile", "not_truncation", "not_auditable")}
    run.write_table(summary_mod.flatten_rows(rows), summary)
    if not summary["cohort"]["cohort_complete"]:
        print(f"WARNING: cohort incomplete — skipped={skipped} "
              f"(every skip must be explained)",
              file=sys.stderr)
    return summary
