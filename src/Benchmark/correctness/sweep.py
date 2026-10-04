import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from statistics import mean, median
from time import perf_counter
from tqdm import tqdm
from src.Operators import Seekers
from src.Benchmark.datasource import load_sidecar, load_query_table, load_union_gt, load_join_gt
from src.Benchmark.correctness.refs import exact_semantic_ids
from src.Benchmark.correctness.arms import run_arm
from src.Benchmark.correctness.overlay import overlay
from src.Benchmark.correctness import syntactic as SY
from src.Benchmark import setdiff_metrics as SD

_EXACT_THRESHOLD = 10 ** 9  # comparison-only; NEVER use as k_coarse


def _semantic_leg(semantic_kind, df, k, *, exact, k_coarse, dataset, ef_search=None, query_col=None):
    overrides = {"dataset": dataset, "query_encode": "off"}
    cls = Seekers.SU if semantic_kind == "su" else Seekers.SJ
    extra = {} if semantic_kind == "su" else {"query_col_name": query_col}
    if exact:
        from src.Semantic.config import SemanticConfig, SemanticOp
        from src.Semantic.retrieve import IndexHandle
        cfg = SemanticConfig.load(overrides=overrides)
        sem_op = SemanticOp.SU if semantic_kind == "su" else SemanticOp.SJ
        op_cfg = cfg.operator(sem_op)
        n_vec = int(IndexHandle.open(cfg, op_cfg.approach, op_cfg.index_name).vector_count)
        return cls(df, k=k, exact_threshold=_EXACT_THRESHOLD, k_coarse=n_vec,
                   config_overrides=overrides, **extra)
    if ef_search is not None:
        overrides = {**overrides, "faiss_hnsw_ef_search": ef_search}
    return cls(df, k=k, k_coarse=k_coarse, config_overrides=overrides, **extra)


def build_legs(dataset, df, k, *, semantic_kind, inp, exact, k_coarse, db, ef_search=None, query_col=None):
    sem = _semantic_leg(semantic_kind, df, k, exact=exact, k_coarse=k_coarse, dataset=dataset,
                        ef_search=ef_search, query_col=query_col)
    sem.DB = db
    syn = SY.build_seeker(inp, df, k, db)
    return [sem, syn]


def _index_info(dataset, semantic_kind):
    from src.Semantic.config import SemanticConfig, SemanticOp
    from src.Semantic.retrieve import IndexHandle
    sem_op = SemanticOp.SU if semantic_kind == "su" else SemanticOp.SJ
    cfg = SemanticConfig.load(overrides={"dataset": dataset})
    oc = cfg.operator(sem_op)
    handle = IndexHandle.open(cfg, oc.approach, oc.index_name)
    return cfg, int(handle.vector_count)


def _timed_arm(make_legs, run, repeats):
    res, times = None, []
    for _ in range(max(1, repeats)):
        legs = make_legs()
        t0 = perf_counter()
        res = run(legs)
        times.append((perf_counter() - t0) * 1000.0)
    return res, median(times)


def run_sweep(dataset, *, plan, combine, k, widths, k_coarses, efs,
              ga_width, repeats, queries, db):
    if combine != "intersection":
        raise NotImplementedError(
            f"correctness redesign supports intersection plans (su_sc/sj_sc); got combine={combine!r}")
    semantic_kind = "su" if plan.startswith("su") else "sj"
    syntactic_kind = SY.SYNTACTIC_KIND[plan]
    i2b, _lake = load_sidecar(dataset)
    all_ids = sorted(i2b)
    _cfg, n_vec = _index_info(dataset, semantic_kind)
    op_k_coarse = _cfg.faiss_k_coarse
    op_ef = _cfg.faiss_hnsw_ef_search
    w_gen = max(widths) if widths else k
    gt: dict | None = None
    try:
        if plan == "su_sc":
            gt = load_union_gt(dataset)
        elif plan == "sj_sc":
            gt = load_join_gt(dataset)
    except FileNotFoundError:
        pass

    rows: list[dict] = []
    skipped = {"missing_csv": 0, "unusable_query": 0, "no_syntactic_cols": 0}
    for q in tqdm(queries, desc=f"correctness/{dataset}/{plan}"):
        qtable, qcol = q if isinstance(q, tuple) else (q, None)
        try:
            df = load_query_table(dataset, qtable)
        except FileNotFoundError:
            skipped["missing_csv"] += 1
            continue
        df.attrs["table_id"] = qtable
        try:
            ex_seeker = _semantic_leg(semantic_kind, df, len(all_ids), exact=True, k_coarse=None,
                                      dataset=dataset, query_col=qcol); ex_seeker.DB = db
            exact_ranking = exact_semantic_ids(ex_seeker, all_ids)
            np_seeker = _semantic_leg(semantic_kind, df, k, exact=False, k_coarse=None,
                                      dataset=dataset, query_col=qcol); np_seeker.DB = db
            nopush_sem = np_seeker.run("")
        except KeyError:
            skipped["unusable_query"] += 1
            continue
        inp = SY.select_inputs(syntactic_kind, df, qcol)
        if inp is None:
            skipped["no_syntactic_cols"] += 1
            continue
        full_set, full_order = SY.full(inp, df, db)
        syn_topk = SY.topk_set(inp, df, k, db, full_order=full_order)
        r_nopush = [i for i in nopush_sem if i in syn_topk]
        r_exact = [i for i in exact_ranking[:k] if i in syn_topk]
        if gt is None:
            q_gt = None
        elif plan == "sj_sc":
            q_gt = gt.get((qtable, qcol))
        else:
            q_gt = gt.get(qtable)

        def op_row(arm, ids, runtime_ms):
            row = _row(qtable, arm, ids, r_nopush, r_exact, point_kind="operating",
                       runtime_ms=runtime_ms)
            if q_gt is not None and arm in ("OPT", "REV", "GA", "GB"):
                row.update(overlay(dropped_ids=set(r_exact) - set(ids),
                                   added_ids=set(ids) - set(r_exact),
                                   relevant=q_gt, i2b=i2b))
            rows.append(row)

        op_row("R-NOPUSH", r_nopush, 0.0)
        op_row("R-EXACT", r_exact, 0.0)

        def mk():
            return build_legs(dataset, df, k, semantic_kind=semantic_kind, inp=inp,
                              exact=False, k_coarse=None, db=db, query_col=qcol)
        op_row("OPT", *_timed_arm(mk, lambda L: run_arm(L, combine, "cost", db, k, guard="none"), repeats))
        op_row("REV", *_timed_arm(mk, lambda L: run_arm(L, combine, [1, 0], db, k, guard="none"), repeats))
        op_row("GA", *_timed_arm(mk, lambda L: run_arm(L, combine, "cost", db, k, guard="A", width=ga_width), repeats))
        op_row("GB", *_timed_arm(mk, lambda L: run_arm(L, combine, "cost", db, k, guard="B"), repeats))

        if not r_exact:
            continue
        for w in widths:
            ids = SY.derive(full_set, full_order, exact_ranking[:w], k)
            rows.append(_row(qtable, "RECOVERY", ids, r_nopush, r_exact,
                             point_kind="recovery", axis="width", axis_val=w, exact_source=True))
        for kc in k_coarses:
            leg = _semantic_leg(semantic_kind, df, w_gen, exact=False, k_coarse=kc, dataset=dataset,
                                ef_search=op_ef, query_col=qcol); leg.DB = db
            ids = SY.derive(full_set, full_order, leg.run(""), k)
            rows.append(_row(qtable, "RECOVERY", ids, r_nopush, r_exact,
                             point_kind="recovery", axis="k_coarse", axis_val=kc, k_coarse_used=kc))
        for ef in efs:
            leg = _semantic_leg(semantic_kind, df, w_gen, exact=False, k_coarse=n_vec, dataset=dataset,
                                ef_search=ef, query_col=qcol); leg.DB = db
            ids = SY.derive(full_set, full_order, leg.run(""), k)
            rows.append(_row(qtable, "RECOVERY", ids, r_nopush, r_exact,
                             point_kind="recovery", axis="ef", axis_val=ef, k_coarse_used=n_vec))

    summary = _summarize(rows, queries, dataset=dataset, plan=plan, k=k,
                         operating={"k_coarse": op_k_coarse, "ef": op_ef,
                                    "exact_threshold": _cfg.exact_threshold,
                                    "ga_width": ga_width})
    summary["skipped"] = skipped
    return rows, summary


def _row(query, arm, ids, r_nopush, r_exact, *, point_kind, runtime_ms=0.0,
         axis=None, axis_val=None, exact_source=False, k_coarse_used=None):
    return {
        "query": query,
        "arm": arm,
        "point_kind": point_kind,
        "axis": axis,
        "axis_val": axis_val,
        "exact_source": exact_source,
        "k_coarse_used": k_coarse_used,
        "retention_vs_exact": SD.retention(ids, r_exact),
        "added_vs_exact": SD.additions(ids, r_exact),
        "rbo_vs_exact": SD.rbo(ids, r_exact),
        "retention_vs_nopush": SD.retention(ids, r_nopush),
        "additions_vs_nopush": SD.additions(ids, r_nopush),
        "n_retrieved": len(ids),
        "n_exact": len(r_exact),
        "n_nopush": len(r_nopush),
        "ids": list(ids),
        "runtime_ms": runtime_ms,
    }


def _mean(xs, default):
    xs = list(xs)
    return mean(xs) if xs else default


def _op(rows, arm):
    return [r for r in rows if r["point_kind"] == "operating" and r["arm"] == arm]


def _op_ex(rows, arm):
    return [r for r in _op(rows, arm) if r["n_exact"] > 0]


def _retention(rows, arm):
    return round(_mean((r["retention_vs_exact"] for r in _op_ex(rows, arm)), 1.0), 4)


def _curve(rows, axis):
    by: dict = {}
    for r in rows:
        if r["point_kind"] == "recovery" and r["axis"] == axis and r["n_exact"] > 0:
            by.setdefault(r["axis_val"], []).append(r["retention_vs_exact"])
    return {str(g): round(mean(v), 4) for g, v in sorted(by.items()) if v}


def _summarize(rows, queries, *, dataset, plan, k, operating):
    ret_opt, ret_rev = _retention(rows, "OPT"), _retention(rows, "REV")

    opt_by_q = {r["query"]: set(r["ids"]) for r in _op_ex(rows, "OPT")}
    rev_by_q = {r["query"]: set(r["ids"]) for r in _op_ex(rows, "REV")}
    shared = sorted(opt_by_q.keys() & rev_by_q.keys())
    jds, changed = [], 0
    for q in shared:
        a, b = opt_by_q[q], rev_by_q[q]
        union = a | b
        jds.append(1.0 - (len(a & b) / len(union) if union else 1.0))
        changed += (a != b)
    a_order = {
        "retention_opt_vs_exact": ret_opt,
        "retention_rev_vs_exact": ret_rev,
        "opt_vs_rev_jaccard_dist": round(_mean(jds, 0.0), 4),
        "queries_changed_frac": round(changed / len(shared), 4) if shared else 0.0,
        "n_queries": len(shared),
    }

    gt_used = plan in ("su_sc", "sj_sc")
    opt_ex = _op_ex(rows, "OPT")
    b_harm = {
        "opt_dropped_frac_vs_exact": round(1.0 - ret_opt, 4),
        "opt_mean_added_vs_exact": round(_mean((r["added_vs_exact"] for r in opt_ex), 0.0), 3),
        "lead_gt_losses_distinct": (sum(r.get("real_losses", 0) for r in opt_ex) if gt_used else None),
        "lead_gt_gains_distinct": (sum(r.get("real_gains", 0) for r in opt_ex) if gt_used else None),
        "gt_note": (
            ("GT = LEAD seeker's standalone discovery GT (union for SU / join "
             "for SJ), aligned with the semantic leg's task, NOT the composed "
             "SU/SJ ∩ SC plan. losses = relevant tables in the exact plan "
             "output OPT dropped (defensible lower bound); gains = lead-seeker-"
             "relevant tables OPT surfaced beyond the exact plan (drift toward "
             "the lead seeker, not a composed-task quality gain). distinct "
             "(query,table), n_exact>0.")
            if gt_used else
            ("GT overlay omitted: the Theorem-1 divergence is measured GT-free "
             "against R-EXACT/R-NOPUSH; the only available GT is the lead "
             "seeker's standalone discovery GT, which is even less aligned with "
             "the composed (unionable-AND-correlated / joinable-AND-multicolumn) "
             "task than for SC, so overlaying it would be more misleading, not "
             "less.")),
        "attribution": {
            "approx_only_dropped_frac": round(1.0 - _retention(rows, "R-NOPUSH"), 4),
            "pushdown_added_dropped_frac": round(
                1.0 - _mean((r["retention_vs_nopush"] for r in opt_ex if r["n_nopush"] > 0), 1.0), 4),
        },
    }

    by_w = _curve(rows, "width")
    floor = next((int(w) for w in sorted((int(x) for x in by_w), reverse=False)
                  if by_w[str(w)] >= 0.99), None)
    c_recovery = {
        "by_result_width_exact_source": by_w,
        "by_k_coarse": _curve(rows, "k_coarse"),
        "by_ef_search": _curve(rows, "ef"),
        "width_for_retention_0.99_exact_source": floor,
        "ga_retention_vs_exact": _retention(rows, "GA"),
        "gb_retention_vs_exact": _retention(rows, "GB"),
    }

    rt = {arm: round(_mean((r["runtime_ms"] for r in _op(rows, arm)), 0.0), 3)
          for arm in ("OPT", "REV", "GA", "GB")}
    d_time = {
        "runtime_ms_median": rt,
        "speedup_opt_over_rev": round(rt["REV"] / rt["OPT"], 3) if rt["OPT"] else None,
        "guard_a_cost_frac": round(rt["GA"] / rt["OPT"] - 1.0, 4) if rt["OPT"] else None,
    }

    return {
        "dataset": dataset, "plan": plan, "k": k, "operating_point": operating,
        "coverage": {
            "rows_total": len(rows),
            "queries_total": len(queries),
            "queries_vs_exact": len({r["query"] for r in rows if r["n_exact"] > 0}),
        },
        "answers": {"a_order": a_order, "b_harm": b_harm,
                    "c_recovery": c_recovery, "d_time": d_time},
    }
