from __future__ import annotations

import argparse
import sys

from src.Benchmark.ann_frontier import PG_VARIANTS
from src.Benchmark.runspec import Field, backend_slug, git_sha, open_run, runtime_info

_PAPER_LABEL = {
    "union": "p@k/recall/MAP are paper metrics; MRR is thesis-added",
    "join": "runtime is the paper metric; p@k/recall/MAP/MRR are thesis-added",
    "imputation": "runtime is the paper metric; hit-rate is thesis-added",
    "negex": "runtime only (paper metric); no accuracy — refinement op, not retrieval (see adapter docstring)",
    "correlation": "P@k/R@k are paper metrics (table-level, synthetic GT); F1 is thesis-added",
}


def _positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {value}")
    return value


def _add_tag(*parsers) -> None:
    for sp in parsers:
        sp.add_argument("--tag", default=None,
                        help="human label inserted into the run-dir name")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bench")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--dataset", required=True)
    r.add_argument("--task", required=True,
                   choices=["union", "join", "imputation", "negex", "correlation"])
    r.add_argument("--k", type=int, required=True)
    r.add_argument("--seeker", choices=["sc", "sho", "su", "sj"], default="sc",
                   help="seeker leg for union/join (default sc = today's behavior)")
    r.add_argument("--pair-hits", dest="pair_hits", action="store_true",
                   help="write a pair-hits sidecar parquet (union/join only)")

    bm = sub.add_parser("bench")
    bm.add_argument("--dataset", required=True,
                    help="query/GT file PREFIX; lake/index = config [Dataset].name")
    bm.add_argument("--tasks", default="union,join")
    bm.add_argument("--seekers", default="sc,sho,liftus,snoopy,deepjoin",
                    help="methods: sc, sho, or approach[:index_name] (bare -> default)")
    bm.add_argument("--repeat", type=int, default=1, help="full re-runs per leg (median)")
    bm.add_argument("--k", default=None, help="comma k-sweep (default: K_RANGE)")
    bm.add_argument("--k-coarse", dest="k_coarse", type=int, default=None)
    bm.add_argument("--k-vote", dest="k_vote", type=int, default=None)
    bm.add_argument("--k-final", dest="k_vote", type=int,
                    help="deprecated alias for --k-vote")
    bm.add_argument("--vote-factor", dest="vote_factor", type=_positive_float, default=None,
                    help="override [Semantic.<OP>] vote_depth_factor for the "
                         "semantic legs; --k-vote still wins as an absolute pin")
    bm.add_argument("--out", default=None)
    bm.add_argument("--limit", type=int, default=None, help="cap #queries (smoke runs)")

    pr = sub.add_parser("prepare")
    pr.add_argument("--dataset", required=True)
    pr.add_argument("--task", required=True, choices=["imputation", "negex", "correlation"])
    pr.add_argument("--n", type=int, default=1000)
    pr.add_argument("--k", type=int, default=10)
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--force", action="store_true")
    pr.add_argument("--require-semantic", dest="require_semantic", action="store_true",
                    help="imputation: only pick key columns enrolled in the [Semantic.SJ] "
                         "index, falling through to the next eligible key per table")

    cp = sub.add_parser("compare")
    cp.add_argument("--dataset", required=True)
    cp.add_argument("--task", required=True, choices=["imputation"])
    cp.add_argument("--k-grid", dest="k_grid", default="10,25,50,100",
                    type=lambda s: [int(x) for x in s.split(",")])
    cp.add_argument("--legs", default=None,
                    help="comma-separated subset of legs to run (default: all)")
    cp.add_argument("--k-coarse", dest="k_coarse", type=int, default=None,
                    help="FAISS coarse depth for semantic legs (default: config faiss_k_coarse)")
    cp.add_argument("--resolve-cache-max", dest="resolve_cache_max", type=int, default=None,
                    help="LRU bound on cached candidate tables in resolve (default: unbounded)")
    cp.add_argument("--warmup", type=int, default=1,
                    help="discarded runs per (leg,k) before timing; 0 = off. Keeps the "
                         "index open / first-query buffer misses out of the reported mean")
    cp.add_argument("--mc-cost-cap", dest="mc_cost_cap", type=int, default=None,
                    help="drop queries whose predicted MC scan exceeds N index rows, "
                         "from EVERY leg (needs optimizer freqs.csv). Scores are always "
                         "reported; without this they are only reported")

    csw = sub.add_parser("correctness-sweep")
    csw.add_argument("--dataset", required=True)
    csw.add_argument("--plan", required=True, choices=["su_sc", "sj_sc", "su_c", "sj_mc"])
    csw.add_argument("--k", type=int, default=10)
    csw.add_argument("--width-grid", dest="widths", default="10,25,50,100,1000",
                     type=lambda s: [int(x) for x in s.split(",")])
    csw.add_argument("--k-coarse-grid", dest="k_coarses", default="50,100,250,500,1000",
                     type=lambda s: [int(x) for x in s.split(",")])
    csw.add_argument("--ef-grid", dest="efs", default="16,32,64,128,256",
                     type=lambda s: [int(x) for x in s.split(",")])
    csw.add_argument("--ga-width", dest="ga_width", type=int, default=200,
                     help="Guard A pushdown filter width at the operating point")
    csw.add_argument("--repeats", type=int, default=1, help="timed repeats per arm (median)")
    csw.add_argument("--limit", type=int, default=None, help="cap #queries (smoke runs)")

    ct = sub.add_parser("correctness-theorem1")
    ct.add_argument("--dataset", required=True)
    ct.add_argument("--shapes", default="sc_sc_flat,sc_sc_prov,sc_kw",
                    type=lambda s: [x.strip() for x in s.split(",") if x.strip()])
    ct.add_argument("--k", type=int, default=10,
                    help="the COMBINER's k; one k per run")
    ct.add_argument("--repeats", type=int, default=1,
                    help=">1 enables witnesses + nondeterminism floor")
    ct.add_argument("--rows", type=int, default=5000,
                    help="query-table row cap; MC shapes need 50")
    ct.add_argument("--arms", default="a3",
                    type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                    help="subset of a3,a1,width")
    ct.add_argument("--width-grid", dest="width_grid",
                    default="10,25,50,100,250,500,1000,2500,none",
                    type=lambda s: [None if x.strip() == "none" else int(x)
                                    for x in s.split(",")])
    ct.add_argument("--bno-construction", dest="bno_construction",
                    choices=["slice", "direct"], default="slice",
                    help="slice = the published construction")
    ct.add_argument("--bno-order", dest="bno_order",
                    choices=["lega", "terminal"], default="lega")
    ct.add_argument("--k-semantics", dest="k_semantics",
                    choices=["faithful", "distinct"], default="faithful")
    ct.add_argument("--sem-position", dest="sem_position",
                    choices=["first", "last"], default="first",
                    help="last sets order_cost=8 on the semantic leg")
    ct.add_argument("--k-coarse", dest="k_coarse", type=int, default=None,
                    help="pins the DEPLOYED semantic legs; gate is on the "
                         "resolved value")
    ct.add_argument("--ef", type=int, default=None,
                    help="beam; defaults to --k-coarse. faiss refuses ef<k_coarse")
    ct.add_argument("--exact-threshold", dest="exact_threshold", type=int,
                    default=None, help="explicit only; the config exact_threshold is never inherited")
    ct.add_argument("--force-exact", dest="force_exact", action="store_true",
                    help="semantic legs take the exact path whenever a filter "
                         "is present")
    ct.add_argument("--verify-soundness", dest="verify_soundness",
                    action="store_true", help="opt/bno/ab/ba subset-of-I; needs width arm")
    ct.add_argument("--allow-measured-costs", dest="allow_measured_costs",
                    action="store_true",
                    help="override the cost_basis=logical gate (recorded)")
    ct.add_argument("--limit", type=int, default=None,
                    help="COMPLETION target: run until N queries finished, "
                         "drawing replacements for skipped/capped ones")
    ct.add_argument("--query-timeout", dest="query_timeout", type=float,
                    default=None,
                    help="wall cap in seconds per (query, shape); a capped "
                         "query is recorded as timeout, its remaining shapes "
                         "skipped, and the next present query drawn instead "
                         "(duckdb: client-side interrupt + async-exc; "
                         "postgres: statement_timeout)")
    ct.add_argument("--resume-from", dest="resume_from", default=None,
                    help="skip (query,shape) complete in that run dir; writes "
                         "a NEW dir")
    ct.add_argument("--out", default=None)

    tb = sub.add_parser("tiebreak-bench")
    tb.add_argument("--dataset", required=True)
    tb.add_argument("--k", type=int, default=10)
    tb.add_argument("--op", choices=["SU", "SJ"], default="SU",
                    help="which same-type pair to order. SU legs are whole query "
                         "tables; SJ legs are single columns, where the n_cols "
                         "placeholder always ties so the comparison is vs chance")
    tb.add_argument("--k-coarse-grid", dest="k_coarse_grid", default="100,1000",
                    type=lambda s: [int(x) for x in s.split(",")],
                    help="k_coarse values to pair across (default 100,1000)")
    tb.add_argument("--repeats", type=int, default=3, help="timed repeats per leg (median)")
    tb.add_argument("--limit", type=int, default=None, help="cap #query tables (smoke runs)")
    tb.add_argument("--seed", type=int, default=0)
    tb.add_argument("--max-pairs", dest="max_pairs", type=int, default=200_000,
                    help="cap on pairs compared; the pair set is QUADRATIC in "
                         "queries x |k_coarse_grid|, so an unlimited run is tens "
                         "of millions of rows. Above the cap, pairs are sampled "
                         "uniformly and it says so. 0 = no cap (memory-hungry)")
    tb.add_argument("--sample-queries", dest="sample_queries", type=int, default=None,
                    help="sample N query tables from the index instead of the curated "
                         "union query set (for synthetic / curated-query-less datasets)")

    pb = sub.add_parser("pushdown-bench")
    pb.add_argument("--dataset", required=True)
    pb.add_argument("--k", type=int, default=10)
    pb.add_argument("--selectivities", default="0.01,0.05,0.10,0.25,0.50,0.90",
                    type=lambda s: [float(x) for x in s.split(",")])
    pb.add_argument("--comps", default="0,k/2,k",
                    help="absolute candidate counts; tokens 'k' and 'k/2' resolve against --k")
    pb.add_argument("--modes", default="prefilter,postfilter",
                    type=lambda s: [x.strip() for x in s.split(",")])
    pb.add_argument("--exact-threshold", dest="exact_threshold", type=int, default=0,
                    help="0 forces the ANN path so prefilter/postfilter can diverge")
    pb.add_argument("--k-coarse", dest="k_coarse", type=int, default=None,
                    help="FAISS coarse depth (default: config faiss_k_coarse)")
    pb.add_argument("--ef", dest="ef", type=int, default=None,
                    help="FAISS beam for this run (default: config faiss_hnsw_ef_search). "
                         "Set it equal to --k-coarse: a narrower beam cannot fill the "
                         "fetch, so the run measures a depth it never reached")
    pb.add_argument("--decoy-band", dest="decoy_band", nargs=2, type=int, default=None,
                    metavar=("LO", "HI"),
                    help="draw the allowed set's decoys from ranks LO:HI of the exact "
                         "semantic ranking instead of from anywhere outside the top-2k "
                         "(the default). A band just below the answers makes the decoys "
                         "plausible-but-wrong rather than deep tail, so sweeping it "
                         "toward the answers measures how much of the exact-vs-ANN "
                         "advantage survives a harder filter. The top-2k is excluded "
                         "either way; omit for the pre-band behaviour")
    pb.add_argument("--repeats", type=int, default=3, help="timed repeats per arm (median)")
    pb.add_argument("--no-warmup", dest="warmup", action="store_false",
                    help="skip the untimed warm-up run before each arm's timed "
                         "repeats. The warm-up exists because the mode loop is "
                         "innermost: a post-filter arm scans the whole lake "
                         "between one cell's pre-filter repeats and the next, "
                         "which inflated pgvector pre-filter timings 25-80%%. "
                         "Pass this only to reproduce a run made without the warm-up")
    pb.add_argument("--limit", type=int, default=None, help="cap #queries")
    pb.add_argument("--seed", type=int, default=0)

    af = sub.add_parser("ann-frontier")
    af.add_argument("--dataset", required=True)
    af.add_argument("--op", choices=["SU", "SJ"], default="SU")
    af.add_argument("--k-coarse", dest="k_coarse", type=int, default=None,
                    help="retrieval LIMIT (default: config faiss_k_coarse)")
    af.add_argument("--ef-grid", dest="efs", default="64,128,250,500",
                    type=lambda s: [int(x) for x in s.split(",")])
    af.add_argument("--queries", type=int, default=40, help="#query tables sampled")
    af.add_argument("--seed", type=int, default=0)
    af.add_argument("--approach", default=None,
                    help="override [Semantic.<OP>] approach, so one loop can "
                         "cover lakes with different encoders")
    af.add_argument("--index-name", dest="index_name", default=None,
                    help="override [Semantic.<OP>] index_name")
    af.add_argument("--variants", default="baseline,relaxed,insql,binary,batch,best",
                    help=f"pgvector variants to measure, or 'none'; "
                         f"available: {','.join(PG_VARIANTS)}")
    af.add_argument("--no-faiss", dest="with_faiss", action="store_false",
                    help="skip the faiss arm (pgvector-only sweep)")
    af.add_argument("--io-stats", dest="io_stats", action="store_true",
                    help="run the pgvector sweep twice and verify shared_buffers "
                         "holds the working set (pass-2 disk reads must be ~0); "
                         "exits nonzero if not. Doubles the pgvector runtime.")

    rr = sub.add_parser("regime-recall")
    rr.add_argument("--pairs", nargs="+", required=True, help="pair-hits parquet(s)")
    rr.add_argument("--registry", required=True, help="overlap registry CSV")
    rr.add_argument("--dataset", default=None)
    rr.add_argument("--task", choices=["join", "union", "auto"], default="auto")
    rr.add_argument("--strata", action="store_true")
    rr.add_argument("--nonnumeric", action="store_true")
    rr.add_argument("--pool-size", dest="pool_size", type=int, default=None)
    rr.add_argument("--k", default="1,5,10,25,50")
    rr.add_argument("--out", default=None)

    cr = sub.add_parser("condensed-recall")
    cr.add_argument("--pairs", nargs="+", required=True, help="pair-hits parquet(s)")
    cr.add_argument("--registry", required=True,
                    help="overlap registry CSV carrying n_inter per GT pair")
    cr.add_argument("--dataset", default=None)
    cr.add_argument("--task", choices=["join", "union", "auto"], default="auto")
    cr.add_argument("--k", default="1,5,10,25,50,150")
    cr.add_argument("--out", default=None)

    pl = sub.add_parser("plan")
    pls = pl.add_subparsers(dest="plan_cmd", required=True)
    pls.add_parser("list")
    plr = pls.add_parser("run")
    plr.add_argument("name")
    plr.add_argument("--dataset", required=True)
    plr.add_argument("--task", choices=["union", "join"], required=True)
    plr.add_argument("--k", type=int, required=True)
    plr.add_argument("--seed", type=int, default=0)
    plr.add_argument("--out", default=None)
    plr.add_argument("--limit", type=int, default=None)

    _add_tag(r, bm, cp, csw, ct, tb, pb, af, rr, cr, plr)
    return p


def _theorem1_config_snapshot() -> dict:
    import configparser
    from src import paths
    parser = configparser.ConfigParser()
    parser.read(paths.config_path())
    return {sec: dict(parser[sec]) for sec in parser.sections()
            if sec == "Database" or sec.startswith("Semantic")
            or sec == "Optimizer"}


def _run_theorem1(args, parser) -> int:
    import json
    import sys as _sys
    from pathlib import Path
    from src.Benchmark.correctness.theorem1 import fidelity
    from src.Benchmark.correctness.theorem1.shapes import (
        SCOREABLE_KINDS, SHAPES, has_mc, is_semantic, terminal_leg)

    unknown = [s for s in args.shapes if s not in SHAPES]
    if unknown:
        parser.error(f"unknown shapes {unknown}; available: {list(SHAPES)}")
    mc = [s for s in args.shapes if has_mc(SHAPES[s])]
    if mc and len(mc) != len(args.shapes):
        parser.error(
            "MC and non-MC shapes in one invocation are refused: "
            "MC ran at --rows 50 and the SC shapes at 5000, so their "
            "magnitudes are not comparable in one run dir. Two invocations, "
            "two run dirs, two manifests.")
    bad = set(args.arms) - {"a3", "a1", "width"}
    if bad:
        parser.error(f"unknown arms {sorted(bad)}; choose from a3,a1,width")
    if args.verify_soundness and "width" not in args.arms:
        parser.error("--verify-soundness needs the width arm (it re-checks "
                     "membership in I = full(a) AND full(b))")
    if args.bno_order == "terminal":
        for s in args.shapes:
            t = terminal_leg(SHAPES[s], sem_position=args.sem_position)
            if t.kind not in SCOREABLE_KINDS:
                parser.error(f"--bno-order terminal: shape {s} terminates in "
                             f"{t.kind}, which has no scored ranking")
    if mc and args.rows > 50:
        print(f"WARNING: MC shapes at --rows {args.rows}; the published MC "
              "numbers used --rows 50 and an uncapped MC query blew "
              "past 200s per leg", file=_sys.stderr)

    from src.Benchmark.correctness.theorem1 import run as T1
    from src.Benchmark.db import open_dataset_db
    from src.Benchmark.datasource import load_union_queries

    db = open_dataset_db(args.dataset)
    try:
        rt = runtime_info(args.dataset, db=db)
        fid = fidelity.check_startup(
            db, allow_measured_costs=args.allow_measured_costs)
        sem_knobs = None
        if any(is_semantic(SHAPES[s]) for s in args.shapes):
            from src.Semantic.config import SemanticConfig
            cfg = SemanticConfig.load(overrides={"dataset": args.dataset})
            sem_knobs = fidelity.resolve_semantic_knobs(
                cfg, cli_k_coarse=args.k_coarse, cli_ef=args.ef,
                cli_exact_threshold=args.exact_threshold)
        resume_rows = []
        if args.resume_from:
            prior = Path(args.resume_from) / "rows.jsonl"
            if not prior.is_file():
                parser.error(f"--resume-from: no rows.jsonl under {args.resume_from}")
            resume_rows = [json.loads(line)
                           for line in prior.read_text().splitlines()
                           if line.strip()]
            resume_rows = [r for r in resume_rows if "error" not in r]
        args.resume_from = str(args.resume_from) if args.resume_from else None
        queries, n_absent = T1.present_queries(
            args.dataset, load_union_queries(args.dataset))
        if n_absent:
            print(f"note: {n_absent} query tables absent from this lake's "
                  f"csvs/ (split lake); cohort = {len(queries)} present",
                  file=_sys.stderr)
        config = {**rt, "fidelity": fid, "semantic_knobs": sem_knobs,
                  "config_snapshot": _theorem1_config_snapshot()}
        values = T1.field_values(args, backend_slug(rt))
        with open_run("correctness-theorem1", lake=args.dataset,
                      fields=T1.FIELDS, values=values, config=config,
                      tag=args.tag, out=args.out) as run:
            if resume_rows:
                run.set_config(resumed_from=args.resume_from,
                               resumed_rows=len(resume_rows))
            if n_absent:
                run.set_config(absent_query_tables=n_absent)
            summary = T1.run_theorem1(
                args.dataset, shape_names=args.shapes, k=args.k,
                repeats=args.repeats, rows_cap=args.rows,
                arm_set=set(args.arms), width_grid=args.width_grid,
                bno_construction=args.bno_construction,
                bno_order=args.bno_order, k_semantics=args.k_semantics,
                sem_position=args.sem_position, sem_knobs=sem_knobs,
                force_exact=args.force_exact,
                verify_soundness=args.verify_soundness,
                resume_rows=resume_rows, queries=queries, db=db, run=run,
                limit=args.limit, query_timeout=args.query_timeout)
            from src.Benchmark.correctness.theorem1.summary import print_summary
            print_summary(summary)
            print(f"wrote {run.path}")
    finally:
        db.close()
    return 0


def _parse_comps(spec: str, k: int) -> list[int]:
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok == "k":
            out.append(k)
        elif tok == "k/2":
            out.append(k // 2)
        else:
            out.append(int(tok))
    return out


def _run_plan(args) -> int:
    from src.Benchmark.db import open_dataset_db
    from src.Benchmark.plans import PLANS
    from src.Benchmark.plans.registry import PlanContext
    from src.Benchmark.plans.writer import write_plan_results
    from src.Semantic.config import SemanticConfig, SemanticOp

    if args.name not in PLANS:
        raise SystemExit(f"unknown bench plan {args.name!r}; "
                         f"available: {list(PLANS)}")
    spec = PLANS[args.name]
    cfg = SemanticConfig.load(overrides={"dataset": args.dataset})
    op = SemanticOp.SU if args.task == "union" else SemanticOp.SJ
    try:
        oc = cfg.operator(op)
        approach, index_name = oc.approach, oc.index_name
    except KeyError:
        approach = index_name = None
    snapshot = {"dataset": args.dataset, "task": args.task,
                "vector_backend": cfg.vector_backend, "k": args.k,
                "faiss_k_coarse": cfg.faiss_k_coarse,
                "approach": approach, "index_name": index_name}
    db = open_dataset_db(args.dataset)
    rt = runtime_info(args.dataset, db=db)
    snapshot["dbms"] = rt["dbms"]
    fields = (Field("plan"), Field("task"), Field("k", prefix="k"),
              Field("seed", prefix="seed", default=0),
              Field("limit", prefix="lim"), Field("backend"))
    values = {"plan": args.name, "task": args.task, "k": args.k,
              "seed": args.seed, "limit": args.limit, "backend": backend_slug(rt)}
    try:
        with open_run("plans", lake=args.dataset, fields=fields, values=values,
                      config=rt, tag=args.tag, out=args.out) as run:
            ctx = PlanContext(
                dataset=args.dataset, db=db, task=args.task, k=args.k,
                seed=args.seed, out_dir=run.path, git_sha=git_sha(),
                config_snapshot=snapshot, limit=args.limit)
            result = spec.run(ctx)
            path = write_plan_results(result, k=args.k, out_dir=run.path)
    finally:
        db.close()

    print(f"\n=== {result.plan} {args.dataset}/{args.task} (k={args.k}) ===")
    print(f"point: {result.point}")
    print(f"  {'arm':<18}{'recall':>9}{'prec':>9}{'ndcg':>9}{'MAP':>9}{'ms/q':>9}")
    for a in result.arms:
        s = a.summary
        print(f"  {a.name:<18}{s['mean_recall']:>9.4f}{s['mean_precision']:>9.4f}"
              f"{s['mean_ndcg']:>9.4f}{s['MAP']:>9.4f}{s['mean_runtime_ms']:>9.1f}")
    print("deltas vs seeker_only:")
    for pair, metrics in result.deltas.items():
        for name, val in metrics.items():
            print(f"  {pair}.{name}: {val:+.4f}")
    print(f"wrote {path}")
    return 0


def _dispatch_run(task: str, dataset: str, k: int, seeker: str = "sc",
                  pair_sink: list | None = None):
    if task == "union":
        from src.Benchmark.adapters.union import run_union
        return run_union(dataset, k, seeker=seeker, pair_sink=pair_sink)
    if task == "join":
        from src.Benchmark.adapters.join import run_join
        return run_join(dataset, k, seeker=seeker, pair_sink=pair_sink)
    if task == "imputation":
        from src.Benchmark.adapters.imputation import run_imputation
        return run_imputation(dataset, k)
    if task == "negex":
        from src.Benchmark.adapters.negative_example import run_negex
        return run_negex(dataset, k)
    if task == "correlation":
        from src.Benchmark.adapters.correlation import run_correlation
        return run_correlation(dataset, k)
    raise ValueError(task)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "run":
        if args.task not in ("union", "join") and (args.seeker != "sc" or args.pair_hits):
            parser.error("--seeker/--pair-hits only apply to union/join")
        if args.task == "union" and args.seeker == "sj":
            parser.error("--seeker sj is join-only")
        if args.task == "join" and args.seeker == "su":
            parser.error("--seeker su is union-only")
        sink = [] if args.pair_hits else None
        rt = runtime_info(args.dataset)
        fields = (Field("task"), Field("seeker"), Field("k", prefix="k"),
                  Field("pair_hits", prefix="ph"), Field("backend"))
        values = {"task": args.task, "seeker": args.seeker, "k": args.k,
                  "pair_hits": args.pair_hits, "backend": backend_slug(rt)}
        with open_run("run", lake=args.dataset, fields=fields, values=values,
                      config=rt, tag=args.tag) as run:
            rows, summary = _dispatch_run(args.task, args.dataset, args.k,
                                          seeker=args.seeker, pair_sink=sink)
            print(f"\n=== {args.task} {args.dataset} (k={args.k}) ===")
            print(f"metrics: {_PAPER_LABEL[args.task]}")
            print(f"evaluated={summary['evaluated']} / {summary['total_queries']} "
                  f"(skipped={summary['skipped']}, effective_n={summary['effective_n']})")
            metric_keys = sorted(
                k for k in summary if k.startswith("mean_") or k in ("MAP", "MRR")
            )
            for key in metric_keys:
                if key in summary:
                    print(f"  {key}: {summary[key]:.4f}")
            print(f"  wall: {summary['wall_s']:.1f}s")
            print(f"wrote {run.write_table(rows, summary)}")
            if sink is not None:
                import pandas as pd
                pd.DataFrame(sink).to_parquet(run.file("pair_hits.parquet"))
                print(f"wrote {run.file('pair_hits.parquet')}")
        return 0
    if args.cmd == "bench":
        from pathlib import Path
        from src.Benchmark.judit import runner
        from src.Benchmark.judit.metrics import K_RANGE
        from src.Semantic.config import SemanticConfig
        cfg = SemanticConfig.load()
        ks = tuple(int(x) for x in args.k.split(",")) if args.k else K_RANGE
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
        methods = [m.strip() for m in args.seekers.split(",") if m.strip()]
        rt = runtime_info(None)
        fields = (Field("prefix", prefix="p"), Field("tasks"), Field("seekers"),
                  Field("repeat", prefix="r", default=1),
                  Field("k_coarse", prefix="kc", default=None),
                  Field("k_vote", prefix="kv", default=None),
                  Field("vote_factor", prefix="vf", default=None),
                  Field("k", prefix="k", default=list(K_RANGE)),
                  Field("limit", prefix="lim"), Field("backend"))
        values = {"prefix": args.dataset if args.dataset != cfg.dataset.name else None,
                  "tasks": tasks, "seekers": methods, "repeat": args.repeat,
                  "k_coarse": args.k_coarse, "k_vote": args.k_vote,
                  "vote_factor": args.vote_factor,
                  "k": list(ks), "limit": args.limit, "backend": backend_slug(rt)}
        with open_run("bench", lake=cfg.dataset.name, fields=fields, values=values,
                      config=rt, tag=args.tag, out=args.out) as run:
            out_dir = runner.run_matrix(cfg, args.dataset, tasks=tasks, methods=methods,
                                        repeat=args.repeat, ks=ks, k_coarse=args.k_coarse,
                                        k_vote=args.k_vote, out_dir=run.path,
                                        limit=args.limit, vote_factor=args.vote_factor,
                                        git_sha=git_sha(),
                                        created=run.stamp)
            agg_path = Path(out_dir) / "aggregate.json"
            rc = 0
            if agg_path.is_file():
                import json
                agg = json.loads(agg_path.read_text())
                if not agg["legs"]:
                    print("bench: no legs ran (all requested (task,method) pairs "
                          "unavailable or failed)", file=sys.stderr)
                    rc = 1
                elif agg.get("failed"):
                    print(f"bench: {len(agg['failed'])} leg(s) failed "
                          f"(see aggregate.json['failed'])", file=sys.stderr)
                    rc = 1
        return rc
    if args.cmd == "prepare":
        from src.Benchmark import prepare as P
        P.dispatch(args.task, args.dataset, n=args.n, k=args.k,
                   seed=args.seed, force=args.force,
                   require_semantic=args.require_semantic)
        return 0
    if args.cmd == "compare":
        from src.Benchmark.impute_compare import run_compare, ALL_LEGS, add_comparison
        legs = dict(ALL_LEGS)
        if args.legs:
            selected = [s.strip() for s in args.legs.split(",")]
            unknown = [name for name in selected if name not in ALL_LEGS]
            if unknown:
                raise SystemExit(f"unknown legs {unknown}; choose from {list(ALL_LEGS)}")
            legs = {name: ALL_LEGS[name] for name in selected}
        rt = runtime_info(args.dataset)
        fields = (Field("task"), Field("k", prefix="k"),
                  Field("legs", default=list(ALL_LEGS)),
                  Field("k_coarse", prefix="kc"),
                  Field("mc_cost_cap", prefix="mccap"), Field("backend"))
        values = {"task": args.task, "k": list(args.k_grid), "legs": list(legs),
                  "k_coarse": args.k_coarse, "mc_cost_cap": args.mc_cost_cap,
                  "backend": backend_slug(rt)}
        with open_run("compare", lake=args.dataset, fields=fields, values=values,
                      config=rt, tag=args.tag) as run:
            out = add_comparison(run_compare(args.dataset, k_grid=args.k_grid, legs=legs,
                                             k_coarse=args.k_coarse,
                                             resolve_cache_max=args.resolve_cache_max,
                                             warmup=args.warmup,
                                             mc_cost_cap=args.mc_cost_cap))
            path = run.write_json("compare.json", out)
            print(f"\n=== compare {args.task} {args.dataset} (k={args.k_grid}) ===")
            for leg, ks in out["results"].items():
                if leg == "comparison":
                    continue
                for k, cell in ks.items():
                    print(f"  {leg:16s} k={k:>3}: overall={cell['overall_accuracy']:.3f} "
                          f"cov={cell['coverage']:.3f} acc@cov={cell['accuracy_at_covered']:.3f} "
                          f"rt={cell['retrieval_runtime_ms']:.1f}ms "
                          f"(med {cell.get('retrieval_runtime_ms_median', 0.0):.1f} "
                          f"p95 {cell.get('retrieval_runtime_ms_p95', 0.0):.1f}) "
                          f"resolve={cell.get('resolve_runtime_ms', 0.0):.1f}ms "
                          f"n_ret={cell.get('n_retrieved', 0.0):.1f} "
                          f"cold={cell.get('resolve_cold_fetches', 0)}")
            print(f"wrote {path}")
        return 0
    if args.cmd == "correctness-sweep":
        from src.Benchmark.correctness.sweep import run_sweep
        from src.Benchmark.db import open_dataset_db
        from src.Benchmark.datasource import load_union_queries, load_join_queries
        db = open_dataset_db(args.dataset)
        rt = runtime_info(args.dataset, db=db)
        fields = (Field("plan"), Field("k", prefix="k", default=10),
                  Field("widths", prefix="w", default=[10, 25, 50, 100, 1000]),
                  Field("k_coarses", prefix="kc", default=[50, 100, 250, 500, 1000]),
                  Field("efs", prefix="ef", default=[16, 32, 64, 128, 256]),
                  Field("ga_width", prefix="ga", default=200),
                  Field("repeats", prefix="rep", default=1),
                  Field("limit", prefix="lim"), Field("backend"))
        values = {"plan": args.plan, "k": args.k, "widths": args.widths,
                  "k_coarses": args.k_coarses, "efs": args.efs,
                  "ga_width": args.ga_width, "repeats": args.repeats,
                  "limit": args.limit, "backend": backend_slug(rt)}
        try:
            with open_run("correctness-sweep", lake=args.dataset, fields=fields,
                          values=values, config=rt, tag=args.tag) as run:
                if args.plan.startswith("sj"):
                    qs = load_join_queries(args.dataset)
                else:
                    qs = [(t, None) for t in load_union_queries(args.dataset)]
                if args.limit:
                    qs = qs[: args.limit]
                rows, summary = run_sweep(args.dataset, plan=args.plan, combine="intersection",
                                          k=args.k, widths=args.widths, k_coarses=args.k_coarses,
                                          efs=args.efs, ga_width=args.ga_width,
                                          repeats=args.repeats, queries=qs, db=db)
                print(f"wrote {run.write_table(rows, summary)}")
        finally:
            db.close()
        return 0
    if args.cmd == "correctness-theorem1":
        return _run_theorem1(args, parser)
    if args.cmd == "pushdown-bench":
        from src.Benchmark.pushdown.bench import run_pushdown_bench
        from src.Benchmark.db import open_dataset_db
        from src.Benchmark.datasource import load_union_queries
        comps = _parse_comps(args.comps, args.k)
        db = open_dataset_db(args.dataset)
        rt = runtime_info(args.dataset, db=db)
        fields = (Field("k", prefix="k", default=10), Field("modes"),
                  Field("selectivities", prefix="sel",
                        default=[0.01, 0.05, 0.10, 0.25, 0.50, 0.90]),
                  Field("comps", prefix="c"),
                  Field("exact_threshold", prefix="xt", default=0),
                  Field("k_coarse", prefix="kc"), Field("ef", prefix="ef"),
                  Field("decoy_band", prefix="band"),
                  Field("warmup", prefix="warm", default=True),
                  Field("repeats", prefix="rep", default=3),
                  Field("limit", prefix="lim"), Field("seed", prefix="seed", default=0),
                  Field("backend"))
        values = {"k": args.k, "modes": args.modes, "selectivities": args.selectivities,
                  "comps": comps, "exact_threshold": args.exact_threshold,
                  "k_coarse": args.k_coarse, "ef": args.ef,
                  "decoy_band": args.decoy_band, "warmup": args.warmup,
                  "repeats": args.repeats,
                  "limit": args.limit, "seed": args.seed, "backend": backend_slug(rt)}
        try:
            with open_run("pushdown-bench", lake=args.dataset, fields=fields,
                          values=values, config=rt, tag=args.tag) as run:
                qs = load_union_queries(args.dataset)
                if args.limit:
                    qs = qs[: args.limit]
                rows, summary = run_pushdown_bench(
                    args.dataset, k=args.k, selectivities=args.selectivities,
                    comps=comps, modes=args.modes, exact_threshold=args.exact_threshold,
                    k_coarse=args.k_coarse, repeats=args.repeats, queries=qs,
                    db=db, seed=args.seed, ef=args.ef,
                    decoy_band=tuple(args.decoy_band) if args.decoy_band else None,
                    warmup=args.warmup)
                path = run.write_table(rows, summary)
        finally:
            db.close()
        print(f"\n=== pushdown-bench {args.dataset} (k={args.k}, "
              f"exact_threshold={args.exact_threshold}, k_coarse={args.k_coarse}, "
              f"ef={summary['params']['ef_search']}) ===")
        print(f"evaluated={summary['coverage']['evaluated']} / "
              f"{summary['coverage']['queries_total']} (skipped={summary['skipped']})")
        print(f"  {'mode':<11}{'s':>6}{'c':>5}{'recall':>9}{'ms':>10}")
        for cell in summary["by_group"].values():
            rec = "N/A" if cell["mean_recall"] is None else f"{cell['mean_recall']:.4f}"
            print(f"  {cell['mode']:<11}{cell['selectivity']:>6}{cell['c']:>5}"
                  f"{rec:>9}{cell['median_latency_ms']:>10.3f}")
        flag = "" if summary["leakage_ok"] else "  <-- LEAKAGE BUG"
        print(f"max_leakage={summary['max_leakage']}{flag}")
        print(f"wrote {path}")
        return 0
    if args.cmd == "tiebreak-bench":
        from src.Benchmark.tiebreak.bench import run_tiebreak_bench
        rt = runtime_info(args.dataset)
        fields = (Field("op", default="SU"), Field("k", prefix="k", default=10),
                  Field("k_coarse_grid", prefix="kc", default=[100, 1000]),
                  Field("repeats", prefix="rep", default=3),
                  Field("limit", prefix="lim"),
                  Field("sample_queries", prefix="sq"),
                  Field("seed", prefix="seed", default=0), Field("backend"))
        values = {"op": args.op, "k": args.k, "k_coarse_grid": args.k_coarse_grid,
                  "repeats": args.repeats, "limit": args.limit,
                  "sample_queries": args.sample_queries, "seed": args.seed,
                  "backend": backend_slug(rt)}
        if args.sample_queries is not None:
            from src.Optimizer.semantic_sweep import _FakeDB
            with open_run("tiebreak-bench", lake=args.dataset, fields=fields,
                          values=values, config=rt, tag=args.tag) as run:
                rows, summary = run_tiebreak_bench(
                    args.dataset, k=args.k, k_coarse_grid=args.k_coarse_grid,
                    queries=[], db=_FakeDB(), repeats=args.repeats, seed=args.seed,
                    sample_queries=args.sample_queries, max_pairs=args.max_pairs,
                    op=args.op)
                path = run.write_table(rows, summary)
        else:
            from src.Benchmark.db import open_dataset_db
            from src.Benchmark.datasource import load_union_queries
            db = open_dataset_db(args.dataset)
            try:
                with open_run("tiebreak-bench", lake=args.dataset, fields=fields,
                              values=values, config=rt, tag=args.tag) as run:
                    run.set_config(**runtime_info(args.dataset, db=db))
                    qs = load_union_queries(args.dataset)
                    if args.limit:
                        qs = qs[: args.limit]
                    rows, summary = run_tiebreak_bench(
                        args.dataset, k=args.k, k_coarse_grid=args.k_coarse_grid,
                        queries=qs, db=db, repeats=args.repeats, seed=args.seed,
                        max_pairs=args.max_pairs, op=args.op)
                    path = run.write_table(rows, summary)
            finally:
                db.close()
        cov = summary["coverage"]
        print(f"\n=== tiebreak-bench {args.dataset} {summary['op']}\u2229{summary['op']} "
              f"(k={args.k}, kc_grid={args.k_coarse_grid}) ===")
        sampled = ("" if cov["pairs"] == cov["pairs_total"]
                   else f" sampled from {cov['pairs_total']}")
        print(f"configs={cov['configs']} pairs={cov['pairs']}{sampled} "
              f"(skipped={cov['skipped']})")
        print(f"  placeholder ties:       {summary['placeholder_ties']}")
        print(f"  placeholder inversions: {summary['placeholder_inversions']}")
        print(f"  decisions changed:      {summary['decisions_changed']}")
        print(f"  decisions corrected:    {summary['decisions_corrected']}  "
              f"(new matches GT where the n_cols placeholder tied/inverted)")
        print(f"  decisions regressed:    {summary['decisions_regressed']}  "
              f"(placeholder was right, model is wrong)")
        print(f"  new correct / pairs:    {summary['new_correct']}/{cov['pairs']}")
        rg = summary["regret"]
        print("  --- regret (ms lost vs always ordering the cheaper leg first) ---")
        print(f"  {'coin flip':>11}: {summary['coin_flip_regret_ms']:>10.1f} ms  "
              f"({summary['coin_flip_regret_vs_oracle_pct']:.2f}% of oracle)   <- chance")
        for name in ("placeholder", "new"):
            r = rg[name]
            print(f"  {name:>11}: {r['regret_ms']:>10.1f} ms  "
                  f"({r['regret_vs_oracle_pct']:.2f}% of oracle)  "
                  f"|gap| when wrong: median={r['median_gap_when_wrong_ms']:.3f} "
                  f"mean={r['mean_gap_when_wrong_ms']:.3f} ms")
        print(f"  (all pairs: median |gap|={summary['median_gap_ms']:.3f} "
              f"mean={summary['mean_gap_ms']:.3f} ms — errors well below these are "
              f"cheap ones; errors near them are not)")
        print(f"wrote {path}")
        return 0
    if args.cmd == "ann-frontier":
        from src.Benchmark.ann_frontier import run_frontier
        sel = [] if args.variants.strip() == "none" else [
            v.strip() for v in args.variants.split(",") if v.strip()]
        unknown = [v for v in sel if v not in PG_VARIANTS]
        if unknown:
            parser.error(f"unknown variants {unknown}; choose from {list(PG_VARIANTS)}")
        from src.Semantic.config import SemanticConfig
        rt = runtime_info(args.dataset)
        kc = int(args.k_coarse if args.k_coarse is not None else
                 SemanticConfig.load(overrides={"dataset": args.dataset}).faiss_k_coarse)
        fields = (Field("op", default="SU"), Field("k_coarse", prefix="kc"),
                  Field("efs", prefix="ef", default=[64, 128, 250, 500]),
                  Field("queries", prefix="q", default=40),
                  Field("variants", default=list(PG_VARIANTS)),
                  Field("nofaiss"), Field("io"),
                  Field("seed", prefix="seed", default=0), Field("backend"))
        values = {"op": args.op, "k_coarse": kc, "efs": args.efs,
                  "queries": args.queries, "variants": sel,
                  "nofaiss": not args.with_faiss, "io": args.io_stats,
                  "seed": args.seed, "backend": backend_slug(rt)}
        with open_run("ann-frontier", lake=args.dataset, fields=fields, values=values,
                      config=rt, tag=args.tag) as run:
            rows, summary = run_frontier(
                args.dataset, op=args.op, k_coarse=args.k_coarse, efs=args.efs,
                n_queries=args.queries, seed=args.seed, variants=sel,
                with_faiss=args.with_faiss, io_stats=args.io_stats,
                approach=args.approach, index_name=args.index_name)
            path = run.write_table(rows, summary)
        print(f"\n=== ann-frontier {args.dataset}/{args.op} "
              f"(k_coarse={summary['k_coarse']}, {summary['n_query_tables']} tables, "
              f"{summary['n_col_searches']} col-searches, "
              f"{summary['mean_cols_per_table']:.2f} cols/table) ===")
        print(f"  {'cell':<22}{'plan':>9}{'ms/table':>10}{'ms/col':>9}"
              f"{'recall':>9}{'real_ids':>10}")
        for name, c in sorted(summary["cells"].items(),
                              key=lambda kv: (kv[1]["variant"], kv[1]["ef"])):
            flag = "" if c["plan"] == "hnsw" or c["arm"] == "faiss" else "  <-- NOT HNSW"
            print(f"  {name:<22}{c['plan']:>9}{c['median_ms_table']:>10.2f}"
                  f"{c['mean_ms_col']:>9.2f}{c['mean_recall'] * 100:>8.1f}%"
                  f"{c['mean_real_ids']:>10.1f}{flag}")
        io = summary.get("io")
        if io:
            print(f"\n  buffer pool: shared_buffers={io['shared_buffers']}, "
                  f"working set={io['working_set']}, free={io['pool_free']}")
            print(f"  disk reads: pass1={io['pass1_reads']} pass2={io['pass2_reads']}")
            for rel, v in io["per_relation"].items():
                if v["pass1_reads"] or v["pass2_reads"]:
                    print(f"    {rel:<52}{v['pass1_reads']:>9}{v['pass2_reads']:>9}")
            if not io["ok"]:
                print("  <-- SHARED_BUFFERS TOO SMALL: pass-2 reads mean pages are "
                      "evicted and re-read; pgvector timings above understate it")
            elif io.get("inconclusive"):
                print("  note: zero reads in BOTH passes — the working set was "
                      "already resident, so this run did not test the pool size")
        print(f"wrote {path}")
        return 0 if (io is None or io["ok"]) else 1
    if args.cmd == "regime-recall":
        import pandas as pd
        from src.Benchmark.regime import run_regime_recall

        ks = [int(x) for x in args.k.split(",") if x.strip()]
        pairs = pd.concat([pd.read_parquet(p) for p in args.pairs], ignore_index=True)
        registry = pd.read_csv(args.registry)

        task = args.task
        if task == "auto":
            if "task" in pairs.columns and pairs["task"].nunique() == 1:
                task = str(pairs["task"].iloc[0])
            else:
                cc = pairs["candidate_column"].astype(str) if "candidate_column" in pairs.columns else pd.Series([""])
                task = "join" if cc.str.len().gt(0).any() else "union"
        if "task" in pairs.columns:
            pairs = pairs[pairs["task"] == task]

        if args.dataset is not None:
            dataset = args.dataset
        elif "dataset" in pairs.columns and pairs["dataset"].nunique() == 1:
            dataset = str(pairs["dataset"].iloc[0])
        elif args.out is None:
            raise SystemExit("cannot infer dataset (absent/mixed in --pairs); pass --dataset")
        else:
            dataset = None
        fields = (Field("task"), Field("k", prefix="k", default=[1, 5, 10, 25, 50]),
                  Field("strata"), Field("nonnumeric", prefix="nonnum"),
                  Field("pool_size", prefix="pool"))
        values = {"task": task, "k": ks, "strata": args.strata,
                  "nonnumeric": args.nonnumeric, "pool_size": args.pool_size}
        with open_run("regime-recall", lake=dataset, fields=fields, values=values,
                      tag=args.tag, out=args.out) as run:
            out = run_regime_recall(pairs, registry, task=task, strata=args.strata,
                                    nonnumeric=args.nonnumeric, pool_size=args.pool_size,
                                    ks=ks)
            hitcols = [f"hit@{k}" for k in ks]
            for pool, g in out.groupby("pool_size"):
                print(f"\n===== pool_size={pool}  (recall@k by variant x bucket) =====")
                print(g.set_index(["variant", "bucket"])[hitcols + ["n"]].to_string())
            out_path = run.file("regime_recall.csv")
            out.to_csv(out_path, index=False)
            print(f"\nwrote {out_path}  ({len(out)} rows)")
        return 0
    if args.cmd == "condensed-recall":
        import pandas as pd
        from src.Benchmark.regime import condensed_recall

        ks = [int(x) for x in args.k.split(",") if x.strip()]
        pairs = pd.concat([pd.read_parquet(p) for p in args.pairs], ignore_index=True)
        registry = pd.read_csv(args.registry)

        task = args.task
        if task == "auto":
            if "task" in pairs.columns and pairs["task"].nunique() == 1:
                task = str(pairs["task"].iloc[0])
            else:
                raise SystemExit(
                    "pairs span multiple tasks; pass --task join|union "
                    "(condense one task at a time)")
        if "task" in pairs.columns:
            pairs = pairs[pairs["task"] == task]

        dataset = args.dataset or (
            str(pairs["dataset"].iloc[0])
            if "dataset" in pairs.columns and pairs["dataset"].nunique() == 1 else None)
        fields = (Field("task"),
                  Field("k", prefix="k", default=[1, 5, 10, 25, 50, 150]))
        values = {"task": task, "k": ks}
        out = condensed_recall(pairs, registry, task=task, ks=ks)
        print(f"\n===== condensed recall@k (task={task}) — "
              f"raw = pruned-GT recall, cond = overlap-excused =====")
        print(out.to_string(index=False))
        if dataset is None and args.out is None:
            return 0
        with open_run("condensed-recall", lake=dataset, fields=fields, values=values,
                      tag=args.tag, out=args.out) as run:
            out_path = run.file("condensed_recall.csv")
            out.to_csv(out_path, index=False)
            print(f"\nwrote {out_path}  ({len(out)} rows)")
        return 0
    if args.cmd == "plan":
        from src.Benchmark.plans import PLANS
        if args.plan_cmd == "list":
            for s in PLANS.values():
                print(f"  {s.name}")
                print(f"    {s.description}")
                print(f"    point: {s.point}")
            return 0
        if args.plan_cmd == "run":
            return _run_plan(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
