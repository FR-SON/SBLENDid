import argparse
import os
import sys
from pathlib import Path

from src.Optimizer.db import (lake_dir, lake_duckdb_path, lake_csv_dir, lake_rowid_slice_path,
                              lake_dbms, use_lake, open_lake_db, load_freqs,
                              ROWID_SLICE_FILENAME, CONFIG_PATH)
from src.dataset import load_dataset_config
from src.optimizer_paths import (freqs_path, optimizer_lake_run_dir,
                                 optimizer_run_dir, resolve_optimizer_run_dir)
from src.Optimizer.freqs import build_freqs_dict
from src.Optimizer.materialize import (build_rowid_slice, require_duckdb_slice,
                                       warn_rowid_slice)
from src.Optimizer import sampling as smp
from src.Optimizer import export_tables as xt
from src import paths


def _runs_base() -> Path:
    return paths.runs_root() / "optimizer"


def _lake_run_dir(lake: str) -> Path:
    return optimizer_lake_run_dir(paths.runs_root(), lake)


def _profile_of(lake: str) -> str:
    return open_lake_db(lake).optimizer_profile_str()


def _write_run_dir(lake: str, profile: str) -> Path:
    return optimizer_run_dir(paths.runs_root(), lake, profile)


def _read_run_dir(lake: str, profile: str, *, subdir: str) -> Path:
    return resolve_optimizer_run_dir(paths.runs_root(), lake, profile, subdir=subdir)

_CALIBRATE_ORDER = ("build-freqs", "build-slice", "sample", "train", "measure",
                    "sweep-semantic", "fit-semantic-cost", "report")


def parse_n_type(pairs) -> dict:
    """Parse ['C=50', 'MC=25'] into {'C': 50, 'MC': 25}."""
    out = {}
    for p in pairs:
        t, _, n = str(p).partition("=")
        if t not in smp.SEEKER_TYPES or not n.isdigit():
            raise SystemExit(
                f"--n-type: expected TYPE=N with TYPE in {list(smp.SEEKER_TYPES)}, got {p!r}")
        out[t] = int(n)
    return out


def _calibrate_stages(args):
    L, k, seed = args.lake, str(args.k), str(args.seed)
    types = [str(t) for t in args.seeker_types]
    stages = [("build-freqs", ["build-freqs", "--lake", L])]
    if args.rowid_slice:
        stages.append(("build-slice", ["build-slice", "--lake", L]))
    sample_argv = ["sample", "--lake", L, "--seed", seed, "--n", str(args.n),
                   "--max-cat-card", str(args.max_cat_card),
                   "--seeker-types", *types]
    if args.n_type:
        sample_argv += ["--n-type", *args.n_type]
    stages.append(("sample", sample_argv))
    train_argv = ["train", "--lake", L, "--k", k, "--repeats", str(args.repeats),
                  "--seeker-types", *types]
    if args.rowid_slice:
        train_argv.append("--rowid-slice")
    if args.no_warmup:
        train_argv.append("--no-warmup")
    stages.append(("train", train_argv))
    measure_argv = ["measure", "--lake", L, "--k", k, "--repeats", str(args.repeats),
                    "--statement-timeout", str(args.statement_timeout),
                    "--seeker-types", *types]
    if args.no_warmup:
        measure_argv.append("--no-warmup")
    stages.append(("measure", measure_argv))
    stages.append(("sweep-semantic",
                   ["sweep-semantic", "--lake", L, "--seed", seed,
                    "--n-su", str(args.n_su), "--n-sj", str(args.n_sj),
                    "--repeats", str(args.sweep_repeats),
                    "--ops", *args.ops]))
    stages.append(("fit-semantic-cost",
                   ["fit-semantic-cost", "--lake", L, "--ef", str(args.ef),
                    "--kc", str(args.kc), "--ops", *args.ops]))
    stages.append(("report", ["report", "--lake", L, "--write",
                              "--ef", str(args.ef), "--kc", str(args.kc)]))
    return [(n, a) for n, a in stages if n not in args.skip]


def _run_pipeline(stages, runner):
    total = len(stages)
    for i, (name, argv) in enumerate(stages, 1):
        print(f"\n=== calibrate [{i}/{total}] {name} ===", flush=True)
        runner(argv)


def build_parser():
    p = argparse.ArgumentParser(prog="src.Optimizer.cli")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("build-freqs", "build-slice", "sample", "train", "eval"):
        sp = sub.add_parser(name)
        sp.add_argument("--lake", required=True)
        sp.add_argument("--k", type=int, default=10)
        if name in ("sample", "train"):
            sp.add_argument("--seeker-types", nargs="+", choices=smp.SEEKER_TYPES,
                            default=list(smp.SEEKER_TYPES))
        if name in ("sample",):
            sp.add_argument("--n", type=int, default=1000)
            sp.add_argument("--n-type", nargs="+", default=[], metavar="TYPE=N",
                            help="per-seeker-type sample size, overriding --n "
                                 "(e.g. --n-type C=50 MC=25); C/MC dominate "
                                 "measure runtime, so cut them here")
            sp.add_argument("--seed", type=int, default=0)
            sp.add_argument("--max-cat-card", type=int, default=1000,
                            help="C only: skip categorical columns with more distinct "
                                 "values than this (they are id/key columns whose huge "
                                 "IN-list dominates runtime and aren't real categories)")
        if name == "train":
            sp.add_argument("--repeats", type=int, default=3)
            sp.add_argument("--no-warmup", action="store_true")
            sp.add_argument("--rowid-slice", action="store_true",
                            help="measure C against the rowid<256 slice (run build-slice first)")
        if name == "eval":
            sp.add_argument("--seeker-types", nargs="+", choices=smp.SEEKER_TYPES,
                            default=["SC", "Keyword", "C"])
            sp.add_argument("--n", type=int, default=50)
            sp.add_argument("--seed", type=int, default=0)
            sp.add_argument("--append", action="store_true",
                            help="append to the existing eval CSV instead of overwriting")

    mp = sub.add_parser("measure")
    mp.add_argument("--lake", required=True)
    mp.add_argument("--k", type=int, default=10)
    mp.add_argument("--seeker-types", nargs="+", choices=smp.SEEKER_TYPES,
                    default=list(smp.SEEKER_TYPES))
    mp.add_argument("--repeats", type=int, default=3)
    mp.add_argument("--no-warmup", action="store_true")
    mp.add_argument("--limit", type=int, default=None,
                    help="measure at most N specs per seeker type, taking a prefix of "
                         "the stratified draw (which round-robins the cardinality "
                         "strata, so a prefix stays balanced). The draw itself is "
                         "unchanged, so a limited profile and a full one estimate the "
                         "same population at different precision")
    mp.add_argument("--statement-timeout", type=int, default=600,
                    help="per-query wall cap in seconds; over-budget queries are "
                         "skipped rows. Server-side on postgres, client-side "
                         "cancellation on duckdb. Applies per execution, so one "
                         "sampled query costs at most (repeats + 1) x this. 0 = off")

    swp = sub.add_parser("sweep-semantic")
    swp.add_argument("--lake", required=True)
    swp.add_argument("--ops", nargs="+", choices=["SU", "SJ"], default=["SU", "SJ"])
    swp.add_argument("--n-su", type=int, default=200)
    swp.add_argument("--n-sj", type=int, default=1000)
    swp.add_argument("--seed", type=int, default=0)
    swp.add_argument("--ef-grid", nargs="+", type=int, default=[16, 32, 64, 128, 256])
    swp.add_argument("--kc-grid", nargs="+", type=int, default=[100, 500, 1000])
    swp.add_argument("--gids-grid", nargs="+", type=int,
                     default=[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    swp.add_argument("--repeats", type=int, default=1,
                     help="time each (ef, k_coarse, query) point N times, emitting "
                          "one row per replicate. The within-point spread is the "
                          "fit's error floor, so this is what makes "
                          "fit-semantic-cost report a noise ceiling for its R2")
    swp.add_argument("--diagonal", action="store_true",
                     help="measure only the ef == k_coarse cells. Under the derived "
                          "depth rule those are the only reachable points on faiss, "
                          "and they are what `fit-semantic-cost --depth-rule diagonal` "
                          "reads; the cross product costs len(grid)x for the same fit")

    rp = sub.add_parser("report")
    rp.add_argument("--lake", required=True)
    rp.add_argument("--ef", type=int, default=64, help="HNSW efSearch cell for the cost basis")
    rp.add_argument("--kc", type=int, default=500, help="k_coarse cell for the cost basis")
    rp.add_argument("--write", action="store_true",
                    help="persist the cost+threshold profile under datasets/<name>/optimizer/<profile>/costs.json")

    ex = sub.add_parser("export-tables",
                        help="write the thesis tables (cost ordering, cross-backend "
                             "ratio, censoring) as CSV")
    ex.add_argument("--out", required=True, help="output directory for the CSVs")
    ex.add_argument("--lakes", nargs="+", required=True)
    ex.add_argument("--profiles", nargs="+",
                    default=[xt.DUCKDB_PROFILE, xt.PGVECTOR_PROFILE])
    ex.add_argument("--datasets-root", dest="datasets_root", default=None,
                    help="dir holding <lake>/optimizer/<profile>/costs.json "
                         "(default: [Dataset] root)")
    ex.add_argument("--runs-root", dest="runs_root", default=None,
                    help="dir holding <lake>/<profile>/measure/*.csv "
                         "(default: <runs>/optimizer)")

    fsp = sub.add_parser("fit-semantic-cost")
    fsp.add_argument("--lake", required=True)
    fsp.add_argument("--ops", nargs="+", choices=["SU", "SJ"], default=["SU", "SJ"])
    fsp.add_argument("--ef", type=int, default=64, help="HNSW efSearch cell for hnsw_per_col")
    fsp.add_argument("--kc", type=int, default=500, help="k_coarse cell for hnsw_per_col")
    fsp.add_argument("--depth-rule", dest="depth_rule",
                     choices=["auto", "diagonal", "fixed_ef"], default="auto",
                     help="rows the k_coarse-aware fit reads: fixed_ef (one ef, "
                          "k_coarse varies) or diagonal (ef == k_coarse). auto = "
                          "diagonal on faiss with no faiss_k_coarse pin, since the "
                          "derived beam tracks the fetch; fixed_ef otherwise")

    cp = sub.add_parser("calibrate",
                        help="run the full optimizer suite for a lake in dependency order")
    cp.add_argument("--lake", required=True)
    cp.add_argument("--k", type=int, default=10)
    cp.add_argument("--seed", type=int, default=0)
    cp.add_argument("--skip", nargs="*", choices=_CALIBRATE_ORDER, default=[],
                    help="stage names to skip (resume a partial run)")
    cp.add_argument("--rowid-slice", action="store_true",
                    help="also build + train C against the rowid<256 slice")
    cp.add_argument("--no-warmup", action="store_true")
    cp.add_argument("--n", type=int, default=1000)
    cp.add_argument("--n-type", nargs="+", default=[], metavar="TYPE=N",
                    help="per-seeker-type sample size (see sample --n-type)")
    cp.add_argument("--seeker-types", nargs="+", choices=smp.SEEKER_TYPES,
                    default=list(smp.SEEKER_TYPES))
    cp.add_argument("--max-cat-card", type=int, default=1000)
    cp.add_argument("--repeats", type=int, default=3)
    cp.add_argument("--statement-timeout", type=int, default=600)
    cp.add_argument("--n-su", type=int, default=200)
    cp.add_argument("--n-sj", type=int, default=1000)
    cp.add_argument("--sweep-repeats", dest="sweep_repeats", type=int, default=1,
                    help="sweep-semantic --repeats (>1 gives fit-semantic-cost "
                         "a noise ceiling for its R2)")
    cp.add_argument("--ops", nargs="+", choices=["SU", "SJ"], default=["SU", "SJ"])
    cp.add_argument("--ef", type=int, default=64)
    cp.add_argument("--kc", type=int, default=500)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    lake = getattr(args, "lake", None)
    out_dir = str(_lake_run_dir(lake)) if lake else None
    if args.cmd == "build-freqs":
        fp = freqs_path(lake_dir(lake))
        fp.parent.mkdir(parents=True, exist_ok=True)
        print(build_freqs_dict(open_lake_db(lake), out_path=fp))
    elif args.cmd == "build-slice":
        require_duckdb_slice(lake_dbms(), "build-slice")
        warn_rowid_slice("build-slice")
        print(build_rowid_slice(lake_duckdb_path(lake), lake_rowid_slice_path(lake)))
    elif args.cmd == "sample":
        n_by_type = parse_n_type(args.n_type)
        stray = set(n_by_type) - set(args.seeker_types)
        if stray:
            raise SystemExit(
                f"--n-type names {sorted(stray)}, absent from --seeker-types "
                f"{list(args.seeker_types)} — those sizes would be ignored")
        os.makedirs(out_dir, exist_ok=True)
        d = lake_csv_dir(lake)
        drawn = smp.sample_many(d, args.seeker_types, n_by_type=n_by_type,
                                n_default=args.n, seed=args.seed,
                                max_cat_card=args.max_cat_card)
        for t, specs in drawn.items():
            smp.save_specs(specs, f"{out_dir}/samples_{t}.jsonl")
            print(t, len(specs))
    elif args.cmd == "train":
        from src.Optimizer import training as tr
        if args.rowid_slice:
            require_duckdb_slice(lake_dbms(), "train --rowid-slice")
            warn_rowid_slice("train")
        load_freqs(freqs_path(lake_dir(lake)))
        db = use_lake(lake)
        type_db = None
        if args.rowid_slice:
            slice_path = lake_rowid_slice_path(lake)
            if not slice_path.exists():
                raise SystemExit(f"{slice_path} missing; run: build-slice --lake {lake}")
            type_db = {"C": open_lake_db(lake, db_filename=ROWID_SLICE_FILENAME)}
        res = tr.train_all(lake, db, lake_csv_dir(lake), out_dir, lake_dir(lake),
                           db.optimizer_profile_str(), k=args.k,
                           repeats=args.repeats, warmup=not args.no_warmup,
                           seeker_types=tuple(args.seeker_types), type_db=type_db)
        for t, (path, n) in res.items():
            print(t, "->", path, "n=", n)
    elif args.cmd == "eval":
        from src.Optimizer import evaluate as ev
        log_dir = f"{out_dir}/eval"
        d = lake_csv_dir(lake)
        all_specs = []
        for t in args.seeker_types:
            specs = ev.sample_plan_specs(d, seeker_type=t, n=args.n, seed=args.seed, k=args.k)
            all_specs += specs
            print(t, "plans:", len(specs))
        ev.evaluate(lake, all_specs, log_dir, clear=not args.append)
        print(ev.summarize(log_dir).to_string())
    elif args.cmd == "measure":
        from src.Optimizer import training as tr
        db = use_lake(lake)
        meas_dir = str(_write_run_dir(lake, db.optimizer_profile_str()) / "measure")
        os.makedirs(meas_dir, exist_ok=True)
        tr.apply_statement_timeout(db, args.statement_timeout)
        measured = tr.measure_single_seekers(
            db, lake_csv_dir(lake), meas_dir,
            seeker_types=args.seeker_types, k=args.k,
            repeats=args.repeats, warmup=not args.no_warmup, samples_dir=out_dir,
            timeout=args.statement_timeout, limit=args.limit)
        print(f"wrote {len(measured)} per-type runtime CSV(s) to {meas_dir}/")
    elif args.cmd == "sweep-semantic":
        from src.Semantic.config import SemanticConfig
        from src.Optimizer import semantic_sweep as ssw
        cfg = SemanticConfig.load(overrides={"dataset": lake})
        sem_dir = str(_write_run_dir(lake, _profile_of(lake)) / "semantic")
        os.makedirs(sem_dir, exist_ok=True)
        on_faiss = cfg.vector_backend == "faiss"
        for op in args.ops:
            queries = (ssw.sample_su_queries(cfg, args.n_su, args.seed) if op == "SU"
                       else ssw.sample_sj_queries(cfg, args.n_sj, args.seed))
            ssw.run_hnsw_surface(cfg, op, queries, args.ef_grid, args.kc_grid,
                                 f"{sem_dir}/hnsw_surface_{op}.csv",
                                 diagonal=args.diagonal, repeats=args.repeats)
            if on_faiss:
                ssw.run_exact_curve(cfg, op, queries, args.gids_grid,
                                    f"{sem_dir}/exact_curve_{op}.csv", seed=args.seed)
            else:
                ssw.run_filtered_curve(cfg, op, queries, args.gids_grid,
                                       f"{sem_dir}/filtered_curve_{op}.csv", seed=args.seed)
        curve = "exact_curve_*.csv" if on_faiss else "filtered_curve_*.csv"
        print(f"wrote {sem_dir}/hnsw_surface_*.csv + {curve} ({cfg.vector_backend} backend)")
    elif args.cmd == "report":
        from pathlib import Path as _Path
        from src.Semantic.config import SemanticConfig
        from src.Optimizer.report import build_cost_report, format_report, write_profile
        backend = SemanticConfig.load(overrides={"dataset": lake}).vector_backend
        prof = _profile_of(lake)
        meas_dir = _read_run_dir(lake, prof, subdir="measure") / "measure"
        sem_dir = _read_run_dir(lake, prof, subdir="semantic") / "semantic"
        print(f"reading measure={meas_dir}\n        semantic={sem_dir}")
        other = "exact_curve" if backend == "pgvector" else "filtered_curve"
        stale = sorted(_Path(sem_dir).glob(f"{other}_*.csv"))
        if stale:
            print(f"note: backend={backend}; ignoring {len(stale)} stale {other}_*.csv "
                  f"(other-backend curve, left in place)")
        rep = build_cost_report(out_dir, ef_default=args.ef, kc_default=args.kc,
                                backend=backend, measure_dir=meas_dir,
                                semantic_dir=sem_dir, samples_dir=_lake_run_dir(lake))
        print(format_report(rep))
        if args.write:
            active = load_dataset_config(CONFIG_PATH).name
            if lake != active:
                print(
                    f"warning: --lake {lake} differs from config [Dataset] name={active}; "
                    f"resolve_cost (cost_basis=measured) reads the [Dataset] name profile "
                    f"— set [Dataset] name={lake} to use this profile.",
                    file=sys.stderr,
                )
            dest = write_profile(rep, lake, _write_run_dir(lake, prof), profile=prof)
            print(f"wrote {dest}")
    elif args.cmd == "export-tables":
        droot = Path(args.datasets_root) if args.datasets_root else paths.datasets_root()
        rroot = Path(args.runs_root) if args.runs_root else paths.runs_root() / "optimizer"
        written = xt.export(args.out, datasets_root=droot, runs_root=rroot,
                            lakes=args.lakes, profiles=args.profiles)
        for name, path in written.items():
            n = max(0, sum(1 for _ in open(path)) - 1)
            print(f"{name:26s} {n:>5d} rows -> {path}")
    elif args.cmd == "fit-semantic-cost":
        from pathlib import Path as _Path
        from src.Semantic.config import SemanticConfig
        from src.Optimizer import semantic_sweep as ssw
        from src.Optimizer import semantic_cost_fit as fit
        cfg = SemanticConfig.load(overrides={"dataset": lake})
        sem_dir = _read_run_dir(lake, _profile_of(lake), subdir="semantic") / "semantic"
        lake_meta = {}
        for op in args.ops:
            h = ssw._handle(cfg, op, cfg.faiss_hnsw_ef_search)
            n = h.vector_count
            ntab = max(1, len(h.table_to_gids))
            lake_meta[op] = {"n_vectors": int(n),
                             "avg_cols_per_table": round(n / ntab, 4)}
        if args.depth_rule == "auto":
            diagonal = (cfg.vector_backend == "faiss" and cfg.faiss_k_coarse is None)
        else:
            diagonal = args.depth_rule == "diagonal"
        print(f"depth rule: {'diagonal (ef == k_coarse)' if diagonal else f'fixed ef={args.ef}'}"
              f"  [--depth-rule {args.depth_rule}]")
        model = fit.build_model(sem_dir, ops=args.ops, ef=args.ef, kc=args.kc,
                                lake_meta=lake_meta, backend=cfg.vector_backend,
                                diagonal=diagonal)
        prof = open_lake_db(lake).optimizer_profile_str()
        dest = fit.write_model(model, cfg.dataset.dir(), prof)
        print(fit.format_fit_summary(model))
        print(f"wrote {dest}")
    elif args.cmd == "calibrate":
        from src.Semantic.config import SemanticConfig
        backend = SemanticConfig.load(overrides={"dataset": lake}).vector_backend
        if args.rowid_slice:
            require_duckdb_slice(lake_dbms(), "calibrate --rowid-slice")
        stages = _calibrate_stages(args)
        print(f"calibrate plan [{backend}]: " + " -> ".join(n for n, _ in stages))
        _run_pipeline(stages, main)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
