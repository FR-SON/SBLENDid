"""Cost-ordering report: measured cost() ranking, exact-vs-HNSW crossover, HNSW sensitivity."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.Optimizer import db
from src.optimizer_paths import costs_path, legacy_costs_path
from src.Optimizer.semantic_cost import assign_cost_integers, crossover, optimal_play_mean

_KEY_NORMALIZE = {"Keyword": "KW"}


def _canon(key: str) -> str:
    return _KEY_NORMALIZE.get(key, key)


def _serialize_report(rep: dict) -> dict:
    ser: dict = {}
    for k, v in rep.items():
        if isinstance(v, list):
            ser[k] = [_canon(x) for x in v]
        elif isinstance(v, dict):
            ser[k] = {
                _canon(op): (val.to_dict(orient="index")
                             if isinstance(val, pd.DataFrame) else val)
                for op, val in v.items()
            }
        else:
            ser[k] = v
    return ser


def write_profile(rep: dict, lake: str, source_run_dir, *, config_path=db.CONFIG_PATH,
                  profile=None) -> Path:
    ser = _serialize_report(rep)
    profile_doc = {
        "costs": ser["cost_int"],
        "report": ser,
        "metadata": {
            "dataset": lake,
            "profile": profile,
            "source_run_dir": str(source_run_dir),
            "backend": rep.get("backend"),
            "ef_default": rep.get("ef_default"),
            "kc_default": rep.get("kc_default"),
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }
    dataset_dir = db.lake_dir(lake, config_path=config_path)
    dest = costs_path(dataset_dir, profile) if profile else legacy_costs_path(dataset_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(profile_doc, indent=2))
    return dest


def censored_median(finite, n_censored: int):
    """Return the median over finite + right-censored observations, or None if censored."""
    xs = sorted(float(x) for x in finite)
    n = len(xs) + max(0, int(n_censored))
    if n == 0:
        return None
    lo, hi = (n - 1) // 2, n // 2
    if hi >= len(xs):
        return None
    return (xs[lo] + xs[hi]) / 2.0


def censored_quantile(finite, n_censored: int, q: float):
    """Return the q-quantile over finite + right-censored observations, or None if censored."""
    import math
    xs = sorted(float(x) for x in finite)
    n = len(xs) + max(0, int(n_censored))
    if n == 0 or not xs:
        return None
    pos = float(q) * (n - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    if hi >= len(xs):
        return None
    return xs[lo] + (pos - lo) * (xs[hi] - xs[lo])


_DEFAULT_EF = 64
_DEFAULT_KC = 500


def build_cost_report(lake_run_dir, ef_default=_DEFAULT_EF, kc_default=_DEFAULT_KC,
                      *, backend="faiss", measure_dir=None, semantic_dir=None,
                      samples_dir=None):
    """Assemble the cost-ordering report dict from the measure and semantic sweep CSVs."""
    base = Path(lake_run_dir)
    measure_dir = Path(measure_dir) if measure_dir else base / "measure"
    semantic_dir = Path(semantic_dir) if semantic_dir else base / "semantic"
    samples_dir = Path(samples_dir) if samples_dir else base
    skipped_ops: dict[str, str] = {}
    med: dict[str, float] = {}
    counts: dict[str, int] = {}
    sampled: dict[str, int] = {}
    censored: dict[str, int | None] = {}
    errored: dict[str, int | None] = {}
    basis: dict[str, str] = {}

    for f in sorted(measure_dir.glob("single_seeker_runtimes_*.csv")):
        d = pd.read_csv(f)
        legacy = "outcome" not in d.columns
        oc = pd.Series("ok", index=d.index) if legacy else d["outcome"].fillna("ok")
        for t, g in d.groupby("seeker_type"):
            og = oc.loc[g.index]
            finite = g.loc[og == "ok", "runtime_s"].dropna().tolist()
            n_cens = int((og == "censored").sum())
            censored[t] = None if legacy else n_cens
            errored[t] = None if legacy else int((og == "error").sum())
            m = censored_median(finite, n_cens)
            if m is None:
                skipped_ops[t] = (
                    f"{n_cens} of {len(finite) + n_cens} sampled queries exceeded the "
                    f"wall cap, so the median is censored and not identifiable. Raise "
                    f"--statement-timeout, or narrow the sampled population."
                )
                continue
            med[t] = m
            counts[t] = len(finite)
            basis[t] = "unfiltered SQL (full index)"
            spec_file = samples_dir / f"samples_{t}.jsonl"
            if spec_file.is_file():
                with spec_file.open() as fh:
                    sampled[t] = sum(1 for line in fh if line.strip())

    semantic: dict[str, dict] = {}
    sensitivity: dict[str, pd.DataFrame] = {}
    crossover_surface: dict[str, pd.DataFrame] = {}
    for op in ("SU", "SJ"):
        hpath = semantic_dir / f"hnsw_surface_{op}.csv"
        if not hpath.is_file():
            skipped_ops[op] = f"no {hpath.name} under {semantic_dir}"
            continue
        h = pd.read_csv(hpath)
        cell = h[(h.efSearch == ef_default) & (h.k_coarse == kc_default)]
        if cell.empty:
            skipped_ops[op] = (
                f"{hpath.name} has no cell at ef={ef_default}/kc={kc_default}; "
                f"it holds ef={sorted(int(x) for x in pd.unique(h.efSearch))} x "
                f"kc={sorted(int(x) for x in pd.unique(h.k_coarse))}. Re-sweep at "
                f"the anchor, or report at a cell the sweep measured."
            )
            continue
        hnsw_s = float(cell["total_ms"].median()) / 1000.0
        med[op] = hnsw_s
        counts[op] = int(len(cell))
        basis[op] = f"unfiltered HNSW @ ef{ef_default}/kc{kc_default}"
        sensitivity[op] = h.pivot_table(
            index="efSearch", columns="k_coarse", values="total_ms", aggfunc="median"
        ).round(3)
        entry = {
            "hnsw_ms": hnsw_s * 1000,
            "faiss_ms": float(cell["faiss_ms"].median()),
            "overhead_ms": float(cell["overhead_ms"].median()),
        }
        cname = f"filtered_curve_{op}.csv" if backend == "pgvector" else f"exact_curve_{op}.csv"
        cpath = semantic_dir / cname
        if cpath.is_file():
            e = pd.read_csv(cpath)
            if backend == "pgvector":
                entry.update(
                    filtered_gids_min=int(e["n_gids"].min()),
                    filtered_gids_max=int(e["n_gids"].max()),
                    filtered_ms_at_min=float(e[e.n_gids == e.n_gids.min()]["exact_ms"].median()),
                    filtered_ms_at_max=float(e[e.n_gids == e.n_gids.max()]["exact_ms"].median()),
                )
            else:
                curve = [(int(g), float(t) / 1000.0)
                         for g, t in e.groupby("n_gids")["exact_ms"].median().items()]
                entry.update(
                    exact_threshold_gids=crossover(curve, hnsw_s),
                    optimal_play_ms=optimal_play_mean(curve, hnsw_s) * 1000,
                    gids_min=int(e["n_gids"].min()),
                    gids_max=int(e["n_gids"].max()),
                    exact_ms_at_min=float(e[e.n_gids == e.n_gids.min()]["exact_ms"].median()),
                    exact_ms_at_max=float(e[e.n_gids == e.n_gids.max()]["exact_ms"].median()),
                )
                hs = (h.groupby(["efSearch", "k_coarse"])["total_ms"].median() / 1000.0).reset_index()
                hs["xover"] = hs["total_ms"].apply(lambda v: crossover(curve, v))
                crossover_surface[op] = hs.pivot(index="efSearch", columns="k_coarse", values="xover")
        semantic[op] = entry

    return {
        "median_s": med,
        "counts": counts,
        "sampled": sampled,
        "censored": censored,
        "errored": errored,
        "skipped_ops": skipped_ops,
        "basis": basis,
        "cost_int": assign_cost_integers(med),
        "order": sorted(med, key=med.get),
        "semantic": semantic,
        "sensitivity": sensitivity,
        "crossover_surface": crossover_surface,
        "ef_default": ef_default,
        "kc_default": kc_default,
        "backend": backend,
    }


def format_report(rep) -> str:
    out = []
    out.append("=== median single-seeker runtime, one scale (ascending) ===")
    out.append(f"  {'seeker':8s} {'median':>11s}   {'cost()':>7s}  {'n':>5s} "
               f"{'skipped':>8s} {'censored':>11s} {'err':>4s}  basis")
    sampled = rep.get("sampled") or {}
    censored = rep.get("censored") or {}
    errored = rep.get("errored") or {}
    for t in rep["order"]:
        n = rep["counts"][t]
        offered = sampled.get(t)
        skip = "-" if offered is None else f"{offered - n} / {offered}"
        c = censored.get(t)
        cens = "n/a" if c is None else (f"{c} ({c / (n + c):.0%})" if c else "0")
        e = errored.get(t)
        out.append(f"  {t:8s} {rep['median_s'][t] * 1000:8.3f} ms   "
                   f"{rep['cost_int'][t]:>7d}  {n:>5d} {skip:>8s} {cens:>11s} "
                   f"{('n/a' if e is None else e):>4}  {rep['basis'][t]}")
    out.append("")
    out.append("  cost() ordering:  "
               + " < ".join(f"{t}={rep['cost_int'][t]}" for t in rep["order"]))
    out.append("")
    for op, why in (rep.get("skipped_ops") or {}).items():
        out.append(f"  !! {op} ABSENT: {why}")
    if rep.get("skipped_ops"):
        out.append("  !! the cost() integers above are normalised by the CHEAPEST "
                   "seeker PRESENT, so a missing semantic seeker rebases all of them.")
        out.append("")
    if rep.get("backend") == "pgvector":
        out.append("=== semantic HNSW vs filtered (planner owns the exact<->ANN flip) ===")
    else:
        out.append("=== semantic HNSW vs exact (exact_threshold unit = |gids|) ===")
    for op, s in rep["semantic"].items():
        breakdown = ("faiss n/a" if pd.isna(s["faiss_ms"])
                     else f"faiss {s['faiss_ms']:.3f} + overhead {s['overhead_ms']:.3f}")
        out.append(f"  {op}: HNSW {s['hnsw_ms']:.3f} ms ({breakdown})")
        if "exact_threshold_gids" in s:
            out.append(f"      exact {s['exact_ms_at_min']:.3f} ms @|gids|={s['gids_min']} "
                       f"-> {s['exact_ms_at_max']:.3f} ms @|gids|={s['gids_max']}")
            out.append(f"      crossover |gids|={s['exact_threshold_gids']} "
                       f"@ef{rep['ef_default']}/kc{rep['kc_default']} (recommended exact_threshold)"
                       f"   optimal-play mean {s['optimal_play_ms']:.3f} ms")
        elif "filtered_ms_at_min" in s:
            out.append(f"      filtered {s['filtered_ms_at_min']:.3f} ms @|gids|={s['filtered_gids_min']} "
                       f"-> {s['filtered_ms_at_max']:.3f} ms @|gids|={s['filtered_gids_max']} "
                       f"(planner-owned flip; pgvector)")
    out.append("")
    out.append(f"=== HNSW sensitivity: median total_ms per efSearch x k_coarse "
               f"(default ef{rep['ef_default']}/kc{rep['kc_default']}) ===")
    for op, piv in rep["sensitivity"].items():
        out.append(f"  {op}:")
        out.append("    " + piv.to_string().replace("\n", "\n    "))
    if rep["crossover_surface"]:
        out.append("")
        out.append("=== recommended exact_threshold (crossover |gids|) "
                   "per efSearch x k_coarse ===")
        out.append("    (same cells as the sensitivity matrix above; exact wins below this |gids|)")
        for op, piv in rep["crossover_surface"].items():
            out.append(f"  {op}:")
            out.append("    " + piv.to_string().replace("\n", "\n    "))
    return "\n".join(out)
