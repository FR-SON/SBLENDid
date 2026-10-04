"""Export the optimizer artifacts as thesis CSV tables."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from src.Optimizer.report import censored_median, censored_quantile
from src.Optimizer.semantic_cost import assign_cost_integers

DUCKDB_PROFILE = "duckdb-faiss-two_table"
PGVECTOR_PROFILE = "postgres-pgvector-single"


def _costs(datasets_root: Path, lake: str, profile: str) -> dict | None:
    p = Path(datasets_root) / lake / "optimizer" / profile / "costs.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def _offered(runs_root: Path, lake: str, seeker: str):
    p = Path(runs_root) / lake / f"samples_{seeker}.jsonl"
    if not p.is_file():
        return None
    with p.open() as fh:
        return sum(1 for line in fh if line.strip())


def _measured_specs(runs_root: Path, lake: str, profile: str, seeker: str):
    p = (Path(runs_root) / lake / profile / "measure" /
         f"single_seeker_runtimes_{seeker}.csv")
    if not p.is_file():
        return None
    d = pd.read_csv(p)
    if not {"csv", "cols"} <= set(d.columns):
        return None
    return set(zip(d["csv"].astype(str), d["cols"].astype(str)))


def _complete(rep: dict, seeker: str) -> bool:
    offered = (rep.get("sampled") or {}).get(seeker)
    if offered is None:
        return True
    counts = (rep.get("counts") or {}).get(seeker) or 0
    cens = (rep.get("censored") or {}).get(seeker)
    err = (rep.get("errored") or {}).get(seeker)
    accounted = counts + (cens or 0) + (err or 0)
    return accounted >= offered


def cost_ordering_rows(datasets_root: Path, lakes, profiles) -> list[dict]:
    """Return one cost-scale row per (lake, profile, seeker); rank 1 is the cheapest."""
    rows = []
    for lake in lakes:
        for profile in profiles:
            doc = _costs(datasets_root, lake, profile)
            if doc is None:
                continue
            rep, meta = doc["report"], doc["metadata"]
            cens, err = rep.get("censored") or {}, rep.get("errored") or {}
            for rank, seeker in enumerate(rep["order"], start=1):
                rows.append({
                    "lake": lake,
                    "profile": profile,
                    "backend": meta.get("backend"),
                    "seeker": seeker,
                    "rank": rank,
                    "cost_int": rep["cost_int"][seeker],
                    "median_s": rep["median_s"][seeker],
                    "median_ms": rep["median_s"][seeker] * 1000.0,
                    "n_measured": (rep.get("counts") or {}).get(seeker),
                    "n_offered": (rep.get("sampled") or {}).get(seeker),
                    "n_censored": cens.get(seeker),
                    "n_errored": err.get(seeker),
                    "complete": _complete(rep, seeker),
                    "basis": (rep.get("basis") or {}).get(seeker),
                    "ef_search": meta.get("ef_default"),
                    "k_coarse": meta.get("kc_default"),
                })
    return rows


SEEKER_FILE = {"KW": "Keyword"}


def cross_backend_rows(datasets_root: Path, lakes, runs_root: Path | None = None,
                       a: str = DUCKDB_PROFILE, b: str = PGVECTOR_PROFILE) -> list[dict]:
    """Return b/a median ratios per seeker measured on both profiles, with spec-set relation."""
    rows = []
    for lake in lakes:
        da, db = _costs(datasets_root, lake, a), _costs(datasets_root, lake, b)
        if da is None or db is None:
            continue
        ra, rb = da["report"], db["report"]
        ma, mb = ra["median_s"], rb["median_s"]
        for seeker in ra["order"]:
            if seeker not in mb or not ma[seeker]:
                continue
            ca, cb = _complete(ra, seeker), _complete(rb, seeker)
            same, shared, nested, na, nb = None, None, None, None, None
            comparison = "unknown"
            if runs_root is not None:
                stem = SEEKER_FILE.get(seeker, seeker)
                sa = _measured_specs(runs_root, lake, a, stem)
                sb = _measured_specs(runs_root, lake, b, stem)
                if sa is not None and sb is not None:
                    na, nb, shared = len(sa), len(sb), len(sa & sb)
                    same = sa == sb
                    nested = sa <= sb or sb <= sa
                    comparison = ("paired" if same else
                                  "nested" if nested else "disjoint")
            rows.append({
                "lake": lake,
                "seeker": seeker,
                "profile_a": a,
                "profile_b": b,
                "median_a_ms": ma[seeker] * 1000.0,
                "median_b_ms": mb[seeker] * 1000.0,
                "ratio_b_over_a": mb[seeker] / ma[seeker],
                "complete_a": ca,
                "complete_b": cb,
                "same_specs": same,
                "specs_nested": nested,
                "n_specs_a": na,
                "n_specs_b": nb,
                "n_shared_specs": shared,
                "comparison": comparison,
                "comparable": comparison != "disjoint",
            })
    return rows


def censoring_rows(runs_root: Path, lakes, profiles) -> list[dict]:
    """Return censoring-corrected vs survivor-only medians per (lake, profile, seeker)."""
    rows = []
    for lake in lakes:
        for profile in profiles:
            mdir = Path(runs_root) / lake / profile / "measure"
            for f in sorted(mdir.glob("single_seeker_runtimes_*.csv")):
                seeker = f.stem.replace("single_seeker_runtimes_", "")
                d = pd.read_csv(f)
                legacy = "outcome" not in d.columns
                oc = pd.Series("ok", index=d.index) if legacy else d["outcome"].fillna("ok")
                finite = d.loc[oc == "ok", "runtime_s"].dropna()
                n_cens = int((oc == "censored").sum())
                offered = _offered(runs_root, lake, seeker)
                complete = offered is None or len(d) >= offered
                corrected = censored_median(finite.tolist(), n_cens)
                survivor = float(finite.median()) if len(finite) else None
                rows.append({
                    "lake": lake,
                    "profile": profile,
                    "seeker": seeker,
                    "schema": "legacy" if legacy else "outcome",
                    "n_offered": offered,
                    "complete": complete,
                    "n_total": len(d),
                    "n_finite": len(finite),
                    "n_censored": n_cens,
                    "n_errored": int((oc == "error").sum()),
                    "censored_frac": (n_cens / len(d)) if len(d) else None,
                    "median_survivor_s": survivor,
                    "median_corrected_s": corrected,
                    "correction_ratio": (corrected / survivor)
                                        if (corrected and survivor) else None,
                })
    return rows


QUANTILES = (0.50, 0.75, 0.90, 0.95, 0.99)


def dispersion_rows(runs_root: Path, lakes, profiles) -> list[dict]:
    """Return per-query runtime quantiles per seeker type, blank where censored or unsupported."""
    rows = []
    for lake in lakes:
        for profile in profiles:
            mdir = Path(runs_root) / lake / profile / "measure"
            for f in sorted(mdir.glob("single_seeker_runtimes_*.csv")):
                seeker = f.stem.replace("single_seeker_runtimes_", "")
                d = pd.read_csv(f)
                legacy = "outcome" not in d.columns
                oc = pd.Series("ok", index=d.index) if legacy else d["outcome"].fillna("ok")
                finite = d.loc[oc == "ok", "runtime_s"].dropna().tolist()
                n_cens = int((oc == "censored").sum())
                offered = _offered(runs_root, lake, seeker)
                complete = offered is None or len(d) >= offered
                n = len(finite) + n_cens
                row = {
                    "lake": lake, "profile": profile, "seeker": seeker,
                    "schema": "legacy" if legacy else "outcome",
                    "n_offered": offered, "complete": complete,
                    "n_total": len(d), "n_finite": len(finite), "n_censored": n_cens,
                    "censored_frac": (n_cens / len(d)) if len(d) else None,
                }
                qs = {}
                for q in QUANTILES:
                    supported = n * (1.0 - q) >= 1.0
                    v = censored_quantile(finite, n_cens, q) if supported else None
                    qs[q] = v
                    row[f"p{int(q * 100)}_ms"] = None if v is None else v * 1000.0
                row["max_finite_ms"] = max(finite) * 1000.0 if finite else None
                row["max_identifiable_q"] = (max((q for q in QUANTILES if qs[q] is not None),
                                                 default=None))
                row["spread_p95_over_p50"] = ((qs[0.95] / qs[0.50])
                                              if (qs[0.95] and qs[0.50]) else None)
                rows.append(row)
    return rows


def depth_sweep_rows(datasets_root: Path, runs_root: Path, lakes, profiles) -> list[dict]:
    """Return the cost scale re-read at every swept (ef, k_coarse) cell."""
    rows = []
    for lake in lakes:
        for profile in profiles:
            sem = Path(runs_root) / lake / profile / "semantic"
            per_op = {}
            for op in ("SU", "SJ"):
                f = sem / f"hnsw_surface_{op}.csv"
                if f.is_file():
                    d = pd.read_csv(f)
                    per_op[op] = d.groupby(["efSearch", "k_coarse"])["total_ms"].median()
            if not per_op:
                continue
            doc = _costs(datasets_root, lake, profile)
            token, backend = {}, ("faiss" if "faiss" in profile else "pgvector")
            if doc is not None:
                rep = doc["report"]
                backend = doc["metadata"].get("backend") or backend
                token = {t: v for t, v in rep["median_s"].items() if t not in ("SU", "SJ")}
            cells = sorted({c for s in per_op.values() for c in s.index})
            for ef, kc in cells:
                med = dict(token)
                for op, series in per_op.items():
                    if (ef, kc) in series.index:
                        med[op] = float(series.loc[(ef, kc)]) / 1000.0
                ints = assign_cost_integers(med) if token else {}
                order = sorted(med, key=med.get)
                for rank, seeker in enumerate(order, start=1):
                    rows.append({
                        "lake": lake,
                        "profile": profile,
                        "backend": backend,
                        "ef_search": int(ef),
                        "k_coarse": int(kc),
                        "reachable": (backend != "faiss") or (int(ef) == int(kc)),
                        "seeker": seeker,
                        "depth_varying": seeker in ("SU", "SJ"),
                        "median_ms": med[seeker] * 1000.0,
                        "cost_int": ints.get(seeker),
                        "rank": rank if token else None,
                    })
    return rows


def semantic_dispersion_rows(runs_root: Path, lakes, profiles) -> list[dict]:
    """Return per-query SU/SJ cost quantiles per swept (ef, k_coarse) cell."""
    rows = []
    for lake in lakes:
        for profile in profiles:
            sem = Path(runs_root) / lake / profile / "semantic"
            backend = "faiss" if "faiss" in profile else "pgvector"
            for op in ("SU", "SJ"):
                f = sem / f"hnsw_surface_{op}.csv"
                if not f.is_file():
                    continue
                d = pd.read_csv(f)
                if not {"table_id", "total_ms"}.issubset(d.columns):
                    continue
                d["col_name"] = (d["col_name"].fillna("") if "col_name" in d.columns
                                 else "")
                key = ["efSearch", "k_coarse", "table_id", "col_name"]
                g = d.groupby(key)["total_ms"]
                per_query = g.median().reset_index()
                spread_within = g.std(ddof=1) / g.mean() * 100.0
                counts = g.size()
                for (ef, kc), cell in per_query.groupby(["efSearch", "k_coarse"]):
                    v = cell["total_ms"]
                    n = len(v)
                    row = {
                        "lake": lake,
                        "profile": profile,
                        "backend": backend,
                        "op": op,
                        "ef_search": int(ef),
                        "k_coarse": int(kc),
                        "reachable": (backend != "faiss") or (int(ef) == int(kc)),
                        "n_queries": n,
                        "n_repeats": int(counts.loc[(ef, kc)].max()),
                    }
                    qs = {}
                    for q in QUANTILES:
                        qs[q] = float(v.quantile(q)) if n * (1.0 - q) >= 1.0 else None
                        row[f"p{int(q * 100)}_ms"] = qs[q]
                    row["max_ms"] = float(v.max())
                    row["spread_p95_over_p50"] = ((qs[0.95] / qs[0.50])
                                                  if (qs[0.95] and qs[0.50]) else None)
                    cvs = spread_within.loc[(ef, kc)].dropna()
                    row["median_within_query_cv_pct"] = (float(cvs.median())
                                                         if len(cvs) else None)
                    rows.append(row)
    return rows


def write_csv(rows: list[dict], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return path
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def export(out_dir, *, datasets_root, runs_root, lakes, profiles) -> dict[str, Path]:
    out_dir = Path(out_dir)
    written = {
        "cost_ordering.csv": write_csv(
            cost_ordering_rows(datasets_root, lakes, profiles),
            out_dir / "cost_ordering.csv"),
        "cross_backend_ratio.csv": write_csv(
            cross_backend_rows(datasets_root, lakes, runs_root=runs_root),
            out_dir / "cross_backend_ratio.csv"),
        "censoring.csv": write_csv(
            censoring_rows(runs_root, lakes, profiles),
            out_dir / "censoring.csv"),
        "depth_sweep.csv": write_csv(
            depth_sweep_rows(datasets_root, runs_root, lakes, profiles),
            out_dir / "depth_sweep.csv"),
        "dispersion.csv": write_csv(
            dispersion_rows(runs_root, lakes, profiles),
            out_dir / "dispersion.csv"),
        "semantic_dispersion.csv": write_csv(
            semantic_dispersion_rows(runs_root, lakes, profiles),
            out_dir / "semantic_dispersion.csv"),
    }
    return written
