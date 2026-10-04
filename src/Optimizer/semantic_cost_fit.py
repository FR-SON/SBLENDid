"""Fit the per-dataset analytic semantic cost model from the sweep CSVs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.optimizer_paths import (semantic_cost_model_path,
                                 legacy_semantic_cost_model_path)


def fit_exact_curve(exact_csv: Path) -> tuple[float, float]:
    """Return per-column (beta, alpha) for C_exact = beta + alpha*a_cols."""
    d = pd.read_csv(exact_csv)
    per_col = (d["exact_ms"] / d["n_query_cols"].clip(lower=1)).to_numpy(dtype=float)
    a = d["n_gids"].to_numpy(dtype=float)
    if len(np.unique(a)) < 2:
        return float(per_col.mean()), 0.0
    alpha, beta = np.polyfit(a, per_col, 1)
    return float(beta), float(alpha)


def hnsw_per_col_ms(hnsw_csv: Path, ef: int, kc: int) -> float:
    d = pd.read_csv(hnsw_csv)
    cell = d[(d.efSearch == ef) & (d.k_coarse == kc)]
    if cell.empty:
        raise ValueError(f"no hnsw_surface cell at ef={ef} kc={kc} in {hnsw_csv}")
    per_col = cell["total_ms"] / cell["n_cols"].clip(lower=1)
    return float(per_col.median())


_UNFILTERED_FORM = {"SU": "su_bilinear", "SJ": "sj_linear_kc"}


def _design(form: str, n_cols, k_coarse):
    n = np.asarray(n_cols, dtype=float)
    kc = np.asarray(k_coarse, dtype=float)
    one = np.ones_like(n)
    if form == "su_bilinear":
        return np.column_stack([one, n, kc, n * kc]), ("a", "b_ncols", "c_kc", "d")
    if form == "sj_linear_kc":
        return np.column_stack([one, kc]), ("a", "c_kc")
    raise ValueError(f"unknown unfiltered form {form!r}")


def _eval(form, coef, n_cols, k_coarse):
    X, names = _design(form, n_cols, k_coarse)
    return X @ np.array([coef[nm] for nm in names], dtype=float)


def noise_ceiling(d: pd.DataFrame) -> tuple[float | None, int]:
    """Return (ceiling_r2, n_replicated) = 1 - var_within/var_total over replicated points."""
    key = ["efSearch", "k_coarse", "table_id", "col_name"]
    if "repeat" not in d.columns or d["repeat"].nunique() < 2:
        return None, 0
    y = d["total_ms"].to_numpy(dtype=float)
    var_total = float(np.var(y))
    if var_total <= 0:
        return None, 0
    # dropna=False: SU rows' col_name="" round-trips through CSV as NaN.
    g = d.groupby(key, dropna=False)["total_ms"]
    sizes = g.size()
    if (sizes > 1).sum() == 0:
        return None, 0
    within = float(((g.transform("count") > 1) * (d["total_ms"] - g.transform("mean")) ** 2).sum())
    dof = int((sizes[sizes > 1] - 1).sum())
    if dof <= 0:
        return None, 0
    return round(max(0.0, 1.0 - (within / dof) / var_total), 6), int((sizes > 1).sum())


def fit_unfiltered(hnsw_csv: Path, op: str, ef: int, *, diagonal: bool = False,
                   test_frac: float = 0.25, seed: int = 0) -> dict:
    """Fit cost_unfiltered(n_cols, k_coarse) on fixed-ef or diagonal (ef == k_coarse) rows, with a held-out split."""
    d = pd.read_csv(hnsw_csv)
    d = d[d.efSearch == d.k_coarse] if diagonal else d[d.efSearch == ef]
    if d.empty:
        where = "on the diagonal ef == k_coarse" if diagonal else f"at ef={ef}"
        raise ValueError(f"no hnsw_surface rows {where} in {hnsw_csv}")
    if diagonal and d["k_coarse"].nunique() < 2:
        raise ValueError(
            f"diagonal fit needs >=2 distinct k_coarse on the diagonal, got "
            f"{sorted(pd.unique(d['k_coarse']))} in {hnsw_csv}. Sweep matching "
            f"grids, e.g. --ef-grid 30 60 110 210 --kc-grid 30 60 110 210."
        )
    form = _UNFILTERED_FORM[op]
    n_cols = d["n_cols"].to_numpy(dtype=float)
    kc = d["k_coarse"].to_numpy(dtype=float)
    y = d["total_ms"].to_numpy(dtype=float)
    X, names = _design(form, n_cols, kc)

    m = len(y)
    n_test = int(round(m * test_frac))
    if m >= 4 and n_test >= 1 and (m - n_test) >= X.shape[1]:
        perm = np.random.default_rng(seed).permutation(m)
        test_idx, train_idx = perm[:n_test], perm[n_test:]
    else:
        test_idx, train_idx = np.array([], dtype=int), np.arange(m)

    beta, *_ = np.linalg.lstsq(X[train_idx], y[train_idx], rcond=None)
    coef = {nm: float(beta[i]) for i, nm in enumerate(names)}

    r2 = mae = mdape = None
    if len(test_idx):
        pred = _eval(form, coef, n_cols[test_idx], kc[test_idx])
        resid = y[test_idx] - pred
        mae = round(float(np.mean(np.abs(resid))), 6)
        mdape = round(float(np.median(np.abs(resid) / np.clip(y[test_idx], 1e-9, None))), 6)
        ss_tot = float(np.sum((y[test_idx] - y[test_idx].mean()) ** 2))
        if ss_tot > 0:
            r2 = round(1.0 - float(np.sum(resid ** 2)) / ss_tot, 6)
    ceiling, n_replicated = noise_ceiling(d)
    return {
        "form": form,
        "coef": {nm: round(v, 8) for nm, v in coef.items()},
        "k_coarse_grid": sorted(int(x) for x in pd.unique(d["k_coarse"])),
        "ef_search": int(ef),
        "depth_rule": "diagonal" if diagonal else "fixed_ef",
        "r2": r2,
        "mae": mae,
        "mdape": mdape,
        "ceiling_r2": ceiling,
        "r2_of_ceiling": (round(r2 / ceiling, 6)
                          if (r2 is not None and ceiling not in (None, 0)) else None),
        "n_replicated_points": n_replicated,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }


def dominant_plan_label(hnsw_csv: Path, ef: int, *, diagonal: bool = False) -> str | None:
    """Return the most common SQL plan label over the fitted rows, or None if unrecorded."""
    d = pd.read_csv(hnsw_csv)
    if "plan_label" not in d.columns:
        return None
    rows = d[d.efSearch == d.k_coarse] if diagonal else d[d.efSearch == ef]
    cell = rows["plan_label"].dropna()
    if cell.empty:
        return None
    return str(cell.mode().iloc[0])


def derive_flip(hnsw_per_col_ms: float, beta: float, alpha: float, n_vectors: int) -> int | None:
    """Return a_cols where the exact line meets the HNSW per-col cost, or None if it never flips."""
    if alpha <= 0:
        return None
    a_flip = (hnsw_per_col_ms - beta) / alpha
    if a_flip <= 0 or a_flip >= n_vectors:
        return None
    return int(round(a_flip))


def _exact_source(sem_dir: Path, op: str, backend: str = "faiss") -> Path | None:
    name = f"filtered_curve_{op}.csv" if backend == "pgvector" else f"exact_curve_{op}.csv"
    p = sem_dir / name
    return p if p.is_file() else None


def build_model(sem_dir: Path, *, ops, ef: int, kc: int, lake_meta: dict, backend: str,
                diagonal: bool = False) -> dict:
    sem_dir = Path(sem_dir)
    model: dict = {}
    for op in ops:
        hnsw_csv = sem_dir / f"hnsw_surface_{op}.csv"
        if not hnsw_csv.is_file():
            continue
        per_col = hnsw_per_col_ms(hnsw_csv, ef, kc)
        n_vectors = int(lake_meta[op]["n_vectors"])
        src = _exact_source(sem_dir, op, backend)
        if src is not None:
            beta, alpha = fit_exact_curve(src)
        else:
            beta = alpha = 0.0
        flip = derive_flip(per_col, beta, alpha, n_vectors) if backend == "pgvector" else None
        model[op] = {
            "backend": backend,
            "k_coarse": int(kc),
            "ef_search": int(ef),
            "hnsw_per_col_ms": round(per_col, 6),
            "exact_beta_ms": round(beta, 6),
            "exact_alpha_ms_per_col": round(alpha, 8),
            "flip_a_cols": flip,
            "n_vectors": n_vectors,
            "avg_cols_per_table": lake_meta[op]["avg_cols_per_table"],
            "unfiltered": fit_unfiltered(hnsw_csv, op, ef, diagonal=diagonal),
            "plan_label": dominant_plan_label(hnsw_csv, ef, diagonal=diagonal),
        }
    return model


def format_fit_summary(model: dict) -> str:
    """Held-out validation per op for the `fit-semantic-cost` stdout."""
    out = ["=== unfiltered cost fit (held-out validation) ==="]
    for op, m in model.items():
        u = m.get("unfiltered", {})
        r2 = "n/a" if u.get("r2") is None else f"{u['r2']:.4f}"
        mae = "n/a" if u.get("mae") is None else f"{u['mae']:.4f}"
        rule = u.get("depth_rule", "fixed_ef")
        beam = "ef=kc" if rule == "diagonal" else f"ef={u.get('ef_search')}"
        mdape = "n/a" if u.get("mdape") is None else f"{u['mdape'] * 100:.1f}%"
        if u.get("ceiling_r2") is None:
            ceil = "  ceiling=n/a (sweep has no repeats; pass --repeats 3)"
        else:
            frac = u.get("r2_of_ceiling")
            ceil = (f"  ceiling={u['ceiling_r2']:.4f}"
                    + (f" ({frac * 100:.0f}% of it)" if frac is not None else "")
                    + f" over {u.get('n_replicated_points', 0)} replicated pts")
        out.append(
            f"  {op}: form={u.get('form', '?')}  R2={r2}{ceil}  "
            f"MAE={mae} ms  MdAPE={mdape}  "
            f"(train={u.get('n_train', 0)}/test={u.get('n_test', 0)}, "
            f"rule={rule} [{beam}], kc_grid={u.get('k_coarse_grid', [])})  "
            f"plan={m.get('plan_label')}"
        )
    return "\n".join(out)


def write_model(model: dict, dataset_dir: Path, profile=None) -> Path:
    dest = (semantic_cost_model_path(dataset_dir, profile) if profile
            else legacy_semantic_cost_model_path(dataset_dir))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(model, indent=2))
    return dest
