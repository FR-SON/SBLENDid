"""Analytic ms predictor for semantic seekers, backed by semantic_cost_model.json."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

from src.optimizer_paths import (semantic_cost_model_path,
                                 legacy_semantic_cost_model_path)

_CACHE: dict[str, dict] = {}
_WARNED_LEGACY: set[str] = set()
_WARNED_PROVENANCE: set[tuple] = set()


def _reset_cache() -> None:
    _CACHE.clear()
    _WARNED_LEGACY.clear()
    _WARNED_PROVENANCE.clear()


def _warn_legacy(legacy: Path, expected: Path) -> None:
    if str(legacy) not in _WARNED_LEGACY:
        _WARNED_LEGACY.add(str(legacy))
        warnings.warn(
            f"reading legacy un-profiled semantic cost model {legacy}; expected "
            f"profile-keyed {expected}. Re-run `fit-semantic-cost` to migrate.",
            stacklevel=2,
        )


def warn_on_provenance_mismatch(model: dict, op: str, *, backend: str,
                                ef_search: int) -> None:
    """Warn when a model is priced at a different backend/ef_search than it was fitted at."""
    m = (model or {}).get(op) or {}
    fitted_backend, fitted_ef = m.get("backend"), m.get("ef_search")
    if fitted_backend is None and fitted_ef is None:
        return
    if fitted_backend == backend and fitted_ef == ef_search:
        return
    key = (op, fitted_backend, fitted_ef, backend, ef_search)
    if key in _WARNED_PROVENANCE:
        return
    _WARNED_PROVENANCE.add(key)
    warnings.warn(
        f"semantic cost model for {op} was fitted at backend={fitted_backend!r} "
        f"ef_search={fitted_ef!r} but is being priced at backend={backend!r} "
        f"ef_search={ef_search!r}; re-run `sweep-semantic` + `fit-semantic-cost` "
        f"or the predictions are for the wrong engine.",
        stacklevel=2,
    )


def _maybe_warn(model: dict, op: str, cfg) -> None:
    if cfg is None:
        return
    backend = getattr(cfg, "vector_backend", None)
    ef = getattr(cfg, "faiss_hnsw_ef_search", None)
    if backend is None or ef is None:
        return
    warn_on_provenance_mismatch(model, op, backend=backend, ef_search=ef)


def _model_path(dataset_dir: Path, profile: str | None = None) -> Path:
    dataset_dir = Path(dataset_dir)
    if profile:
        p = semantic_cost_model_path(dataset_dir, profile)
        if p.is_file():
            return p
        legacy = legacy_semantic_cost_model_path(dataset_dir)
        if legacy.is_file():
            _warn_legacy(legacy, p)
            return legacy
        return p
    return legacy_semantic_cost_model_path(dataset_dir)


def load_model(dataset_dir: Path, profile: str | None = None) -> dict:
    path = _model_path(dataset_dir, profile)
    key = str(path)
    if key not in _CACHE:
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} missing; run `uv run python -m src.Optimizer.cli "
                f"fit-semantic-cost --lake <name>` (or set semantic_ml_cost=off)."
            )
        _CACHE[key] = json.loads(path.read_text())
    return _CACHE[key]


def cost_unfiltered(op: str, n_cols: int, k_coarse: int, *, dataset_dir: Path,
                    profile: str | None = None, cfg=None) -> float:
    """Predicted unfiltered (run-first) ms, k_coarse-aware, for the same-op tiebreak."""
    model = load_model(dataset_dir, profile)
    _maybe_warn(model, op, cfg)
    m = model[op]
    u = m.get("unfiltered")
    if u is None:
        return float(n_cols) * float(m["hnsw_per_col_ms"])
    coef, form = u["coef"], u["form"]
    if form == "su_bilinear":
        return (coef["a"] + coef["b_ncols"] * n_cols + coef["c_kc"] * k_coarse
                + coef["d"] * n_cols * k_coarse)
    if form == "sj_linear_kc":
        return coef["a"] + coef["c_kc"] * k_coarse
    raise ValueError(f"unknown unfiltered form {form!r}")


def predict_filtered_ms(op: str, n_cols: int, a_cols: int, *, dataset_dir: Path,
                        profile: str | None = None,
                        exact_threshold: int | None = None, cfg=None) -> float:
    model = load_model(dataset_dir, profile)
    _maybe_warn(model, op, cfg)
    m = model[op]
    flip = exact_threshold if exact_threshold is not None else m.get("flip_a_cols")
    if flip is not None and a_cols > flip:
        per_col = float(m["hnsw_per_col_ms"])
    else:
        per_col = float(m["exact_beta_ms"]) + float(m["exact_alpha_ms_per_col"]) * float(a_cols)
    return float(n_cols) * per_col


def predicted_filtered_ms_for(seeker, a_cols: int) -> float:
    flip = seeker._exact_threshold if seeker._cfg.vector_backend == "faiss" else None
    db = getattr(seeker, "DB", None)
    profile = db.optimizer_profile_str() if db is not None else None
    return predict_filtered_ms(
        seeker.SEMANTIC_OP, seeker.input.shape[1], a_cols,
        dataset_dir=seeker._cfg.dataset.dir(),
        profile=profile,
        exact_threshold=flip,
        cfg=seeker._cfg,
    )
