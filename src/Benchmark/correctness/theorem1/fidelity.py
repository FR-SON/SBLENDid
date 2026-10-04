"""Config gates, knob resolution and the forced-exact patch."""
from __future__ import annotations

import contextlib


class FidelityError(RuntimeError):
    pass


def check_startup(db, *, allow_measured_costs: bool = False) -> dict:
    import src.cost_model as cost_model
    guard = cost_model.pushdown_guard()
    basis = cost_model.cost_basis()
    ml = bool(type(db).USE_ML_OPTIMIZER)
    if guard != "none":
        raise FidelityError(
            f"[Optimizer] pushdown_guard={guard!r}: guard branches must be "
            "inert for the fidelity argument (need pushdown_guard = none)")
    if basis != "logical" and not allow_measured_costs:
        raise FidelityError(
            f"[Optimizer] cost_basis={basis!r}: measured costs change which leg "
            "is the source — a different experiment. Pass --allow-measured-costs "
            "to record and proceed anyway")
    if ml:
        raise FidelityError(
            "DBHandler.USE_ML_OPTIMIZER is on: same-type legs stop tying on "
            "ml_cost, so execution order is no longer declaration order and "
            "'opt' means something else. Unset BLEND_USE_ML_OPTIMIZER")
    return {"pushdown_guard": guard, "cost_basis": basis,
            "use_ml_optimizer": ml,
            "allow_measured_costs": bool(allow_measured_costs)}


def resolve_semantic_knobs(cfg, *, cli_k_coarse, cli_ef, cli_exact_threshold) -> dict:
    backend = cfg.vector_backend
    if cli_k_coarse is not None:
        kc, kc_src = int(cli_k_coarse), "cli"
    elif cfg.faiss_k_coarse is not None:
        kc, kc_src = int(cfg.faiss_k_coarse), "config"
    else:
        kc, kc_src = None, "derived"
    if cli_ef is not None:
        ef, ef_src = int(cli_ef), "cli"
    elif cli_k_coarse is not None:
        ef, ef_src = int(cli_k_coarse), "derived"
    else:
        ef, ef_src = int(cfg.faiss_hnsw_ef_search), "config"
    if cli_exact_threshold is not None:
        et, et_src = int(cli_exact_threshold), "cli"
    else:
        et, et_src = cfg.exact_threshold, "config"
    if backend == "faiss" and kc is not None:
        if cli_ef is not None and cli_k_coarse is not None and ef != kc:
            raise FidelityError(
                f"faiss: --ef {ef} != --k-coarse {kc} is refused: faiss needs ef == k_coarse")
        if ef < kc:
            raise FidelityError(
                f"faiss + pinned k_coarse={kc} (source: {kc_src}) with "
                f"ef={ef} (source: {ef_src}): ef < k_coarse cannot fill the "
                "fetch. Pass --k-coarse (ef follows it) or --ef >= k_coarse")
    return {"k_coarse": kc, "k_coarse_source": kc_src,
            "ef_search": ef, "ef_search_source": ef_src,
            "exact_threshold": et, "exact_threshold_source": et_src,
            "vector_backend": backend}


def reference_knobs(n_gids: int, exact_threshold: int | None = None) -> dict:
    """Knobs for an untruncated reference leg: k_coarse = n_gids, threshold admitting n_gids."""
    et = int(n_gids) if exact_threshold is None else int(exact_threshold)
    if et < int(n_gids):
        raise FidelityError(
            f"reference leg exact_threshold={et} < n_gids={n_gids}: an "
            "inherited threshold cannot serve a reference leg")
    return {"exact_threshold": et, "k_coarse": int(n_gids)}


def _always_exact_when_filtered(n_gids, exact_threshold, *, vector_backend="faiss"):
    # `n_gids is not None` is load-bearing: an unfiltered semantic leg must stay HNSW
    return n_gids is not None


@contextlib.contextmanager
def forced_exact():
    """Patch retrieve.should_use_exact to go exact whenever filtered."""
    import src.Semantic.retrieve as retrieve
    saved = retrieve.should_use_exact
    retrieve.should_use_exact = _always_exact_when_filtered
    try:
        yield
    finally:
        retrieve.should_use_exact = saved


def assert_clean_ids(ids, *, where: str) -> None:
    for t in ids:
        if t is None or (isinstance(t, int) and t < 0):
            raise FidelityError(
                f"{where}: padded/invalid TableId {t!r} in a result list "
                "(padding check, promoted to an invariant)")


def depth_probe(cfg, *, k, n_gids, n_total, exact_used, k_coarse_pin, ef_pin):
    from src.Semantic import depths
    return depths.derive(
        k=k, vote_factor=cfg.vote_depth_factor, n_gids=n_gids, n_total=n_total,
        exact_used=exact_used, backend=cfg.vector_backend,
        pinned_k_coarse=k_coarse_pin, pinned_ef_search=ef_pin,
        restricted_k_coarse=cfg.restricted_k_coarse)
