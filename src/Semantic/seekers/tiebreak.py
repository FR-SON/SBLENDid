"""Tiebreak ml_cost for semantic seekers: predicted unfiltered cost in seconds."""
from __future__ import annotations

from ..cost_predict import cost_unfiltered
from ..depths import effective_k_coarse


def tiebreak_cost(seeker) -> float:
    if getattr(seeker._cfg, "semantic_ml_cost", "off") != "analytic":
        return 1.0
    db = getattr(seeker, "DB", None)
    profile = db.optimizer_profile_str() if db is not None else None
    ms = cost_unfiltered(
        seeker.SEMANTIC_OP,
        seeker.input.shape[1],
        effective_k_coarse(seeker.k, vote_factor=seeker._vote_factor,
                           pinned_k_coarse=seeker._k_coarse),
        dataset_dir=seeker._cfg.dataset.dir(),
        profile=profile,
        cfg=seeker._cfg,
    )
    return ms / 1000.0
