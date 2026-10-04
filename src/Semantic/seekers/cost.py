"""cost() for semantic seekers: measured cost priced at the leg's own retrieval depth."""
from __future__ import annotations

from src.cost_model import cost_basis, cost_scale_anchor_s, resolve_cost

from ..cost_predict import cost_unfiltered
from ..depths import effective_k_coarse


def measured_cost(seeker, op: str, baseline: int) -> int:
    db = getattr(seeker, "DB", None)
    if cost_basis() != "measured":
        return baseline
    dataset_dir = seeker._cfg.dataset.dir()
    profile = db.optimizer_profile_str() if db is not None else None
    anchor_s = cost_scale_anchor_s(dataset_dir=dataset_dir, profile=profile)
    if anchor_s:
        try:
            ms = cost_unfiltered(
                op,
                seeker.input.shape[1],
                effective_k_coarse(seeker.k, vote_factor=seeker._vote_factor,
                                   pinned_k_coarse=seeker._k_coarse),
                dataset_dir=dataset_dir,
                profile=profile,
                cfg=seeker._cfg,
            )
            return max(1, round(ms / 1000.0 / anchor_s))
        except (FileNotFoundError, KeyError, ValueError):
            pass
    return resolve_cost(op, baseline, db=db)
