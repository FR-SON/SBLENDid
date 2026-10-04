"""Derive retrieval depth, vote depth and beam width from the plan's `k`."""

from __future__ import annotations

from dataclasses import dataclass

# Fetch exceeds the vote because the query table's own columns are dropped before voting.
FETCH_MARGIN = 10

DEFAULT_VOTE_FACTOR = 2.0


@dataclass(frozen=True)
class Depths:
    """Resolved depths for one search call."""

    k_coarse: int
    k_vote: int
    ef_search: int | None
    """None leaves the configured beam alone (pgvector, exact path)."""
    derived: bool
    regime: str = "unrestricted"
    """One of "pinned" | "unrestricted" | "restricted_exact" | "restricted_ann"."""


def derive(
    *,
    k: int,
    vote_factor: float = DEFAULT_VOTE_FACTOR,
    n_gids: int | None = None,
    n_total: int | None = None,
    exact_used: bool = False,
    restricted_k_coarse: int = 500,
    backend: str = "faiss",
    pinned_k_coarse: int | None = None,
    pinned_ef_search: int | None = None,
) -> Depths:
    """Resolve (k_coarse, k_vote, ef_search) for one search."""
    if pinned_k_coarse is not None:
        kc = int(pinned_k_coarse)
        return Depths(k_coarse=kc, k_vote=kc, ef_search=pinned_ef_search,
                      derived=False, regime="pinned")

    if n_gids is not None:
        kc = int(restricted_k_coarse)
        if n_total:
            kc = min(kc, int(n_total))
        # faiss needs ef >= k_coarse to fill the width; pgvector's iterative scan refills regardless.
        ef = kc if backend == "faiss" else None
        return Depths(k_coarse=kc, k_vote=kc, ef_search=ef, derived=True,
                      regime=("restricted_exact" if exact_used
                              else "restricted_ann"))

    k_vote = max(1, int(round(vote_factor * max(1, int(k)))))

    k_coarse = k_vote + FETCH_MARGIN
    ef = k_coarse if backend == "faiss" else None
    return Depths(k_coarse=k_coarse, k_vote=k_vote, ef_search=ef, derived=True,
                  regime="unrestricted")


def effective_k_coarse(
    k: int,
    *,
    vote_factor: float = DEFAULT_VOTE_FACTOR,
    pinned_k_coarse: int | None = None,
) -> int:
    """The k_coarse an unrestricted search at this `k` will use."""
    return derive(k=k, vote_factor=vote_factor,
                  pinned_k_coarse=pinned_k_coarse).k_coarse
