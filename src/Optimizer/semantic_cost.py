"""Pure analysis for the semantic cost-ordering deliverable. No I/O, no torch."""
from __future__ import annotations


def crossover(curve: list[tuple[int, float]], hnsw_runtime: float) -> int:
    """Return the largest |gids| whose exact runtime is <= hnsw_runtime, else 0."""
    below = [g for g, t in curve if t <= hnsw_runtime]
    return max(below) if below else 0


def optimal_play_mean(curve: list[tuple[int, float]], hnsw_runtime: float) -> float:
    """Mean of min(exact, hnsw_runtime) over the curve's |gids| points."""
    if not curve:
        return hnsw_runtime
    return sum(min(t, hnsw_runtime) for _, t in curve) / len(curve)


def assign_cost_integers(medians: dict[str, float]) -> dict[str, int]:
    """Map runtime medians to linear cost() integers, round(median / min_median), min 1."""
    if not medians:
        return {}
    lo = min(medians.values())
    return {name: max(1, round(val / lo)) for name, val in medians.items()}
