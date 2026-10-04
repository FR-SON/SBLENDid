from __future__ import annotations

import math


def precision_at_k(retrieved: list[str], relevant: set[str]) -> float:
    if not retrieved:
        return 0.0
    return sum(1 for r in retrieved if r in relevant) / len(retrieved)


def recall_at_k(retrieved: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    return sum(1 for r in retrieved if r in relevant) / len(relevant)


def average_precision(retrieved: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    hits = 0
    s = 0.0
    for i, t in enumerate(retrieved, start=1):
        if t in relevant:
            hits += 1
            s += hits / i
    return s / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, t in enumerate(retrieved, start=1):
        if t in relevant:
            return 1.0 / i
    return 0.0


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def hit_rate(retrieved: list[str], relevant: set[str]) -> float:
    return 1.0 if any(t in relevant for t in retrieved) else 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2)
              for i, t in enumerate(retrieved[:k]) if t in relevant)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg else 0.0
