"""Indexer + BuiltIndex protocols + shared config + validators."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np


@dataclass(frozen=True)
class IndexerConfig:
    quant: Literal["pq", "flat"] = "pq"
    pq_m: int | None = 28
    pq_nbits: int = 8
    hnsw_M: int = 64
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 64
    train_sample_size: int | None = None


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalisation; returns float32. Shared by build + query paths."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (x / norms).astype(np.float32)


def validate_pq_alignment(
    cfg: IndexerConfig,
    segment_offsets: dict[str, tuple[int, int]],
) -> None:
    """Raise ValueError if cfg.quant=='pq' and subvector boundaries don't align to segments."""
    if cfg.quant != "pq":
        return
    if cfg.pq_m is None:
        raise ValueError("quant=pq requires pq_m to be set")
    total_dim = max(end for _, end in segment_offsets.values())
    if total_dim % cfg.pq_m != 0:
        raise ValueError(f"pq_m={cfg.pq_m} does not divide total_dim={total_dim}")
    subvec_dim = total_dim // cfg.pq_m
    for name, (start, end) in segment_offsets.items():
        seg_dim = end - start
        if seg_dim % subvec_dim != 0:
            raise ValueError(
                f"segment {name!r} dim {seg_dim} not divisible by subvec_dim {subvec_dim}; "
                f"PQ would mix signals across the fusion boundary"
            )


class BuiltIndex(Protocol):
    def search(
        self,
        query: np.ndarray,
        k: int,
        weights: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """query: (Q, total_dim). Returns (scores: (Q,k), global_ids: (Q,k))."""
        ...

    def save(self, path: Path) -> None: ...


class Indexer(Protocol):
    name: str

    def build(
        self,
        vectors: np.ndarray,
        ids: np.ndarray,
        segment_offsets: dict[str, tuple[int, int]],
        config: IndexerConfig,
    ) -> BuiltIndex: ...

    def load(self, path: Path) -> BuiltIndex: ...
