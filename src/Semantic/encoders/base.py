"""SegmentEncoder protocol + canonical sidecar format."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from .._time import now_iso_z
from pathlib import Path
from typing import Iterable, Iterator, Protocol

import numpy as np

from ..registry import ColumnRef


@dataclass(frozen=True)
class SidecarMeta:
    segment_name: str
    segment_dim: int
    n_rows: int
    registry_hash: str
    ckpt_path: str
    ckpt_path_resolved: str
    ckpt_hash: str
    encoder_version: str
    natively_normalized: bool


def write_sidecar(out_path: Path, meta: SidecarMeta) -> None:
    payload = {
        **asdict(meta),
        "dtype": "float32",
        "created_at": now_iso_z(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))


def read_sidecar(path: Path) -> dict:
    return json.loads(path.read_text())


def verify_sidecar_consistency(sidecar_path: Path, npy_path: Path) -> None:
    """Raise ValueError if sidecar's claimed shape doesn't match the npy."""
    data = read_sidecar(sidecar_path)
    arr = np.load(npy_path, mmap_mode="r")
    if arr.dtype != np.float32:
        raise ValueError(f"npy dtype {arr.dtype}, sidecar claims float32")
    if arr.ndim != 2:
        raise ValueError(f"npy ndim {arr.ndim}, expected 2")
    if arr.shape[0] != data["n_rows"]:
        raise ValueError(f"row mismatch: npy {arr.shape[0]}, sidecar {data['n_rows']}")
    if arr.shape[1] != data["segment_dim"]:
        raise ValueError(f"dim mismatch: npy {arr.shape[1]}, sidecar {data['segment_dim']}")


class SegmentEncoder(Protocol):
    """Build-time column encoder; the query path uses precomputed vectors unless a column is unindexed."""
    name: str
    dim: int
    encoder_version: str
    natively_normalized: bool
    na_cell: str

    def load(self, ckpt_path: Path) -> None: ...

    def encode(self, columns: Iterable[ColumnRef]) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (global_id, float32 vector) per column."""
        ...

    def encode_one(self, ref: ColumnRef, *, cells: list[str] | None = None) -> np.ndarray:
        """Encode a single column; `cells` replaces the lake read."""
        ...
