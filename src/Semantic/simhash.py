"""SimHash primitives for the SHO seeker; codes are bit-identical to SemDisc's, packed MSB-first."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

SHO_ENCODER_MODEL = "paraphrase-distilroberta-base-v1"


def table_seed(global_seed: int, basename: str) -> int:
    digest = hashlib.sha256(f"{global_seed}:{basename}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def sample_row_indices(n_rows: int, row_cap: int, seed: int) -> np.ndarray:
    """Seeded ascending row sample without replacement; identity when n_rows <= row_cap."""
    if n_rows <= row_cap:
        return np.arange(n_rows)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=row_cap, replace=False))


def get_hyperplanes(bits: int, dim: int, seed: int) -> np.ndarray:
    # SemDisc: np.random.randn(bits, dim); seeded Generator here.
    return np.random.default_rng(seed).standard_normal((bits, dim))


def codes_from_embeddings(embeddings: np.ndarray, hyperplanes: np.ndarray) -> np.ndarray:
    """sign(emb · hp.T) > 0 packed MSB-first into int64 (== int(bitstring, 2))."""
    bits = (np.dot(embeddings, hyperplanes.T) > 0)
    weights = (1 << np.arange(bits.shape[1] - 1, -1, -1)).astype(np.int64)
    return bits.astype(np.int64) @ weights


def is_numeric_column(values: list[str]) -> bool:
    """SemDisc column_is_numeric: >50% digit chars across the top-10 frequent values."""
    if not values:
        return False
    top = pd.Series(values).value_counts().index[:10].tolist()
    joined = "".join(str(v) for v in top)
    if not joined:
        return False
    digits = sum(ch.isdigit() for ch in joined)
    return digits / len(joined) > 0.5


def clean_cell(v) -> str | None:
    """Raw str(value), None for ''/'nan'; deliberately not tokenize_cell (SemDisc embeds raw cased values)."""
    s = str(v)
    if s == "" or s.lower() == "nan":
        return None
    return s
