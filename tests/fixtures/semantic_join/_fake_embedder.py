"""Deterministic stand-in for fasttext.get_sentence_vector."""

from __future__ import annotations
import hashlib

import numpy as np


def fake_embedder(d: int):
    def embed(cells: list[str]) -> np.ndarray:
        out = np.zeros((len(cells), d), dtype=np.float32)
        for i, c in enumerate(cells):
            h = hashlib.sha256(c.encode("utf-8")).digest()
            for j in range(d):
                out[i, j] = (h[j % 32] / 255.0) * 2.0 - 1.0
        return out
    return embed
