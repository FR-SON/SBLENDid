"""Lazy sentence-transformers encoder factory for SHO."""
from __future__ import annotations

import numpy as np


def build_encode_fn(model_name: str, device: str = "cpu", batch_size: int = 256):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)

    def encode(strings: list[str]) -> np.ndarray:
        # verbatim SemDisc text_encoder.encode: model.encode + L2-normalize
        emb = np.asarray(model.encode(list(strings), batch_size=batch_size), dtype=np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return emb / norms

    return encode
