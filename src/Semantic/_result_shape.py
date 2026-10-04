import numpy as np


def pad_to_k(scores, gids, k):
    """Pad (scores, gids) to width k with (-inf, -1) to match FaissIndexer.search's (1, k) contract."""
    n = scores.shape[0]
    if n >= k:
        return scores[:k].reshape(1, -1), gids[:k].reshape(1, -1)
    pad_scores = np.full(k - n, -np.inf, dtype=np.float32)
    pad_gids = np.full(k - n, -1, dtype=np.int64)
    return (
        np.concatenate([scores, pad_scores]).reshape(1, -1),
        np.concatenate([gids, pad_gids]).reshape(1, -1),
    )
