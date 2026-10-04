import numpy as np

from src.Semantic.indexers.base import IndexerConfig
from src.Semantic.indexers.faiss_indexer import FaissIndexer


def _unit_vectors(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def test_build_flat_search_returns_self_first(tmp_path):
    vecs = _unit_vectors(50, 128)
    ids = np.arange(50, dtype=np.int64)
    indexer = FaissIndexer()
    cfg = IndexerConfig(quant="flat", hnsw_M=16, hnsw_ef_construction=80, hnsw_ef_search=32)
    built = indexer.build(vecs, ids, {"liftus": (0, 128)}, cfg)
    out_path = tmp_path / "hnsw.faiss"
    built.save(out_path)

    scores, hit_ids = built.search(vecs[0:1], k=5)
    assert scores.shape == (1, 5)
    assert hit_ids.shape == (1, 5)
    assert int(hit_ids[0, 0]) == 0
    assert float(scores[0, 0]) > 0.99

    reloaded = FaissIndexer().load(out_path)
    s2, i2 = reloaded.search(vecs[0:1], k=5)
    assert int(i2[0, 0]) == 0


def test_set_ef_search_updates_hnsw():
    import numpy as np, faiss
    from src.Semantic.indexers.faiss_indexer import FaissIndexer
    from src.Semantic.indexers.base import IndexerConfig
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((64, 8)).astype("float32")
    ids = np.arange(64, dtype="int64")
    cfg = IndexerConfig(quant="flat", hnsw_M=8, hnsw_ef_construction=40,
                        hnsw_ef_search=16)
    built = FaissIndexer().build(vecs, ids, {"seg": (0, 8)}, cfg)
    built.set_ef_search(123)
    inner = faiss.downcast_index(built.raw_index.index)
    assert inner.hnsw.efSearch == 123
