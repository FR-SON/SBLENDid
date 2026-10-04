"""FAISS-backed Indexer: IndexIDMap2(IndexHNSWPQ|IndexHNSWFlat) with IP metric."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from .base import IndexerConfig, validate_pq_alignment


class FaissBuiltIndex:
    """Search handle over a FAISS index; efSearch is passed per call as SearchParameters."""

    def __init__(self, index: faiss.Index, ef_search: int):
        self._index = index
        self._ef_search = int(ef_search)
        inner = faiss.downcast_index(index.index) if hasattr(index, "index") else index
        self._is_hnsw = hasattr(inner, "hnsw")
        if self._is_hnsw:
            inner.hnsw.efSearch = self._ef_search
        self._params = self._build_params()

    def _build_params(self):
        if not self._is_hnsw:
            return None
        params = faiss.SearchParametersHNSW()
        params.efSearch = self._ef_search
        return params

    def set_ef_search(self, ef_search: int) -> None:
        self._ef_search = int(ef_search)
        index = self._index
        inner = faiss.downcast_index(index.index) if hasattr(index, "index") else index
        if hasattr(inner, "hnsw"):
            inner.hnsw.efSearch = self._ef_search
        self._params = self._build_params()

    @property
    def ef_search(self) -> int:
        return self._ef_search

    @property
    def raw_index(self):
        return self._index

    def search(
        self,
        query: np.ndarray,
        k: int,
        weights: np.ndarray | None = None,
        ef: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-k inner-product search; `ef` overrides the beam for this call only."""
        if query.dtype != np.float32:
            query = query.astype(np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        params = self._params
        if ef is not None and self._is_hnsw:
            params = faiss.SearchParametersHNSW()
            params.efSearch = int(ef)
        if params is None:
            scores, ids = self._index.search(query, k)
        else:
            scores, ids = self._index.search(query, k, params=params)
        return scores, ids

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))


class FaissIndexer:
    name = "faiss"

    def build(
        self,
        vectors: np.ndarray,
        ids: np.ndarray,
        segment_offsets: dict[str, tuple[int, int]],
        config: IndexerConfig,
    ) -> FaissBuiltIndex:
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        if ids.dtype != np.int64:
            ids = ids.astype(np.int64)

        d = vectors.shape[1]

        if config.quant == "pq":
            validate_pq_alignment(config, segment_offsets)
            base = faiss.IndexHNSWPQ(
                d, config.pq_m, config.hnsw_M, config.pq_nbits, faiss.METRIC_INNER_PRODUCT,
            )
            if base.metric_type != faiss.METRIC_INNER_PRODUCT:
                raise RuntimeError(
                    f"FAISS built IndexHNSWPQ with metric={base.metric_type}, "
                    f"expected METRIC_INNER_PRODUCT ({faiss.METRIC_INNER_PRODUCT}). "
                    f"Check faiss-cpu version; this combination requires faiss-cpu >= 1.7.4."
                )
            base.hnsw.efConstruction = config.hnsw_ef_construction

            n = vectors.shape[0]
            ksub = 2 ** config.pq_nbits
            min_train = ksub * config.pq_m
            if n < min_train:
                raise ValueError(
                    f"Too few training samples: need at least {min_train} "
                    f"(pq_m={config.pq_m} × ksub={ksub}), got {n}"
                )
            sample_size = config.train_sample_size or min(n, 256_000)
            if sample_size < 256:
                sample_size = n
            rng = np.random.RandomState(0)
            idxs = rng.choice(n, size=min(n, sample_size), replace=False)
            base.train(vectors[idxs])
        elif config.quant == "flat":
            base = faiss.IndexHNSWFlat(d, config.hnsw_M, faiss.METRIC_INNER_PRODUCT)
            if base.metric_type != faiss.METRIC_INNER_PRODUCT:
                raise RuntimeError(
                    f"FAISS built IndexHNSWFlat with metric={base.metric_type}, "
                    f"expected METRIC_INNER_PRODUCT ({faiss.METRIC_INNER_PRODUCT})."
                )
            base.hnsw.efConstruction = config.hnsw_ef_construction
        else:
            raise NotImplementedError(
                f"FaissIndexer supports quant in {{'pq','flat'}}, got {config.quant!r}"
            )

        wrapped = faiss.IndexIDMap2(base)
        wrapped.add_with_ids(vectors, ids)

        return FaissBuiltIndex(wrapped, ef_search=config.hnsw_ef_search)

    def load(self, path: Path) -> FaissBuiltIndex:
        index = faiss.read_index(str(path))
        manifest_path = path.with_suffix(".manifest.json")
        ef_search = 64
        if manifest_path.exists():
            m = json.loads(manifest_path.read_text())
            ef_search = int(m.get("hnsw", {}).get("ef_search_default", ef_search))
        return FaissBuiltIndex(index, ef_search=ef_search)
