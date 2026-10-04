from __future__ import annotations

import hashlib
import json
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pandas as pd

from . import approaches
from ._result_shape import pad_to_k as _pad_to_k
from .aggregators.base import Aggregator
from .config import SemanticConfig
from .encoders.base import SegmentEncoder, verify_sidecar_consistency
from . import depths
from .indexers.base import BuiltIndex, l2_normalize_rows
from .indexers.faiss_indexer import FaissIndexer
from .registry import load_registry


_HANDLE_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()
_WARNED_EF: set = set()
_QUERY_ENCODE_WARNED: set = set()
_QVEC_CACHE_MAX = 256
QUERY_SENTINEL = "__query__"


def _warn_if_ef_below_k(cfg: SemanticConfig) -> None:
    """faiss cannot fill k_coarse slots when ef < k_coarse (pads -1); pgvector's iterative scan is exempt."""
    if cfg.vector_backend != "faiss":
        return
    if cfg.faiss_k_coarse is None:
        return
    ef, kc = int(cfg.faiss_hnsw_ef_search), int(cfg.faiss_k_coarse)
    if ef >= kc:
        return
    key = (ef, kc)
    if key in _WARNED_EF:
        return
    _WARNED_EF.add(key)
    import warnings
    warnings.warn(
        f"faiss_hnsw_ef_search={ef} < faiss_k_coarse={kc} on the faiss backend: "
        f"faiss cannot return {kc} candidates with an ef of {ef} and will pad the "
        f"remainder with -1, silently lowering recall. Set ef_search >= k_coarse "
        f"(ef == k_coarse is the measured knee on every lake).",
        stacklevel=2,
    )


def reset_semantic_cache() -> None:
    with _CACHE_LOCK:
        for handle in _HANDLE_CACHE.values():
            pg = handle.__dict__.get("pg")
            if pg is not None:
                try:
                    pg.close()
                except Exception:
                    pass
        _HANDLE_CACHE.clear()
        _WARNED_EF.clear()
        _QUERY_ENCODE_WARNED.clear()


class QueryColumnNotIndexed(KeyError):
    """Query (table_id, col_name) absent from the registry."""


@dataclass(eq=False)
class IndexHandle:
    cfg: SemanticConfig
    approach: str
    index_name: str
    manifest: dict
    sidecar: dict
    dim: int
    natively_normalized: bool
    gid_to_table: dict[int, str]
    gid_to_col_name: dict[int, str | None]
    table_to_int_id: dict[str, int]
    faiss_index: BuiltIndex
    aggregator: Aggregator
    vectors: np.memmap
    vector_backend: str = "faiss"
    table_to_gids: dict[str, list[int]] = field(default_factory=dict)
    int_to_table: dict[int, str] = field(default_factory=dict)
    table_col_to_gid: dict[tuple[str, str], int] = field(default_factory=dict)
    _qvec_cache: "OrderedDict[tuple, np.ndarray]" = field(default_factory=OrderedDict, repr=False)

    @cached_property
    def encoder(self) -> SegmentEncoder:
        """Lazy encoder: index build, eval callers, and query-time encoding of unindexed columns."""
        plugin = approaches.get(self.approach)
        bundle = plugin.load_artifacts(
            self.cfg.approach_dir(self.approach, self.index_name)
        )
        return plugin.build_encoder(bundle, self.cfg)

    @cached_property
    def pg(self):
        """Lazy pgvector store; only built when vector_backend == 'pgvector'."""
        from .pgvector_store import PgVectorStore
        return PgVectorStore(
            self.cfg.dataset.name,
            self.approach,
            self.index_name,
            hnsw_ef_search=self.cfg.faiss_hnsw_ef_search,
            hnsw_iterative_scan=self.cfg.pgvector_hnsw_iterative_scan,
        )

    @property
    def vector_count(self) -> int:
        return self.pg.vector_count if self.vector_backend == "pgvector" else int(self.vectors.shape[0])

    def get_vector(self, gid: int) -> np.ndarray:
        if self.vector_backend == "pgvector":
            return self.pg.get_vector(gid)
        return np.asarray(self.vectors[gid], dtype=np.float32)

    def search_gid(self, gid, k, table_filter=None, exact_threshold=None,
                   ef=None):
        """(scores (1,k), gids (1,k)). FAISS reuses faiss_or_exact_search; pgvector goes to PG."""
        if self.vector_backend == "pgvector":
            exact = should_use_exact(
                _count_gids(self, table_filter), exact_threshold,
                vector_backend=self.vector_backend,
            )
            allowed = _pg_allowed_ids(
                table_filter, self.table_to_int_id,
                postfilter=self.cfg.pg_pushdown_mode == "postfilter",
                exact=exact,
            )
            return self.pg.search_gid(int(gid), k, allowed_int_ids=allowed, exact=exact)
        vec = self.get_vector(gid)
        q = vec[None, :]
        if not self.natively_normalized:
            q = l2_normalize_rows(q)
        return faiss_or_exact_search(self, q, k=k, table_filter=table_filter,
                                     exact_threshold=exact_threshold, ef=ef)

    def query_vectors(self, qtid: str, df: pd.DataFrame, cols: list) -> dict[str, np.ndarray]:
        """{str(col): query-ready vector} for unindexed columns, encoded from their cells and cached."""
        from .query_encode import encode_query_columns
        out: dict[str, np.ndarray] = {}
        todo: list[tuple] = []
        for col in cols:
            key = (qtid, str(col), _series_digest(df[col]))
            hit = self._qvec_cache.get(key)
            if hit is None:
                todo.append((col, key))
            else:
                self._qvec_cache.move_to_end(key)
                out[str(col)] = hit
        if todo:
            fresh = encode_query_columns(self.encoder, qtid, df, [c for c, _ in todo])
            for col, key in todo:
                vec = fresh[str(col)]
                if not self.natively_normalized:
                    vec = l2_normalize_rows(vec[None, :])[0]
                self._qvec_cache[key] = vec
                if len(self._qvec_cache) > _QVEC_CACHE_MAX:
                    self._qvec_cache.popitem(last=False)
                out[str(col)] = vec
        return out

    @classmethod
    def open(
        cls,
        cfg: SemanticConfig,
        approach: str,
        index_name: str,
    ) -> "IndexHandle":
        key = (cfg.signature(), approach, index_name)
        with _CACHE_LOCK:
            cached = _HANDLE_CACHE.get(key)
            if cached is not None:
                return cached
            handle = cls._build(cfg, approach, index_name)
            _HANDLE_CACHE[key] = handle
            return handle

    @classmethod
    def _build(
        cls,
        cfg: SemanticConfig,
        approach: str,
        index_name: str,
    ) -> "IndexHandle":
        index_dir = cfg.index_dir(approach, index_name)
        manifest_path = index_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Semantic index {approach}/{index_name} not built; "
                f"missing {manifest_path}. Build it with "
                f"scripts/create_semantic_index.py."
            )
        manifest = json.loads(manifest_path.read_text())
        sidecar_path = index_dir / "embeddings.sidecar.json"
        if not sidecar_path.is_file():
            raise FileNotFoundError(
                f"Semantic index {approach}/{index_name} missing "
                f"{sidecar_path}; rebuild the index."
            )
        sidecar = json.loads(sidecar_path.read_text())

        gid_to_table: dict[int, str] = {}
        gid_to_col_name: dict[int, str | None] = {}
        table_to_gids: dict[str, list[int]] = {}
        table_col_to_gid: dict[tuple[str, str], int] = {}
        for r in load_registry(index_dir / "registry.parquet"):
            gid_to_table[r.global_id] = r.table_id
            gid_to_col_name[r.global_id] = r.col_name
            table_to_gids.setdefault(r.table_id, []).append(r.global_id)
            if r.col_name is not None:
                table_col_to_gid[(r.table_id, r.col_name)] = r.global_id

        table_to_int_id = _load_basename_map(cfg, gid_to_table.values())
        int_to_table = {v: k for k, v in table_to_int_id.items()}

        aggregator = approaches.get(approach).build_aggregator()

        faiss_index = FaissIndexer().load(index_dir / "hnsw.faiss")
        faiss_index.set_ef_search(cfg.faiss_hnsw_ef_search)
        _warn_if_ef_below_k(cfg)

        vectors_path = index_dir / "embeddings.fp32.npy"
        if not vectors_path.is_file():
            raise FileNotFoundError(
                f"Semantic index {approach}/{index_name} missing "
                f"embeddings.fp32.npy at {vectors_path}; rebuild the index."
            )
        verify_sidecar_consistency(sidecar_path, vectors_path)
        vectors = np.load(vectors_path, mmap_mode="r")
        if vectors.shape[0] != len(gid_to_table):
            raise ValueError(
                f"embeddings.fp32.npy at {vectors_path} has "
                f"{vectors.shape[0]} rows but registry has "
                f"{len(gid_to_table)}; rebuild the index."
            )
        dim = int(sidecar["segment_dim"])
        nn_flag = sidecar["natively_normalized"]
        if not isinstance(nn_flag, bool):
            raise ValueError(
                f"sidecar at {sidecar_path} has natively_normalized="
                f"{nn_flag!r} (type {type(nn_flag).__name__}); must be a JSON "
                "bool. Rebuild the index."
            )

        return cls(
            cfg=cfg, approach=approach, index_name=index_name,
            manifest=manifest, sidecar=sidecar,
            dim=dim,
            natively_normalized=nn_flag,
            gid_to_table=gid_to_table, gid_to_col_name=gid_to_col_name,
            table_to_int_id=table_to_int_id,
            faiss_index=faiss_index, aggregator=aggregator,
            table_to_gids=table_to_gids, int_to_table=int_to_table,
            vectors=vectors,
            vector_backend=cfg.vector_backend,
            table_col_to_gid=table_col_to_gid,
        )


def _load_basename_map(
    cfg: SemanticConfig,
    registry_table_ids,
) -> dict[str, int]:
    if not cfg.blend_basenames_path.is_file():
        raise FileNotFoundError(
            f"blend_basenames_path={cfg.blend_basenames_path} does not exist. "
            f"Run scripts/create_blend_csv_index.py --dataset {cfg.dataset.name} "
            "to create it."
        )
    df = pd.read_parquet(cfg.blend_basenames_path)
    if set(df.columns) < {"table_int_id", "basename"}:
        raise ValueError(
            f"blend_basenames sidecar must have columns (table_int_id, basename); "
            f"got {list(df.columns)}"
        )
    mapping = dict(zip(df["basename"].astype(str), df["table_int_id"].astype(int)))
    missing = sorted(set(registry_table_ids) - mapping.keys())
    if missing:
        raise KeyError(
            f"blend basenames sidecar at {cfg.blend_basenames_path} is missing "
            f"{len(missing)} table_ids that the semantic registry references; "
            f"first few: {missing[:5]}. Re-run scripts/create_blend_csv_index.py "
            "over the same csv_dir that the semantic index was built from."
        )
    return mapping


def _series_digest(series: pd.Series) -> str:
    # Digest what the encoder will see (cell strings + NaN mask), not raw dtype bits.
    h = hashlib.sha1(pd.util.hash_pandas_object(series.astype(str), index=False).values.tobytes())
    h.update(series.isna().values.tobytes())
    return h.hexdigest()


def _resolve_query_table_id(query_table_id, df: pd.DataFrame, policy: str) -> str:
    qtid = query_table_id or df.attrs.get("table_id")
    if qtid:
        return str(qtid)
    if policy == "auto":
        return QUERY_SENTINEL
    raise ValueError(
        "semantic search requires query_table_id (kwarg or df.attrs['table_id']) "
        "when query_encode = off."
    )


def _announce_query_encode(handle: "IndexHandle", qtid: str, cols: list, skipped: list) -> None:
    key = (handle.approach, handle.index_name, qtid)
    if key in _QUERY_ENCODE_WARNED:
        return
    _QUERY_ENCODE_WARNED.add(key)
    print(
        f"[SEMANTIC][QUERY-ENCODE] approach={handle.approach}/{handle.index_name} "
        f"table={qtid!r} cols={len(cols)} skipped={[str(c) for c in skipped]}: not in the "
        "index, encoding at query time (encoder loaded; results are not comparable to "
        "in-index lookups)",
        file=sys.stderr, flush=True,
    )


def _encode_missing(handle: "IndexHandle", qtid: str, df: pd.DataFrame, missing: list,
                    policy: str) -> tuple[set[str], dict[str, np.ndarray]]:
    """(skipped str(col) set, {str(col): vector}) for query columns absent from the registry."""
    if not missing:
        return set(), {}
    if policy != "auto":
        raise QueryColumnNotIndexed(
            f"semantic search: query (table_id={qtid!r}, col_name={missing[0]!r}) not present "
            f"in the {handle.approach}/{handle.index_name} registry. the query column must be "
            "enrolled in the index (query_encode = off)."
        )
    from .query_encode import empty_columns
    skipped = empty_columns(df, missing)
    todo = [c for c in missing if c not in skipped]
    if todo or skipped:
        _announce_query_encode(handle, qtid, todo, skipped)
    encoded = handle.query_vectors(qtid, df, todo) if todo else {}
    return {str(c) for c in skipped}, encoded


def _make_aggregation_ctx(handle, q_rows, query_table_id):
    def candidate_vectors_fn(table_id: str) -> np.ndarray:
        gids = handle.table_to_gids.get(table_id, [])
        if not gids:
            return np.empty((0, handle.dim), dtype=np.float32)
        if handle.vector_backend == "pgvector":
            mat = np.stack([handle.pg.get_vector(g) for g in gids]).astype(np.float32)
        else:
            mat = np.array(handle.vectors[np.asarray(gids, dtype=np.int64)],
                           dtype=np.float32)
        if not handle.natively_normalized:
            mat = l2_normalize_rows(mat)
        return mat

    from .aggregators.base import AggregationContext
    return AggregationContext(
        qmat=np.vstack(q_rows).astype(np.float32),
        candidate_vectors_fn=candidate_vectors_fn,
        query_table_id=query_table_id,
    )


def _count_gids(handle: "IndexHandle", table_filter: set[str] | None) -> int | None:
    if table_filter is None:
        return None
    return sum(len(handle.table_to_gids.get(t, ())) for t in table_filter)


def should_use_exact(
    n_gids: int | None,
    exact_threshold: int | None,
    *,
    vector_backend: str = "faiss",
) -> bool:
    """Return whether the exact path replaces HNSW (|gids| <= threshold; never on pgvector)."""
    if vector_backend == "pgvector":
        return False
    return (
        n_gids is not None
        and exact_threshold is not None
        and n_gids <= exact_threshold
    )


def _pg_allowed_ids(table_filter, table_to_int_id, *, postfilter, exact):
    if table_filter is None:
        return None
    if postfilter and not exact:
        return None
    return {table_to_int_id[t] for t in table_filter if t in table_to_int_id}


def _pg_excluded_ids(exclude_filter, table_to_int_id, *, postfilter, exact):
    if not exclude_filter:
        return None
    if postfilter and not exact:
        return None
    return {table_to_int_id[t] for t in exclude_filter if t in table_to_int_id}


def apply_rollup_filters(ranked, table_filter, exclude_filter, *, allow_active):
    """Apply the allow-list (if allow_active) and deny-list to a ranked table list."""
    if allow_active and table_filter is not None:
        ranked = [(t, s) for t, s in ranked if t in table_filter]
    if exclude_filter:
        ranked = [(t, s) for t, s in ranked if t not in exclude_filter]
    return ranked


@dataclass(frozen=True)
class _ExactBundle:
    gids: np.ndarray
    vecs: np.ndarray


def prepare_exact_bundle(
    handle: IndexHandle,
    table_filter: set[str],
) -> _ExactBundle:
    """Gather and L2-normalise the vectors of every gid in table_filter."""
    allowed_gids = np.fromiter(
        (g for tid in sorted(table_filter)
         for g in handle.table_to_gids.get(tid, ())),
        dtype=np.int64,
    )
    if allowed_gids.size == 0:
        return _ExactBundle(
            gids=allowed_gids,
            vecs=np.empty((0, handle.dim), dtype=np.float32),
        )
    vecs = np.array(handle.vectors[allowed_gids], copy=False)
    if not handle.natively_normalized:
        vecs = l2_normalize_rows(vecs)
    return _ExactBundle(gids=allowed_gids, vecs=vecs)


def _exact_topk(
    q: np.ndarray, k: int, bundle: _ExactBundle,
) -> tuple[np.ndarray, np.ndarray]:
    if k <= 0 or bundle.gids.size == 0:
        return _pad_to_k(
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
            max(k, 0),
        )
    q2 = np.ascontiguousarray(q.reshape(1, -1), dtype=np.float32)
    scores = (q2 @ bundle.vecs.T)[0]
    top_k = min(k, scores.shape[0])
    part = np.argpartition(-scores, top_k - 1)[:top_k]
    order = part[np.argsort(-scores[part])]
    return _pad_to_k(scores[order], bundle.gids[order], k)


def faiss_or_exact_search(
    handle: IndexHandle,
    q: np.ndarray,
    k: int,
    table_filter: set[str] | None = None,
    exact_threshold: int | None = None,
    exact_bundle: _ExactBundle | None = None,
    ef: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Search one query vector via exact brute-force or FAISS; returns (scores, gids) of shape (1, k)."""
    if exact_bundle is None and should_use_exact(
        _count_gids(handle, table_filter), exact_threshold,
        vector_backend=handle.vector_backend,
    ):
        exact_bundle = prepare_exact_bundle(handle, table_filter)  # type: ignore[arg-type]
    if exact_bundle is not None:
        return _exact_topk(q, k, exact_bundle)
    return handle.faiss_index.search(q, k=k, ef=ef)


def search(
    handle: IndexHandle,
    df: pd.DataFrame,
    k: int,
    table_filter: set[str] | None = None,
    k_coarse: int | None = None,
    query_table_id: str | None = None,
    exact_threshold: int | None = None,
    exclude_filter: set[str] | None = None,
    aggregator: "Aggregator | None" = None,
    vote_factor: float | None = None,
    query_encode: str | None = None,
) -> list[int]:
    """Resolve each query column to its stored vector (or encode it when unindexed), search, aggregate, return top-k Blend table ids."""
    if df.shape[1] == 0:
        return []
    policy = query_encode or handle.cfg.query_encode
    qtid = _resolve_query_table_id(query_table_id, df, policy)
    missing = [c for c in df.columns if (qtid, str(c)) not in handle.table_col_to_gid]
    skipped, encoded = _encode_missing(handle, qtid, df, missing, policy)
    n_gids = _count_gids(handle, table_filter)
    if handle.vector_backend == "pgvector":
        exact_used = False
    else:
        exact_used = should_use_exact(
            n_gids, exact_threshold, vector_backend=handle.vector_backend)
    eff_depths = depths.derive(
        k=k,
        vote_factor=(handle.cfg.vote_depth_factor
                     if vote_factor is None else float(vote_factor)),
        n_gids=n_gids,
        n_total=(handle.vector_count if n_gids is not None else None),
        exact_used=exact_used,
        backend=handle.vector_backend,
        pinned_k_coarse=(handle.cfg.faiss_k_coarse
                         if k_coarse is None else int(k_coarse)),
        pinned_ef_search=handle.cfg.faiss_hnsw_ef_search,
        restricted_k_coarse=handle.cfg.restricted_k_coarse,
    )
    eff_k_coarse = eff_depths.k_coarse
    if handle.vector_backend == "pgvector":
        return _search_pgvector(
            handle, df, k, table_filter=table_filter,
            k_coarse=eff_k_coarse, k_vote=eff_depths.k_vote, query_table_id=qtid,
            exact_threshold=exact_threshold,
            exclude_filter=exclude_filter,
            aggregator=aggregator,
            skipped=skipped, encoded=encoded,
        )
    natively_normed = handle.natively_normalized
    agg = aggregator if aggregator is not None else handle.aggregator
    need_ctx = getattr(agg, "requires_ctx", False)

    exact_bundle = (
        prepare_exact_bundle(handle, table_filter)  # type: ignore[arg-type]
        if exact_used else None
    )

    per_col_hits: list[tuple[list[str], list[float]]] = []
    q_rows: list[np.ndarray] = []
    for col_name in df.columns:
        if str(col_name) in skipped:
            continue
        gid = handle.table_col_to_gid.get((qtid, str(col_name)))
        if gid is None:
            q = encoded[str(col_name)][None, :]
        else:
            q = np.asarray(handle.vectors[gid], dtype=np.float32)[None, :]
            if not natively_normed:
                q = l2_normalize_rows(q)
        if need_ctx:
            q_rows.append(q[0])
        scores, gids = faiss_or_exact_search(
            handle, q, k=eff_k_coarse, exact_bundle=exact_bundle,
            ef=eff_depths.ef_search,
        )
        tables: list[str] = []
        dists: list[float] = []
        for g, s in zip(gids[0].tolist(), scores[0].tolist()):
            if g < 0:
                continue
            tid = handle.gid_to_table.get(int(g))
            if tid is None or tid == qtid:
                continue
            tables.append(tid)
            dists.append(-float(s))
        per_col_hits.append((tables, dists))

    if not per_col_hits:
        return []
    ctx = _make_aggregation_ctx(handle, q_rows, qtid) if need_ctx else None
    ranked = agg.aggregate(per_col_hits, k=eff_depths.k_vote, ctx=ctx)
    ranked = apply_rollup_filters(
        ranked, table_filter, exclude_filter, allow_active=not exact_used)
    return [handle.table_to_int_id[tid] for tid, _ in ranked[:k]]


def _search_pgvector(handle, df, k, *, table_filter, k_coarse, k_vote,
                     query_table_id, exact_threshold, exclude_filter=None,
                     aggregator=None, skipped=frozenset(), encoded=None):
    exact = should_use_exact(
        _count_gids(handle, table_filter), exact_threshold,
        vector_backend=handle.vector_backend,
    )
    postfilter = handle.cfg.pg_pushdown_mode == "postfilter"
    allowed = _pg_allowed_ids(table_filter, handle.table_to_int_id,
                              postfilter=postfilter, exact=exact)
    excluded = _pg_excluded_ids(exclude_filter, handle.table_to_int_id,
                                postfilter=postfilter, exact=exact)
    agg = aggregator if aggregator is not None else handle.aggregator
    need_ctx = getattr(agg, "requires_ctx", False)
    q_rows: list[np.ndarray] = []
    per_col_hits: list[tuple[list[str], list[float]]] = []
    encoded = encoded or {}
    for col_name in df.columns:
        if str(col_name) in skipped:
            continue
        gid = handle.table_col_to_gid.get((query_table_id, str(col_name)))
        if gid is None:
            vec = encoded[str(col_name)]
            if need_ctx:
                q_rows.append(vec)
            scores, gids = handle.pg.search_vec(
                vec, k_coarse, allowed_int_ids=allowed,
                excluded_int_ids=excluded, exact=exact)
        else:
            if need_ctx:
                vec = handle.pg.get_vector(gid)
                if not handle.natively_normalized:
                    vec = l2_normalize_rows(vec[None, :])[0]
                q_rows.append(vec)
            scores, gids = handle.pg.search_gid(
                gid, k_coarse, allowed_int_ids=allowed,
                excluded_int_ids=excluded, exact=exact)
        tables, dists = [], []
        for g, s in zip(gids[0].tolist(), scores[0].tolist()):
            if g < 0:
                continue
            tid = handle.gid_to_table.get(int(g))
            if tid is None or tid == query_table_id:
                continue
            tables.append(tid)
            dists.append(-float(s))
        per_col_hits.append((tables, dists))
    if not per_col_hits:
        return []
    ctx = _make_aggregation_ctx(handle, q_rows, query_table_id) if need_ctx else None
    ranked = agg.aggregate(per_col_hits, k=k_vote, ctx=ctx)
    ranked = apply_rollup_filters(
        ranked, table_filter, exclude_filter, allow_active=not exact)
    return [handle.table_to_int_id[t] for t, _ in ranked[:k]]


__all__ = [
    "IndexHandle", "QUERY_SENTINEL", "search", "apply_rollup_filters", "faiss_or_exact_search",
    "prepare_exact_bundle", "should_use_exact", "reset_semantic_cache",
]
