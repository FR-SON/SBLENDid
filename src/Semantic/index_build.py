from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from . import approaches
from ._time import now_iso_z
from .config import SemanticConfig
from .encoders.base import SidecarMeta, write_sidecar
from .indexers.base import IndexerConfig, l2_normalize_rows
from .indexers.faiss_indexer import FaissIndexer
from .registry import (
    LakeTableEntry, build_registry, hash_registry, load_registry, sha256_ckpt,
)


def build_index_registry(csv_dir: Path, index_dir: Path) -> tuple[Path, int]:
    csv_paths = sorted(
        p for p in Path(csv_dir).iterdir()
        if p.suffix == ".csv" and not p.name.startswith("._")
    )
    entries = [LakeTableEntry(table_path=p, source_path=p.name) for p in csv_paths]
    registry_path = index_dir / "registry.parquet"
    n_cols = build_registry(entries, registry_path)
    return registry_path, n_cols


def finalize_index(
    index_dir: Path,
    arr: np.ndarray,
    meta: SidecarMeta,
    cfg: SemanticConfig,
    approach: str,
    index_name: str,
    dataset: str | None,
    extra_manifest: dict | None = None,
) -> Path:
    npy_path = index_dir / "embeddings.fp32.npy"
    np.save(npy_path, arr)
    write_sidecar(index_dir / "embeddings.sidecar.json", meta)

    if not meta.natively_normalized:
        arr = l2_normalize_rows(arr)

    indexer = FaissIndexer()
    icfg = IndexerConfig(
        quant=cfg.faiss_quant,
        pq_m=cfg.faiss_pq_m,
        pq_nbits=cfg.faiss_pq_nbits,
        hnsw_M=cfg.faiss_hnsw_M,
        hnsw_ef_construction=cfg.faiss_hnsw_ef_construction,
        hnsw_ef_search=cfg.faiss_hnsw_ef_search,
    )
    n_rows, dim = meta.n_rows, meta.segment_dim
    segment_offsets = {meta.segment_name: (0, dim)}
    ids = np.arange(n_rows, dtype=np.int64)

    if icfg.quant == "pq":
        ksub = 2 ** icfg.pq_nbits
        if icfg.pq_m is None:
            icfg = type(icfg)(**{**asdict(icfg), "pq_m": dim // 8})
        if n_rows < ksub * icfg.pq_m:
            print(
                f"[semantic-index-build] n={n_rows} < ksub*pq_m={ksub * icfg.pq_m};"
                " falling back to flat",
                file=sys.stderr, flush=True,
            )
            icfg = type(icfg)(**{**asdict(icfg), "quant": "flat", "pq_m": None})

    built = indexer.build(arr, ids, segment_offsets, icfg)
    built.save(index_dir / "hnsw.faiss")

    manifest = {
        "approach": approach,
        "index_name": index_name,
        "dataset": dataset,
        "encoder_version": meta.encoder_version,
        "dim": dim,
        "n_rows": n_rows,
        "segment_offsets": {k: list(v) for k, v in segment_offsets.items()},
        "registry_hash": meta.registry_hash,
        "ckpt_hash": meta.ckpt_hash,
        "faiss": {
            "quant": icfg.quant,
            "hnsw_M": icfg.hnsw_M,
            "ef_search_default": icfg.hnsw_ef_search,
            "pq_m": icfg.pq_m,
            "pq_nbits": icfg.pq_nbits,
        },
        "created_at": now_iso_z(),
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    (index_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return index_dir


def build_semantic_index(
    csv_dir: Path,
    approach: str,
    index_name: str,
    cfg: SemanticConfig,
) -> Path:
    """Build registry + embeddings + FAISS index for csv_dir."""
    plugin = approaches.get(approach)
    approach_dir = cfg.approach_dir(approach, index_name)
    bundle = plugin.load_artifacts(approach_dir)

    index_dir = cfg.index_dir(approach, index_name)
    index_dir.mkdir(parents=True, exist_ok=True)

    registry_path, n_cols = build_index_registry(csv_dir, index_dir)
    registry_hash = hash_registry(registry_path)

    encoder = plugin.build_encoder(bundle, cfg)
    refs = list(load_registry(registry_path))
    arr = np.zeros((n_cols, encoder.dim), dtype=np.float32)
    filled = np.zeros(n_cols, dtype=bool)
    print(f"[semantic-index-build] encoding {n_cols} columns with {encoder.name}",
          flush=True)
    for gid, vec in tqdm(encoder.encode(refs), total=n_cols,
                         desc=f"encode-{encoder.name}", unit="col",
                         file=sys.stderr):
        if vec.dtype != np.float32:
            vec = vec.astype(np.float32)
        arr[gid] = vec
        filled[gid] = True
    if not filled.all():
        missing = int((~filled).sum())
        raise RuntimeError(
            f"encoder {encoder.name!r} skipped {missing} of {n_cols} columns"
        )

    meta = SidecarMeta(
        segment_name=encoder.name,
        segment_dim=encoder.dim,
        n_rows=n_cols,
        registry_hash=registry_hash,
        ckpt_path=str(bundle.ckpt_path),
        ckpt_path_resolved=str(bundle.ckpt_path.resolve()),
        ckpt_hash=f"sha256:{sha256_ckpt(bundle.ckpt_path)}",
        encoder_version=encoder.encoder_version,
        natively_normalized=bool(encoder.natively_normalized),
    )
    return finalize_index(
        index_dir, arr, meta, cfg,
        approach=approach, index_name=index_name,
        dataset=bundle.extras.get("dataset"),
    )


__all__ = ["build_semantic_index", "build_index_registry", "finalize_index"]
