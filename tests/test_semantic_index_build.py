import json
from pathlib import Path

import numpy as np
import pytest

from src.Semantic.config import SemanticConfig
from src.Semantic.index_build import build_semantic_index
from src.Semantic.indexers.faiss_indexer import FaissIndexer


FIXTURE = Path(__file__).parent / "fixtures" / "semantic"


@pytest.fixture()
def staged(tmp_path):
    """Stage the fixture ckpt + aspects under the dataset layout."""
    root = tmp_path / "data"
    approach_dir = root / "test_ds" / "semantic" / "liftus" / "demo"
    (approach_dir / "ckpt").mkdir(parents=True)
    (approach_dir / "aspects").mkdir()
    for p in (FIXTURE / "ckpt").iterdir():
        (approach_dir / "ckpt" / p.name).write_bytes(p.read_bytes())
    for sub in (FIXTURE / "aspects").iterdir():
        out = approach_dir / "aspects" / sub.name
        out.mkdir()
        for f in sub.iterdir():
            (out / f.name).write_bytes(f.read_bytes())
    return root


def test_build_writes_all_artifacts(tmp_path, staged):
    cfg = SemanticConfig.load(
        path=tmp_path / "missing.ini",
        overrides={"dataset": "test_ds", "dataset_root": str(staged),
                   "faiss_quant": "flat"},
    )
    index_dir = build_semantic_index(
        csv_dir=FIXTURE / "csvs",
        approach="liftus",
        index_name="demo",
        cfg=cfg,
    )
    assert (index_dir / "registry.parquet").is_file()
    assert (index_dir / "embeddings.fp32.npy").is_file()
    assert (index_dir / "embeddings.sidecar.json").is_file()
    assert (index_dir / "hnsw.faiss").is_file()
    manifest = json.loads((index_dir / "manifest.json").read_text())
    assert manifest["approach"] == "liftus"
    assert manifest["index_name"] == "demo"
    assert manifest["dim"] == 128
    assert manifest["faiss"]["quant"] == "flat"
    assert manifest["n_rows"] > 0
    arr = np.load(index_dir / "embeddings.fp32.npy")
    assert arr.shape[1] == 128
    assert arr.shape[0] == manifest["n_rows"]


def test_build_index_searchable(tmp_path, staged):
    cfg = SemanticConfig.load(
        path=tmp_path / "missing.ini",
        overrides={"dataset": "test_ds", "dataset_root": str(staged),
                   "faiss_quant": "flat"},
    )
    index_dir = build_semantic_index(
        csv_dir=FIXTURE / "csvs", approach="liftus",
        index_name="demo", cfg=cfg,
    )
    built = FaissIndexer().load(index_dir / "hnsw.faiss")
    arr = np.load(index_dir / "embeddings.fp32.npy")
    q = arr[0:1].copy()
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    scores, ids = built.search(q.astype(np.float32), k=1)
    assert int(ids[0, 0]) == 0
