import json

import numpy as np

from src.Semantic.config import SemanticConfig
from src.Semantic.encoders.base import SidecarMeta
from src.Semantic.index_build import build_index_registry, finalize_index
from src.Semantic.registry import load_registry


def test_build_index_registry_skips_appledouble(tmp_path):
    csvs = tmp_path / "csvs"
    csvs.mkdir()
    (csvs / "a.csv").write_text("x,y\n1,2\n")
    (csvs / "._a.csv").write_bytes(b"\x00junk")
    idx = tmp_path / "index"
    idx.mkdir()
    registry_path, n_cols = build_index_registry(csvs, idx)
    refs = list(load_registry(registry_path))
    assert n_cols == 2
    assert {r.table_id for r in refs} == {"a.csv"}


def _meta(n, dim, normalized=True):
    return SidecarMeta(
        segment_name="deepjoin", segment_dim=dim, n_rows=n,
        registry_hash="sha256:test", ckpt_path="imported",
        ckpt_path_resolved="imported", ckpt_hash="sha256:test",
        encoder_version="deepjoin@test", natively_normalized=normalized,
    )


def test_finalize_index_writes_artifacts(tmp_path):
    idx = tmp_path / "index"
    idx.mkdir()
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(6, 8)).astype(np.float32)
    cfg = SemanticConfig.load(overrides={"dataset": "x", "faiss_quant": "flat"})
    out = finalize_index(idx, arr, _meta(6, 8, normalized=False), cfg,
                         approach="deepjoin", index_name="default",
                         dataset="opendata")
    assert (out / "embeddings.fp32.npy").is_file()
    assert (out / "hnsw.faiss").is_file()
    sidecar = json.loads((out / "embeddings.sidecar.json").read_text())
    assert sidecar["segment_dim"] == 8 and sidecar["n_rows"] == 6
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["approach"] == "deepjoin"
    assert manifest["index_name"] == "default"
    assert manifest["dataset"] == "opendata"
    assert manifest["registry_hash"] == "sha256:test"
    raw = np.load(out / "embeddings.fp32.npy")
    assert not np.allclose(np.linalg.norm(raw, axis=1), 1.0)


def test_finalize_index_extra_manifest(tmp_path):
    idx = tmp_path / "index"
    idx.mkdir()
    arr = np.eye(4, dtype=np.float32)
    cfg = SemanticConfig.load(overrides={"dataset": "x", "faiss_quant": "flat"})
    finalize_index(idx, arr, _meta(4, 4), cfg, approach="liftus",
                   index_name="default", dataset=None,
                   extra_manifest={"imported_from": {"npy": "p"}})
    manifest = json.loads((idx / "manifest.json").read_text())
    assert manifest["imported_from"] == {"npy": "p"}
