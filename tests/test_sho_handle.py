import json

import numpy as np
import pytest

from src.Semantic.config import OperatorConfig, SemanticConfig, SemanticOp
from src.Semantic.emb_store import PersistentEmbeddingStore
from src.Semantic.simhash import codes_from_embeddings, get_hyperplanes

DIM = 8


def _vec(x):
    v = np.full(DIM, x, dtype=np.float32)
    v[0] = 1.0
    return v / np.linalg.norm(v)


def _dataset(tmp_path, manifest_overrides=None):
    ds_root = tmp_path / "datasets"
    sho_dir = ds_root / "fx" / "semantic" / "simhash" / "default"
    sho_dir.mkdir(parents=True)
    man = {"bits": 6, "seed": 0, "sample_seed": 0, "row_cap": 1000, "skip_numeric": True,
           "encoder_model": "m", "dim": DIM, "n_tables": 1, "n_codes": 1}
    man.update(manifest_overrides or {})
    (sho_dir / "manifest.json").write_text(json.dumps(man))
    (sho_dir / "codes.parquet").write_bytes(b"")
    store = PersistentEmbeddingStore(sho_dir / "emb_store", model="m")
    store.append({"foo": _vec(0.3), "bar": _vec(-0.7)})
    cfg = SemanticConfig.load(
        path=tmp_path / "absent.ini",
        overrides={"dataset": "fx", "dataset_root": str(ds_root)},
        operators={SemanticOp.SHO: OperatorConfig("simhash", "default")},
    )
    return cfg, man


def test_open_caches_by_signature(tmp_path):
    from src.Semantic.sho_artifacts import ShoHandle
    cfg, _ = _dataset(tmp_path)
    h1 = ShoHandle.open(cfg)
    h2 = ShoHandle.open(cfg)
    assert h1 is h2
    assert h1.manifest["bits"] == 6
    assert h1.hyperplanes.shape == (6, DIM)


def test_codes_for_values_matches_primitives(tmp_path):
    from src.Semantic.sho_artifacts import ShoHandle
    cfg, _ = _dataset(tmp_path)
    h = ShoHandle.open(cfg)
    hp = get_hyperplanes(6, DIM, seed=0)
    expected = codes_from_embeddings(np.stack([_vec(0.3), _vec(-0.7)]), hp).tolist()
    assert h.codes_for_values(["foo", "bar"]) == expected


def test_missing_manifest_hard_fails(tmp_path):
    from src.Semantic.sho_artifacts import ShoHandle
    cfg, _ = _dataset(tmp_path)
    (cfg.approach_dir("simhash", "default") / "manifest.json").unlink()
    ShoHandle._CACHE.clear()
    with pytest.raises(FileNotFoundError, match="manifest"):
        ShoHandle.open(cfg)


def test_missing_codes_artifact_hard_fails(tmp_path):
    from src.Semantic.sho_artifacts import ShoHandle
    cfg, _ = _dataset(tmp_path)
    (cfg.approach_dir("simhash", "default") / "codes.parquet").unlink()
    ShoHandle._CACHE.clear()
    with pytest.raises(FileNotFoundError, match="codes.parquet"):
        ShoHandle.open(cfg)


def test_store_model_mismatch_hard_fails(tmp_path):
    from src.Semantic.sho_artifacts import ShoHandle
    cfg, _ = _dataset(tmp_path, manifest_overrides={"encoder_model": "other"})
    ShoHandle._CACHE.clear()
    with pytest.raises(ValueError, match="model mismatch"):
        ShoHandle.open(cfg)
