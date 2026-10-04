"""DeepJoin artifact resolution + ckpt hashing units. No torch, no model load."""
import hashlib

import pytest

from src.Semantic.registry import sha256_ckpt, sha256_file


def test_sha256_ckpt_file_matches_sha256_file(tmp_path):
    f = tmp_path / "model.pth"
    f.write_bytes(b"weights")
    assert sha256_ckpt(f) == sha256_file(f)


def test_sha256_ckpt_dir_hashes_safetensors(tmp_path):
    d = tmp_path / "hf_dir"
    d.mkdir()
    st = d / "model.safetensors"
    st.write_bytes(b"tensor-bytes")
    assert sha256_ckpt(d) == hashlib.sha256(b"tensor-bytes").hexdigest()


def test_sha256_ckpt_dir_without_safetensors_raises(tmp_path):
    d = tmp_path / "hf_dir"
    d.mkdir()
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        sha256_ckpt(d)


import json

from src.Semantic import approaches
from src.Semantic.approaches import _deepjoin_load_artifacts


def _make_hf_dir(ckpt_root, name="deepjoin_opendata_2026-06-28T17-17-15Z",
                 dataset="opendata"):
    hf = ckpt_root / name
    hf.mkdir(parents=True)
    (hf / "modules.json").write_text("[]")
    (hf / "model.safetensors").write_bytes(b"w")
    (hf / "sidecar.json").write_text(json.dumps({"dataset": dataset}))
    trainer = hf / "_trainer" / "checkpoint-1"
    trainer.mkdir(parents=True)
    (trainer / "modules.json").write_text("[]")
    (trainer / "model.safetensors").write_bytes(b"w")
    return hf


def test_load_artifacts_happy_path(tmp_path):
    hf = _make_hf_dir(tmp_path / "ckpt")
    bundle = _deepjoin_load_artifacts(tmp_path)
    assert bundle.ckpt_path == hf
    assert bundle.extras["dataset"] == "opendata"
    assert bundle.extras["sentences_path"] is None


def test_load_artifacts_finds_sentence_cache(tmp_path):
    _make_hf_dir(tmp_path / "ckpt")
    sdir = tmp_path / "sentences"
    sdir.mkdir()
    (sdir / "opendata.flat.pkl").write_bytes(b"\x80\x04N.")
    bundle = _deepjoin_load_artifacts(tmp_path)
    assert bundle.extras["sentences_path"] == sdir / "opendata.flat.pkl"


def test_load_artifacts_ignores_appledouble(tmp_path):
    _make_hf_dir(tmp_path / "ckpt")
    junk = tmp_path / "ckpt" / "._junkdir"
    junk.mkdir()
    (junk / "modules.json").write_text("[]")
    (junk / "model.safetensors").write_bytes(b"w")
    bundle = _deepjoin_load_artifacts(tmp_path)
    assert bundle.extras["dataset"] == "opendata"


def test_load_artifacts_no_candidate(tmp_path):
    (tmp_path / "ckpt").mkdir()
    with pytest.raises(FileNotFoundError, match="exactly one"):
        _deepjoin_load_artifacts(tmp_path)


def test_load_artifacts_two_candidates(tmp_path):
    _make_hf_dir(tmp_path / "ckpt", name="deepjoin_a")
    _make_hf_dir(tmp_path / "ckpt", name="deepjoin_b")
    with pytest.raises(FileNotFoundError, match="exactly one"):
        _deepjoin_load_artifacts(tmp_path)


def test_load_artifacts_missing_sidecar(tmp_path):
    hf = _make_hf_dir(tmp_path / "ckpt")
    (hf / "sidecar.json").unlink()
    with pytest.raises(FileNotFoundError, match="sidecar"):
        _deepjoin_load_artifacts(tmp_path)


def test_load_artifacts_missing_dataset_field(tmp_path):
    hf = _make_hf_dir(tmp_path / "ckpt")
    (hf / "sidecar.json").write_text(json.dumps({"encoder": "deepjoin"}))
    with pytest.raises(KeyError, match="dataset"):
        _deepjoin_load_artifacts(tmp_path)


def test_deepjoin_plugin_registered():
    import src.Semantic  # noqa: F401
    plugin = approaches.get("deepjoin")
    assert plugin.name == "deepjoin"
