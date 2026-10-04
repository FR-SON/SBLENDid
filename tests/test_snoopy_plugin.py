from __future__ import annotations
import json
import shutil
from pathlib import Path

import pytest

from src.Semantic.approaches import _snoopy_build_encoder, _snoopy_load_artifacts


FIXTURE = Path(__file__).parent / "fixtures" / "semantic_join"


def _seed_ckpt_dir(dst: Path) -> Path:
    """Copy the fixture ckpt + sidecar into a temp approach_dir and return it."""
    (dst / "ckpt").mkdir(parents=True)
    shutil.copy(FIXTURE / "ckpt" / "tiny.pth", dst / "ckpt" / "tiny.pth")
    shutil.copy(FIXTURE / "ckpt" / "tiny.json", dst / "ckpt" / "tiny.json")
    return dst


def test_load_artifacts_missing_pth_raises(tmp_path):
    (tmp_path / "ckpt").mkdir()
    with pytest.raises(FileNotFoundError, match="no .pth checkpoint"):
        _snoopy_load_artifacts(tmp_path)


def test_load_artifacts_missing_sidecar_raises(tmp_path):
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "ckpt" / "x.pth").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="missing sidecar JSON"):
        _snoopy_load_artifacts(tmp_path)


def test_load_artifacts_sidecar_missing_required_key_raises(tmp_path):
    approach = _seed_ckpt_dir(tmp_path)
    sidecar_path = approach / "ckpt" / "tiny.json"
    s = json.loads(sidecar_path.read_text())
    del s["row_cap"]
    sidecar_path.write_text(json.dumps(s))
    with pytest.raises(KeyError, match="row_cap"):
        _snoopy_load_artifacts(approach)


def _make_cfg(tmp_path):
    from src.Semantic.config import SemanticConfig
    from src.dataset import DatasetConfig
    from dataclasses import replace
    (tmp_path / "x" / "csvs").mkdir(parents=True, exist_ok=True)
    return replace(SemanticConfig(), dataset=DatasetConfig(name="x", root=tmp_path))


def test_build_encoder_missing_lake_root_raises(tmp_path, monkeypatch):
    from src.Semantic.config import SemanticConfig
    from src.dataset import DatasetConfig
    from dataclasses import replace
    approach = _seed_ckpt_dir(tmp_path)
    bundle = _snoopy_load_artifacts(approach)
    cfg = replace(SemanticConfig(), dataset=DatasetConfig(name="missing", root=tmp_path))
    monkeypatch.setenv("BLEND_SNOOPY_FASTTEXT_PATH", str(tmp_path / "fake.bin"))
    (tmp_path / "fake.bin").touch()
    with pytest.raises(FileNotFoundError, match="lake_root .* does not exist"):
        _snoopy_build_encoder(bundle, cfg)


def test_build_encoder_env_unset_raises(tmp_path, monkeypatch):
    approach = _seed_ckpt_dir(tmp_path)
    bundle = _snoopy_load_artifacts(approach)
    monkeypatch.delenv("BLEND_SNOOPY_FASTTEXT_PATH", raising=False)
    with pytest.raises(EnvironmentError, match="BLEND_SNOOPY_FASTTEXT_PATH is unset"):
        _snoopy_build_encoder(bundle, _make_cfg(tmp_path))


def test_build_encoder_env_file_missing_raises(tmp_path, monkeypatch):
    approach = _seed_ckpt_dir(tmp_path)
    bundle = _snoopy_load_artifacts(approach)
    monkeypatch.setenv("BLEND_SNOOPY_FASTTEXT_PATH", str(tmp_path / "nope.bin"))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _snoopy_build_encoder(bundle, _make_cfg(tmp_path))
