from pathlib import Path

from src import paths
from src.dataset import DatasetConfig, load_dataset_config

_REPO = Path(paths.__file__).resolve().parent.parent


def test_config_path_default(monkeypatch):
    monkeypatch.delenv("BLEND_CONFIG", raising=False)
    assert paths.config_path() == _REPO / "config" / "config.ini"


def test_config_path_env(monkeypatch, tmp_path):
    monkeypatch.setenv("BLEND_CONFIG", str(tmp_path / "c.ini"))
    assert paths.config_path() == tmp_path / "c.ini"


def test_datasets_root_default(monkeypatch):
    monkeypatch.delenv("BLEND_DATASETS_DIR", raising=False)
    assert paths.datasets_root() == _REPO / "datasets"


def test_datasets_root_env_tilde(monkeypatch):
    monkeypatch.setenv("BLEND_DATASETS_DIR", "~/lake")
    assert paths.datasets_root() == Path("~/lake").expanduser()


def test_runs_root_default(monkeypatch):
    monkeypatch.delenv("BLEND_RUNS_DIR", raising=False)
    assert paths.runs_root() == _REPO / "runs"


def test_runs_root_env(monkeypatch, tmp_path):
    monkeypatch.setenv("BLEND_RUNS_DIR", str(tmp_path))
    assert paths.runs_root() == tmp_path


def test_dataset_root_env_beats_config(monkeypatch, tmp_path):
    ini = tmp_path / "c.ini"
    ini.write_text("[Dataset]\nname = santos\nroot = /from/ini\n")
    monkeypatch.setenv("BLEND_DATASETS_DIR", str(tmp_path / "from_env"))
    cfg = load_dataset_config(ini, None)
    assert cfg.root == tmp_path / "from_env"


def test_dataset_root_override_beats_env(monkeypatch, tmp_path):
    monkeypatch.setenv("BLEND_DATASETS_DIR", str(tmp_path / "from_env"))
    cfg = load_dataset_config(None, {"dataset_root": str(tmp_path / "explicit")})
    assert cfg.root == tmp_path / "explicit"


def test_dataset_root_default_factory(monkeypatch):
    monkeypatch.delenv("BLEND_DATASETS_DIR", raising=False)
    assert DatasetConfig().root == _REPO / "datasets"


import pytest

from src.Semantic.config import SemanticConfig
from src.DBHandler import DBHandler


def test_semanticconfig_uses_blend_config(monkeypatch, tmp_path):
    ini = tmp_path / "c.ini"
    ini.write_text("[Dataset]\nname = wonderland\n")
    monkeypatch.setenv("BLEND_CONFIG", str(ini))
    assert SemanticConfig.load().dataset.name == "wonderland"


def test_dbhandler_uses_blend_config(monkeypatch, tmp_path):
    missing = tmp_path / "nope.ini"
    monkeypatch.setenv("BLEND_CONFIG", str(missing))
    with pytest.raises(FileNotFoundError) as exc:
        DBHandler()
    assert str(missing) in str(exc.value)


def test_benchmark_dataset_dir_honors_env(monkeypatch, tmp_path):
    (tmp_path / "santos").mkdir()
    monkeypatch.setenv("BLEND_DATASETS_DIR", str(tmp_path))
    from src.Benchmark.datasource import dataset_dir
    assert dataset_dir("santos") == tmp_path / "santos"


def test_optimizer_runs_base_honors_env(monkeypatch, tmp_path):
    monkeypatch.setenv("BLEND_RUNS_DIR", str(tmp_path))
    from src.Optimizer.cli import _runs_base
    assert _runs_base() == tmp_path / "optimizer"
