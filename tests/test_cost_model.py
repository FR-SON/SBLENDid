import json
import textwrap
from pathlib import Path

import pytest

import src.cost_model as cm


def _write_config(tmp_path: Path, *, basis: str | None, ds_root: Path, name: str = "ds") -> Path:
    optimizer = "" if basis is None else f"[Optimizer]\ncost_basis = {basis}\n"
    cfg = tmp_path / "config.ini"
    cfg.write_text(textwrap.dedent(f"""\
        [Dataset]
        name = {name}
        root = {ds_root}

        [Database]
        dbms = duckdb
        db_filename = blend.duckdb
    """) + optimizer)
    return cfg


def _write_costs(ds_root: Path, name: str, costs: dict) -> Path:
    dest = ds_root / name / "optimizer" / "costs.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"costs": costs, "report": {}, "metadata": {}}))
    return dest


@pytest.fixture(autouse=True)
def _clear_cache():
    cm._reset_cache()
    yield
    cm._reset_cache()


def test_logical_is_default_when_section_absent(tmp_path):
    cfg = _write_config(tmp_path, basis=None, ds_root=tmp_path / "data")
    assert cm.resolve_cost("SC", 4, config_path=cfg) == 4
    assert cm.resolve_cost("MC", 10, config_path=cfg) == 10


def test_logical_explicit(tmp_path):
    cfg = _write_config(tmp_path, basis="logical", ds_root=tmp_path / "data")
    assert cm.resolve_cost("C", 6, config_path=cfg) == 6


def test_measured_overrides_baseline(tmp_path):
    root = tmp_path / "data"
    _write_costs(root, "ds", {"SC": 34, "MC": 1371, "C": 5899, "KW": 49, "SU": 5, "SJ": 1})
    cfg = _write_config(tmp_path, basis="measured", ds_root=root)
    assert cm.resolve_cost("SC", 4, config_path=cfg) == 34
    assert cm.resolve_cost("SJ", 1, config_path=cfg) == 1


def test_measured_missing_file_raises_with_hint(tmp_path):
    cfg = _write_config(tmp_path, basis="measured", ds_root=tmp_path / "data")
    with pytest.raises(FileNotFoundError, match="report --write"):
        cm.resolve_cost("SC", 4, config_path=cfg)


def test_measured_missing_key_raises(tmp_path):
    root = tmp_path / "data"
    _write_costs(root, "ds", {"SC": 34})
    cfg = _write_config(tmp_path, basis="measured", ds_root=root)
    with pytest.raises(KeyError, match="MC"):
        cm.resolve_cost("MC", 10, config_path=cfg)
