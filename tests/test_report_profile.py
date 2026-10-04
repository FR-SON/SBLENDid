import json
import textwrap
from pathlib import Path

import pandas as pd

import src.cost_model as cm
from src.Optimizer.report import write_profile


def _rep():
    surface = pd.DataFrame(
        {100: [84, 136], 500: [136, 1035]}, index=[16, 64]
    )
    surface.index.name = "efSearch"
    surface.columns.name = "k_coarse"
    return {
        "median_s": {"SC": 0.034, "Keyword": 0.049, "SU": 0.005, "SJ": 0.001},
        "counts": {"SC": 50, "Keyword": 50, "SU": 200, "SJ": 1000},
        "basis": {"SC": "unfiltered SQL (full index)"},
        "cost_int": {"SC": 34, "Keyword": 49, "SU": 5, "SJ": 1},
        "order": ["SJ", "SU", "SC", "Keyword"],
        "semantic": {"SU": {"hnsw_ms": 2.221, "exact_threshold_gids": 1035}},
        "sensitivity": {"SU": surface},
        "crossover_surface": {"SU": surface},
        "ef_default": 64,
        "kc_default": 500,
    }


def _config(tmp_path, root, name="ds"):
    cfg = tmp_path / "config.ini"
    cfg.write_text(textwrap.dedent(f"""\
        [Dataset]
        name = {name}
        root = {root}

        [Database]
        dbms = duckdb
        db_filename = blend.duckdb

        [Optimizer]
        cost_basis = measured
    """))
    return cfg


def test_write_profile_normalizes_keys_and_is_lossless(tmp_path):
    root = tmp_path / "data"
    cfg = _config(tmp_path, root)
    dest = write_profile(_rep(), "ds", "runs/optimizer/ds", config_path=cfg)
    data = json.loads(Path(dest).read_text())

    assert data["costs"] == {"SC": 34, "KW": 49, "SU": 5, "SJ": 1}
    assert data["report"]["order"] == ["SJ", "SU", "SC", "KW"]
    assert "Keyword" not in data["report"]["cost_int"]

    cs = data["report"]["crossover_surface"]["SU"]
    assert cs["64"]["500"] == 1035
    assert data["report"]["sensitivity"]["SU"]["16"]["100"] == 84

    assert set(data["report"].keys()) == set(_rep().keys())

    assert data["metadata"]["dataset"] == "ds"
    assert data["metadata"]["source_run_dir"] == "runs/optimizer/ds"


def test_resolve_cost_reads_written_profile(tmp_path):
    root = tmp_path / "data"
    cfg = _config(tmp_path, root)
    write_profile(_rep(), "ds", "runs/optimizer/ds", config_path=cfg)
    cm._reset_cache()
    assert cm.resolve_cost("KW", 3, config_path=cfg) == 49
    assert cm.resolve_cost("SU", 2, config_path=cfg) == 5
