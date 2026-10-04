from pathlib import Path

import pytest

from src.Semantic.config import OperatorConfig, SemanticConfig, SemanticOp


def test_defaults_with_no_file(tmp_path):
    cfg = SemanticConfig.load(path=tmp_path / "missing.ini")
    assert cfg.operators == {}
    assert cfg.faiss_quant == "flat"
    assert cfg.faiss_k_coarse is None
    assert cfg.vote_depth_factor == 2.0
    assert cfg.restricted_k_coarse == 500
    assert cfg.dataset.name == "default"


def test_load_from_ini(tmp_path):
    ini = tmp_path / "config.ini"
    ini.write_text(
        "[Dataset]\n"
        "name = santos\n"
        "root = my-data\n"
        "[Semantic]\n"
        "faiss_quant = pq\n"
        "faiss_k_coarse = 100\n"
        "[Semantic.SU]\n"
        "approach = liftus\n"
        "index_name = default\n"
        "[Semantic.SJ]\n"
        "approach = snoopy\n"
        "index_name = v2\n"
    )
    cfg = SemanticConfig.load(path=ini)
    assert cfg.dataset.name == "santos"
    assert cfg.dataset.root.name == "my-data"
    assert cfg.faiss_quant == "pq"
    assert cfg.faiss_k_coarse == 100
    assert cfg.operator(SemanticOp.SU) == OperatorConfig("liftus", "default")
    assert cfg.operator(SemanticOp.SJ) == OperatorConfig("snoopy", "v2")


def test_overrides_win(tmp_path):
    cfg = SemanticConfig.load(
        path=tmp_path / "missing.ini",
        overrides={"faiss_k_coarse": 42, "device": "cuda"},
    )
    assert cfg.faiss_k_coarse == 42
    assert cfg.device == "cuda"


def test_dataset_overrides(tmp_path):
    cfg = SemanticConfig.load(
        path=tmp_path / "missing.ini",
        overrides={"dataset": "santos", "dataset_root": str(tmp_path / "data")},
    )
    assert cfg.dataset.name == "santos"
    assert cfg.dataset.root == tmp_path / "data"


def test_index_dir_layout(tmp_path):
    cfg = SemanticConfig.load(
        path=tmp_path / "missing.ini",
        overrides={"dataset": "santos", "dataset_root": str(tmp_path / "data")},
    )
    p = cfg.index_dir("liftus", "demo")
    assert p == tmp_path / "data" / "santos" / "semantic" / "liftus" / "demo" / "index"


def test_blend_basenames_path_property(tmp_path):
    cfg = SemanticConfig.load(
        path=tmp_path / "missing.ini",
        overrides={"dataset": "santos", "dataset_root": str(tmp_path / "data")},
    )
    assert cfg.blend_basenames_path == tmp_path / "data" / "santos" / "blend_index_basenames.parquet"


def test_signature_stable():
    cfg = SemanticConfig.load(path=Path("/nonexistent"))
    assert cfg.signature() == SemanticConfig.load(path=Path("/nonexistent")).signature()


def test_signature_differs_per_dataset(tmp_path):
    a = SemanticConfig.load(
        path=tmp_path / "missing.ini",
        overrides={"dataset": "santos"},
    )
    b = SemanticConfig.load(
        path=tmp_path / "missing.ini",
        overrides={"dataset": "dev"},
    )
    assert a.signature() != b.signature()


def test_operator_missing_raises_with_hint(tmp_path):
    cfg = SemanticConfig.load(path=tmp_path / "missing.ini")
    with pytest.raises(KeyError, match=r"\[Semantic\.SU\]"):
        cfg.operator(SemanticOp.SU)


def test_operators_kwarg_overrides_ini(tmp_path):
    ini = tmp_path / "config.ini"
    ini.write_text(
        "[Semantic.SU]\n"
        "approach = liftus\n"
        "index_name = default\n"
    )
    cfg = SemanticConfig.load(
        path=ini,
        operators={SemanticOp.SU: OperatorConfig("snoopy", "v2")},
    )
    assert cfg.operator(SemanticOp.SU) == OperatorConfig("snoopy", "v2")


def test_operators_kwarg_merges_with_ini(tmp_path):
    ini = tmp_path / "config.ini"
    ini.write_text(
        "[Semantic.SU]\n"
        "approach = liftus\n"
        "index_name = default\n"
    )
    cfg = SemanticConfig.load(
        path=ini,
        operators={SemanticOp.SJ: OperatorConfig("snoopy", "default")},
    )
    assert cfg.operator(SemanticOp.SU) == OperatorConfig("liftus", "default")
    assert cfg.operator(SemanticOp.SJ) == OperatorConfig("snoopy", "default")


def test_signature_differs_per_operators_config(tmp_path):
    a = SemanticConfig.load(
        path=tmp_path / "missing.ini",
        operators={SemanticOp.SU: OperatorConfig("liftus", "default")},
    )
    b = SemanticConfig.load(
        path=tmp_path / "missing.ini",
        operators={SemanticOp.SU: OperatorConfig("liftus", "v2")},
    )
    assert a.signature() != b.signature()


def test_ini_missing_required_op_key_raises(tmp_path):
    ini = tmp_path / "config.ini"
    ini.write_text(
        "[Semantic.SU]\n"
        "approach = liftus\n"
    )
    with pytest.raises(KeyError):
        SemanticConfig.load(path=ini)


def test_vote_factor_for_precedence(tmp_path):
    ini = tmp_path / "config.ini"
    ini.write_text(
        "[Semantic]\n"
        "vote_depth_factor = 3.0\n"
        "[Semantic.SU]\n"
        "approach = liftus\n"
        "index_name = default\n"
        "vote_depth_factor = 5.0\n"
        "[Semantic.SJ]\n"
        "approach = snoopy\n"
        "index_name = default\n"
    )
    cfg = SemanticConfig.load(path=ini)
    assert cfg.vote_factor_for(SemanticOp.SU) == 5.0
    assert cfg.vote_factor_for(SemanticOp.SJ) == 3.0
    assert cfg.vote_factor_for(SemanticOp.SU, override=1.5) == 1.5


def test_vote_factor_for_unconfigured_op_falls_back(tmp_path):
    cfg = SemanticConfig.load(path=tmp_path / "missing.ini")
    assert cfg.operators == {}
    assert cfg.vote_factor_for(SemanticOp.SJ) == 2.0


@pytest.mark.parametrize("bad", ["0", "0.0", "-1.5"])
def test_vote_factor_rejects_non_positive(tmp_path, bad):
    ini = tmp_path / "config.ini"
    ini.write_text(
        "[Semantic.SU]\n"
        "approach = liftus\n"
        "index_name = default\n"
        f"vote_depth_factor = {bad}\n"
    )
    cfg = SemanticConfig.load(path=ini)
    with pytest.raises(ValueError, match="vote_depth_factor must be > 0"):
        cfg.vote_factor_for(SemanticOp.SU)
    with pytest.raises(ValueError, match="vote_depth_factor must be > 0"):
        cfg.vote_factor_for(SemanticOp.SJ, override=0)


def test_query_encode_default_off(tmp_path):
    assert SemanticConfig.load(path=tmp_path / "missing.ini").query_encode == "off"


def test_query_encode_from_ini_and_override(tmp_path):
    ini = tmp_path / "config.ini"
    ini.write_text("[Semantic]\nquery_encode = off\n")
    assert SemanticConfig.load(path=ini).query_encode == "off"
    assert SemanticConfig.load(path=ini, overrides={"query_encode": "auto"}).query_encode == "auto"


def test_query_encode_rejects_unknown_value(tmp_path):
    ini = tmp_path / "config.ini"
    ini.write_text("[Semantic]\nquery_encode = maybe\n")
    with pytest.raises(ValueError, match="query_encode"):
        SemanticConfig.load(path=ini)


def test_query_encode_not_in_signature(tmp_path):
    a = SemanticConfig.load(path=tmp_path / "missing.ini", overrides={"query_encode": "auto"})
    b = SemanticConfig.load(path=tmp_path / "missing.ini", overrides={"query_encode": "off"})
    assert a.signature() == b.signature()
