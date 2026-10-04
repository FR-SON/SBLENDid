import pytest

from src.Semantic.config import OperatorConfig, SemanticConfig, SemanticOp


def _write_cfg(tmp_path, extra=""):
    p = tmp_path / "config.ini"
    p.write_text(
        "[Dataset]\nname = fixture\nroot = /tmp/ds\n\n"
        "[Semantic.SU]\napproach = liftus\nindex_name = default\n\n" + extra
    )
    return p


def test_sho_op_exists_and_section_parses(tmp_path):
    p = _write_cfg(tmp_path, "[Semantic.SHO]\napproach = simhash\nindex_name = default\n")
    cfg = SemanticConfig.load(path=p)
    oc = cfg.operator(SemanticOp.SHO)
    assert oc == OperatorConfig(approach="simhash", index_name="default")
    assert cfg.approach_dir("simhash", "default").parts[-3:] == ("semantic", "simhash", "default")


def test_missing_sho_section_raises_with_hint(tmp_path):
    cfg = SemanticConfig.load(path=_write_cfg(tmp_path))
    with pytest.raises(KeyError, match=r"Semantic\.SHO"):
        cfg.operator(SemanticOp.SHO)


def test_signature_changes_with_sho_config(tmp_path):
    base = SemanticConfig.load(path=_write_cfg(tmp_path))
    with_sho = SemanticConfig.load(
        path=_write_cfg(tmp_path),
        operators={SemanticOp.SHO: OperatorConfig("simhash", "default")},
    )
    assert base.signature() != with_sho.signature()
