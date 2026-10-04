from src.Semantic.config import SemanticConfig


def test_default_off():
    assert SemanticConfig().semantic_ml_cost == "off"


def test_override_analytic():
    cfg = SemanticConfig.load(overrides={"semantic_ml_cost": "analytic"})
    assert cfg.semantic_ml_cost == "analytic"
