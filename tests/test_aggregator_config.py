import pytest

from src.Semantic.aggregators import build_op_aggregator
from src.Semantic.aggregators.decay_vote import DecayVoteAggregator
from src.Semantic.aggregators.max_pool import MaxPoolAggregator
from src.Semantic.aggregators.munkres import MunkresAggregator
from src.Semantic.config import OperatorConfig, SemanticConfig, SemanticOp


def test_operator_config_defaults():
    oc = OperatorConfig(approach="liftus", index_name="default")
    assert oc.aggregator == "default"
    assert oc.munkres_threshold is None


def test_config_parses_aggregator_keys(tmp_path):
    ini = tmp_path / "config.ini"
    ini.write_text(
        "[Dataset]\nname = x\nroot = /tmp\n\n"
        "[Semantic.SU]\napproach = deepjoin\nindex_name = default\n"
        "aggregator = munkres\nmunkres_threshold = 0.3\n"
    )
    cfg = SemanticConfig.load(path=ini)
    oc = cfg.operator(SemanticOp.SU)
    assert oc.aggregator == "munkres"
    assert oc.munkres_threshold == pytest.approx(0.3)


def test_build_op_aggregator_default_passthrough():
    default = DecayVoteAggregator()
    oc = OperatorConfig(approach="a", index_name="i")
    assert build_op_aggregator(oc, default) is default


def test_build_op_aggregator_decay_vote_explicit():
    oc = OperatorConfig(approach="a", index_name="i", aggregator="decay_vote")
    agg = build_op_aggregator(oc, DecayVoteAggregator())
    assert isinstance(agg, DecayVoteAggregator)


def test_build_op_aggregator_max_pool():
    oc = OperatorConfig(approach="a", index_name="i", aggregator="max_pool")
    agg = build_op_aggregator(oc, DecayVoteAggregator())
    assert isinstance(agg, MaxPoolAggregator)
    assert agg.requires_ctx is False


def test_build_op_aggregator_munkres():
    oc = OperatorConfig(approach="a", index_name="i", aggregator="munkres",
                        munkres_threshold=0.3)
    agg = build_op_aggregator(oc, DecayVoteAggregator())
    assert isinstance(agg, MunkresAggregator)
    assert agg.threshold == pytest.approx(0.3)


def test_build_op_aggregator_munkres_without_threshold():
    oc = OperatorConfig(approach="a", index_name="i", aggregator="munkres")
    with pytest.raises(ValueError, match="munkres_threshold"):
        build_op_aggregator(oc, DecayVoteAggregator())


def test_build_op_aggregator_unknown():
    oc = OperatorConfig(approach="a", index_name="i", aggregator="wat")
    with pytest.raises(ValueError, match="wat"):
        build_op_aggregator(oc, DecayVoteAggregator())
