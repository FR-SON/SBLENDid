from pathlib import Path

import pytest

from src.Semantic.approaches import ApproachPlugin, get


FIXTURE = Path(__file__).parent / "fixtures" / "semantic"


def test_liftus_plugin_registered_on_import():
    import src.Semantic  # noqa: F401
    plug = get("liftus")
    assert isinstance(plug, ApproachPlugin)
    assert plug.name == "liftus"


def test_liftus_plugin_loads_artifacts_from_fixture():
    plug = get("liftus")
    bundle = plug.load_artifacts(FIXTURE)
    assert bundle.ckpt_path.is_file()
    assert bundle.sidecar["hidden_size"] == 128
    assert bundle.extras["dataset"] == "opendata"
    assert (bundle.extras["aspects_dir"] / "statistic").is_dir()


def test_liftus_plugin_builds_encoder_and_aggregator():
    from src.Semantic.config import SemanticConfig
    plug = get("liftus")
    bundle = plug.load_artifacts(FIXTURE)
    enc = plug.build_encoder(bundle, SemanticConfig())
    assert enc.dim == 128
    assert enc.name == "liftus"
    agg = plug.build_aggregator()
    assert agg.name == "decay_vote"


def test_get_unknown_raises():
    with pytest.raises(KeyError):
        get("not_a_real_approach")
