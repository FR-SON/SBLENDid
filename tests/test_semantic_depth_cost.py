"""cost() and the ml tiebreak are priced at the leg's own retrieval depth."""
import json

import pytest

from src.Semantic.seekers import cost as seeker_cost


class _Cfg:
    semantic_ml_cost = "analytic"
    vector_backend = "faiss"
    faiss_hnsw_ef_search = 64

    def __init__(self, dataset_dir):
        self.dataset = type("D", (), {"dir": lambda _s=None: dataset_dir})()


class _Input:
    def __init__(self, n_cols):
        self.shape = (1, n_cols)


class _Seeker:
    DB = None

    def __init__(self, dataset_dir, *, k, n_cols=4, pin=None, vote_factor=2.0):
        self.k = k
        self.input = _Input(n_cols)
        self._cfg = _Cfg(dataset_dir)
        self._k_coarse = pin
        self._vote_factor = vote_factor
        self.SEMANTIC_OP = "SU"


def _profile(tmp_path, *, su_median_s, cheapest_s):
    """A costs.json + semantic_cost_model.json pair on one dataset dir."""
    opt = tmp_path / "optimizer"
    opt.mkdir(parents=True, exist_ok=True)
    (opt / "costs.json").write_text(json.dumps({
        "costs": {"SC": 1, "SU": round(su_median_s / cheapest_s)},
        "report": {"median_s": {"SC": cheapest_s, "SU": su_median_s}},
    }))
    (opt / "semantic_cost_model.json").write_text(json.dumps({
        "SU": {"backend": "faiss", "ef_search": 64, "k_coarse": 60,
               "hnsw_per_col_ms": 5.0,
               "unfiltered": {"form": "su_bilinear",
                              "coef": {"a": 0.0, "b_ncols": 10.0,
                                       "c_kc": 1.0, "d": 0.0}}},
    }))
    return tmp_path


@pytest.fixture(autouse=True)
def _measured(monkeypatch):
    from src import cost_model
    from src.Semantic import cost_predict
    cost_model._reset_cache()
    cost_predict._reset_cache()
    monkeypatch.setattr(seeker_cost, "cost_basis", lambda **kw: "measured")
    yield
    cost_model._reset_cache()
    cost_predict._reset_cache()


def test_cost_scales_with_the_legs_depth(tmp_path):
    d = _profile(tmp_path, su_median_s=0.100, cheapest_s=0.010)
    assert seeker_cost.measured_cost(_Seeker(d, k=10), "SU", 2) == 7
    assert seeker_cost.measured_cost(_Seeker(d, k=25), "SU", 2) == 10
    assert seeker_cost.measured_cost(_Seeker(d, k=150), "SU", 2) == 35


def test_cost_at_the_anchor_depth_matches_the_stored_integer(tmp_path):
    """k=25 -> kc=60 is the anchor depth, so cost must equal the stored integer."""
    d = _profile(tmp_path, su_median_s=0.100, cheapest_s=0.010)
    stored = json.loads((d / "optimizer" / "costs.json").read_text())["costs"]["SU"]
    assert seeker_cost.measured_cost(_Seeker(d, k=25), "SU", 2) == stored


def test_cost_uses_the_pin_when_one_is_set(tmp_path):
    d = _profile(tmp_path, su_median_s=0.100, cheapest_s=0.010)
    pinned = seeker_cost.measured_cost(_Seeker(d, k=10, pin=500), "SU", 2)
    assert pinned == 54
    assert pinned != seeker_cost.measured_cost(_Seeker(d, k=10), "SU", 2)


def test_cost_scales_with_query_width(tmp_path):
    d = _profile(tmp_path, su_median_s=0.100, cheapest_s=0.010)
    narrow = seeker_cost.measured_cost(_Seeker(d, k=25, n_cols=1), "SU", 2)
    wide = seeker_cost.measured_cost(_Seeker(d, k=25, n_cols=40), "SU", 2)
    assert narrow < wide


def test_falls_back_to_the_stored_integer_without_a_model(tmp_path, monkeypatch):
    d = _profile(tmp_path, su_median_s=0.100, cheapest_s=0.010)
    (d / "optimizer" / "semantic_cost_model.json").unlink()
    seen = []
    monkeypatch.setattr(seeker_cost, "resolve_cost",
                        lambda op, base, **kw: seen.append((op, base)) or 10)
    assert seeker_cost.measured_cost(_Seeker(d, k=25), "SU", 2) == 10
    assert seen == [("SU", 2)]


def test_falls_back_when_the_profile_has_no_medians(tmp_path, monkeypatch):
    d = _profile(tmp_path, su_median_s=0.100, cheapest_s=0.010)
    (d / "optimizer" / "costs.json").write_text(json.dumps({"costs": {"SU": 10}}))
    monkeypatch.setattr(seeker_cost, "resolve_cost", lambda op, base, **kw: 10)
    assert seeker_cost.measured_cost(_Seeker(d, k=25), "SU", 2) == 10


def test_logical_basis_returns_the_baseline(tmp_path, monkeypatch):
    d = _profile(tmp_path, su_median_s=0.100, cheapest_s=0.010)
    monkeypatch.setattr(seeker_cost, "cost_basis", lambda **kw: "logical")
    assert seeker_cost.measured_cost(_Seeker(d, k=25), "SU", 2) == 2


def _ini(dataset_dir):
    """A config.ini whose [Dataset] resolves to `dataset_dir`."""
    ini = dataset_dir.parent / "cfg.ini"
    ini.write_text(
        "[Dataset]\n"
        f"name = {dataset_dir.name}\n"
        f"root = {dataset_dir.parent}\n"
        "[Optimizer]\ncost_basis = measured\n"
    )
    return ini


def test_anchor_is_none_without_a_profile(tmp_path):
    from src.cost_model import cost_scale_anchor_s
    assert cost_scale_anchor_s(config_path=_ini(tmp_path / "nope")) is None
