from dataclasses import dataclass

import pytest

from src.Benchmark.correctness.theorem1 import fidelity as F


class _Db:
    USE_ML_OPTIMIZER = False


class _DbMl:
    USE_ML_OPTIMIZER = True


def test_guard_gate(monkeypatch):
    monkeypatch.setattr("src.cost_model.pushdown_guard", lambda **k: "A")
    with pytest.raises(F.FidelityError, match="pushdown_guard"):
        F.check_startup(_Db())


def test_cost_basis_gate(monkeypatch):
    monkeypatch.setattr("src.cost_model.pushdown_guard", lambda **k: "none")
    monkeypatch.setattr("src.cost_model.cost_basis", lambda **k: "measured")
    with pytest.raises(F.FidelityError, match="cost_basis"):
        F.check_startup(_Db())
    rec = F.check_startup(_Db(), allow_measured_costs=True)
    assert rec["cost_basis"] == "measured" and rec["allow_measured_costs"]


def test_ml_optimizer_gate(monkeypatch):
    monkeypatch.setattr("src.cost_model.pushdown_guard", lambda **k: "none")
    monkeypatch.setattr("src.cost_model.cost_basis", lambda **k: "logical")
    with pytest.raises(F.FidelityError, match="USE_ML_OPTIMIZER"):
        F.check_startup(_DbMl())
    assert F.check_startup(_Db())["pushdown_guard"] == "none"


@dataclass
class _Cfg:
    vector_backend: str = "faiss"
    faiss_k_coarse: int | None = 500
    faiss_hnsw_ef_search: int = 64
    exact_threshold: int | None = 1035


def test_faiss_pinned_ef_below_k_coarse_refused():
    with pytest.raises(F.FidelityError, match="ef"):
        F.resolve_semantic_knobs(_Cfg(), cli_k_coarse=None, cli_ef=None,
                                 cli_exact_threshold=None)


def test_cli_k_coarse_pulls_ef_along():
    r = F.resolve_semantic_knobs(_Cfg(), cli_k_coarse=500, cli_ef=None,
                                 cli_exact_threshold=None)
    assert (r["k_coarse"], r["k_coarse_source"]) == (500, "cli")
    assert (r["ef_search"], r["ef_search_source"]) == (500, "derived")
    assert (r["exact_threshold"], r["exact_threshold_source"]) == (1035, "config")


def test_faiss_cli_ef_mismatch_refused():
    with pytest.raises(F.FidelityError):
        F.resolve_semantic_knobs(_Cfg(), cli_k_coarse=500, cli_ef=1000,
                                 cli_exact_threshold=None)


def test_pgvector_ef_unconstrained():
    r = F.resolve_semantic_knobs(_Cfg(vector_backend="pgvector"),
                                 cli_k_coarse=500, cli_ef=64,
                                 cli_exact_threshold=None)
    assert r["ef_search"] == 64


def test_reference_knobs():
    assert F.reference_knobs(6314) == {"exact_threshold": 6314, "k_coarse": 6314}
    with pytest.raises(F.FidelityError, match="reference"):
        F.reference_knobs(6314, exact_threshold=1035)


def test_forced_exact_preserves_unfiltered_hnsw(duckdb_unit_env):
    import src.Semantic.retrieve as R
    orig = R.should_use_exact
    with F.forced_exact():
        assert R.should_use_exact(None, 10, vector_backend="faiss") is False
        assert R.should_use_exact(None, 10, vector_backend="pgvector") is False
        assert R.should_use_exact(5, None, vector_backend="pgvector") is True
    assert R.should_use_exact is orig


def test_forced_exact_restores_on_exception(duckdb_unit_env):
    import src.Semantic.retrieve as R
    orig = R.should_use_exact
    with pytest.raises(RuntimeError):
        with F.forced_exact():
            raise RuntimeError("boom")
    assert R.should_use_exact is orig


def test_assert_clean_ids():
    F.assert_clean_ids([1, 2, 3], where="x")
    with pytest.raises(F.FidelityError):
        F.assert_clean_ids([1, -1], where="x")
    with pytest.raises(F.FidelityError):
        F.assert_clean_ids([1, None], where="x")
