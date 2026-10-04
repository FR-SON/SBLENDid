"""Warnings when a fitted cost model is used at a different backend or ef_search."""
import warnings

import pytest

from src.Semantic.cost_predict import _reset_cache, warn_on_provenance_mismatch


def setup_function():
    _reset_cache()


def test_warns_on_backend_mismatch():
    model = {"SU": {"backend": "faiss", "ef_search": 64, "hnsw_per_col_ms": 1.0}}
    with pytest.warns(UserWarning, match="backend='pgvector'"):
        warn_on_provenance_mismatch(model, "SU", backend="pgvector", ef_search=64)


def test_warns_on_ef_mismatch():
    model = {"SU": {"backend": "faiss", "ef_search": 64}}
    with pytest.warns(UserWarning, match="ef_search"):
        warn_on_provenance_mismatch(model, "SU", backend="faiss", ef_search=500)


def test_warns_only_once_per_combination():
    model = {"SU": {"backend": "faiss", "ef_search": 64}}
    with pytest.warns(UserWarning):
        warn_on_provenance_mismatch(model, "SU", backend="pgvector", ef_search=500)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_on_provenance_mismatch(model, "SU", backend="pgvector", ef_search=500)


def test_silent_when_provenance_matches():
    model = {"SU": {"backend": "pgvector", "ef_search": 64}}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_on_provenance_mismatch(model, "SU", backend="pgvector", ef_search=64)


def test_silent_for_pre_provenance_models():
    """Models fitted before backend/ef were stamped must not start warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_on_provenance_mismatch({"SU": {"hnsw_per_col_ms": 1.0}}, "SU",
                                    backend="faiss", ef_search=64)


def test_silent_for_missing_op():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_on_provenance_mismatch({"SU": {"backend": "faiss", "ef_search": 64}},
                                    "SJ", backend="pgvector", ef_search=500)


def test_cost_unfiltered_warns_when_cfg_given(tmp_path):
    """Passing cfg surfaces the mismatch; omitting it stays silent."""
    import json
    from dataclasses import replace

    from src.Semantic.config import SemanticConfig
    from src.Semantic.cost_predict import cost_unfiltered

    opt = tmp_path / "optimizer"
    opt.mkdir()
    (opt / "semantic_cost_model.json").write_text(json.dumps({
        "SU": {"backend": "faiss", "ef_search": 64, "hnsw_per_col_ms": 2.0},
    }))
    cfg = replace(SemanticConfig(), vector_backend="pgvector",
                  faiss_hnsw_ef_search=64)

    with pytest.warns(UserWarning, match="backend='pgvector'"):
        ms = cost_unfiltered("SU", 3, 500, dataset_dir=tmp_path, cfg=cfg)
    assert ms == pytest.approx(6.0)

    _reset_cache()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert cost_unfiltered("SU", 3, 500, dataset_dir=tmp_path) == pytest.approx(6.0)
