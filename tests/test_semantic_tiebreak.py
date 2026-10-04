"""Unit tests for the semantic-seeker ms tiebreak (stub seeker, no index)."""
import json
from types import SimpleNamespace

import pandas as pd

from src.Semantic import cost_predict, depths
from src.Semantic.seekers import tiebreak


def _stub(dataset_dir, mode, n_cols, op="SU", k_coarse=500, k=10, vote_factor=2.0):
    """`k_coarse` is the seeker's pin; None derives the depth from `k`."""
    cfg = SimpleNamespace(
        semantic_ml_cost=mode,
        dataset=SimpleNamespace(dir=lambda: dataset_dir),
    )
    df = pd.DataFrame({f"c{i}": [0] for i in range(n_cols)})
    return SimpleNamespace(_cfg=cfg, SEMANTIC_OP=op, input=df, k=k,
                           _k_coarse=k_coarse, _vote_factor=vote_factor)


def test_tiebreak_off_returns_one(tmp_path):
    assert tiebreak.tiebreak_cost(_stub(tmp_path, "off", 3)) == 1.0


def test_tiebreak_analytic_returns_seconds_not_ms(tmp_path):
    cost_predict._reset_cache()
    (tmp_path / "optimizer").mkdir(parents=True)
    (tmp_path / "optimizer" / "semantic_cost_model.json").write_text(json.dumps({
        "SU": {"hnsw_per_col_ms": 2.0, "exact_beta_ms": 1.0,
               "exact_alpha_ms_per_col": 0.0, "flip_a_cols": 1000,
               "n_vectors": 1, "avg_cols_per_table": 1.0,
               "k_coarse": 500, "ef_search": 64, "backend": "faiss",
               "unfiltered": {"form": "su_bilinear",
                              "coef": {"a": 1.0, "b_ncols": 0.5,
                                       "c_kc": 0.002, "d": 0.0001},
                              "k_coarse_grid": [100, 1000], "ef_search": 64,
                              "r2": 1.0, "mae": 0.0, "n_train": 9, "n_test": 3}}
    }))
    seeker = _stub(tmp_path, "analytic", 3, k_coarse=1000)
    ms = cost_predict.cost_unfiltered("SU", 3, 1000, dataset_dir=tmp_path)
    assert tiebreak.tiebreak_cost(seeker) == ms / 1000.0 == 0.0048


def test_tiebreak_analytic_legacy_model_uses_per_col_fallback(tmp_path):
    cost_predict._reset_cache()
    (tmp_path / "optimizer").mkdir(parents=True)
    (tmp_path / "optimizer" / "semantic_cost_model.json").write_text(json.dumps({
        "SU": {"hnsw_per_col_ms": 2.0, "exact_beta_ms": 1.0,
               "exact_alpha_ms_per_col": 0.0, "flip_a_cols": 1000,
               "n_vectors": 1, "avg_cols_per_table": 1.0,
               "k_coarse": 500, "ef_search": 64, "backend": "faiss"}
    }))
    assert tiebreak.tiebreak_cost(_stub(tmp_path, "analytic", 3)) == 0.006


def test_tiebreak_analytic_is_k_coarse_aware(tmp_path):
    cost_predict._reset_cache()
    (tmp_path / "optimizer").mkdir(parents=True)
    (tmp_path / "optimizer" / "semantic_cost_model.json").write_text(json.dumps({
        "SU": {"hnsw_per_col_ms": 2.0, "exact_beta_ms": 1.0,
               "exact_alpha_ms_per_col": 0.0, "flip_a_cols": 1000,
               "n_vectors": 1, "avg_cols_per_table": 1.0,
               "k_coarse": 500, "ef_search": 64, "backend": "faiss",
               "unfiltered": {"form": "su_bilinear",
                              "coef": {"a": 1.0, "b_ncols": 0.5,
                                       "c_kc": 0.002, "d": 0.0001},
                              "k_coarse_grid": [100, 1000], "ef_search": 64,
                              "r2": 1.0, "mae": 0.0, "n_train": 9, "n_test": 3}}
    }))
    lo = tiebreak.tiebreak_cost(_stub(tmp_path, "analytic", 3, k_coarse=100))
    hi = tiebreak.tiebreak_cost(_stub(tmp_path, "analytic", 3, k_coarse=1000))
    assert hi > lo


def test_tiebreak_derives_the_depth_when_the_pin_is_absent(tmp_path):
    """An unpinned `_k_coarse` of None must not reach the model."""
    cost_predict._reset_cache()
    (tmp_path / "optimizer").mkdir(parents=True)
    (tmp_path / "optimizer" / "semantic_cost_model.json").write_text(json.dumps({
        "SU": {"hnsw_per_col_ms": 2.0, "exact_beta_ms": 1.0,
               "exact_alpha_ms_per_col": 0.0, "flip_a_cols": 1000,
               "n_vectors": 1, "avg_cols_per_table": 1.0,
               "k_coarse": 60, "ef_search": 64, "backend": "faiss",
               "unfiltered": {"form": "su_bilinear",
                              "coef": {"a": 0.0, "b_ncols": 0.0,
                                       "c_kc": 1.0, "d": 0.0},
                              "k_coarse_grid": [30, 60], "ef_search": 64,
                              "r2": 1.0, "mae": 0.0, "n_train": 9, "n_test": 3}}
    }))
    assert tiebreak.tiebreak_cost(_stub(tmp_path, "analytic", 3, k_coarse=None,
                                        k=10)) == 0.030
    assert tiebreak.tiebreak_cost(_stub(tmp_path, "analytic", 3, k_coarse=None,
                                        k=25)) == 0.060
    assert tiebreak.tiebreak_cost(_stub(tmp_path, "analytic", 3, k_coarse=500,
                                        k=25)) == 0.500


def test_effective_k_coarse_follows_the_plans_k():
    assert depths.effective_k_coarse(10) == 30
    assert depths.effective_k_coarse(25) == 60
    assert depths.effective_k_coarse(25, vote_factor=5.0) == 135
    assert depths.effective_k_coarse(25, pinned_k_coarse=500) == 500
