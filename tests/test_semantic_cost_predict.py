import json

import pytest

from src.Semantic import cost_predict


def _write_model(dataset_dir):
    (dataset_dir / "optimizer").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "optimizer" / "semantic_cost_model.json").write_text(json.dumps({
        "SU": {
            "hnsw_per_col_ms": 2.0,
            "exact_beta_ms": 1.0,
            "exact_alpha_ms_per_col": 0.001,
            "flip_a_cols": 10000,
            "n_vectors": 6322, "avg_cols_per_table": 10.8,
            "k_coarse": 500, "ef_search": 64, "backend": "pgvector",
        }
    }))


def _write_model_with_unfiltered(dataset_dir):
    (dataset_dir / "optimizer").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "optimizer" / "semantic_cost_model.json").write_text(json.dumps({
        "SU": {
            "hnsw_per_col_ms": 2.0, "exact_beta_ms": 1.0,
            "exact_alpha_ms_per_col": 0.001, "flip_a_cols": 10000,
            "n_vectors": 6322, "avg_cols_per_table": 10.8,
            "k_coarse": 500, "ef_search": 64, "backend": "pgvector",
            "unfiltered": {
                "form": "su_bilinear",
                "coef": {"a": 1.0, "b_ncols": 0.5, "c_kc": 0.002, "d": 0.0001},
                "k_coarse_grid": [100, 500, 1000], "ef_search": 64,
                "r2": 1.0, "mae": 0.0, "n_train": 9, "n_test": 3,
            },
        },
        "SJ": {
            "hnsw_per_col_ms": 4.0, "exact_beta_ms": 2.0,
            "exact_alpha_ms_per_col": 0.001, "flip_a_cols": None,
            "n_vectors": 6322, "avg_cols_per_table": 10.8,
            "k_coarse": 500, "ef_search": 64, "backend": "pgvector",
            "unfiltered": {
                "form": "sj_linear_kc", "coef": {"a": 2.0, "c_kc": 0.003},
                "k_coarse_grid": [100, 500, 1000], "ef_search": 64,
                "r2": 1.0, "mae": 0.0, "n_train": 4, "n_test": 2,
            },
        },
    }))


def test_unfiltered_uses_bilinear_coef_with_k_coarse(tmp_path):
    cost_predict._reset_cache()
    _write_model_with_unfiltered(tmp_path)
    assert cost_predict.cost_unfiltered("SU", 3, 1000, dataset_dir=tmp_path) == 4.8


def test_unfiltered_su_is_k_coarse_sensitive(tmp_path):
    cost_predict._reset_cache()
    _write_model_with_unfiltered(tmp_path)
    lo = cost_predict.cost_unfiltered("SU", 3, 100, dataset_dir=tmp_path)
    hi = cost_predict.cost_unfiltered("SU", 3, 1000, dataset_dir=tmp_path)
    assert hi > lo


def test_unfiltered_sj_ignores_n_cols(tmp_path):
    cost_predict._reset_cache()
    _write_model_with_unfiltered(tmp_path)
    assert cost_predict.cost_unfiltered("SJ", 1, 500, dataset_dir=tmp_path) == 3.5
    assert cost_predict.cost_unfiltered("SJ", 9, 500, dataset_dir=tmp_path) == 3.5


def test_unfiltered_legacy_fallback_when_no_unfiltered_key(tmp_path):
    cost_predict._reset_cache()
    _write_model(tmp_path)
    assert cost_predict.cost_unfiltered("SU", 3, 1000, dataset_dir=tmp_path) == 6.0


def test_filtered_below_flip_uses_exact_line(tmp_path):
    cost_predict._reset_cache()
    _write_model(tmp_path)
    assert cost_predict.predict_filtered_ms("SU", 2, 5000, dataset_dir=tmp_path) == 12.0


def test_filtered_above_flip_uses_hnsw_proxy(tmp_path):
    cost_predict._reset_cache()
    _write_model(tmp_path)
    assert cost_predict.predict_filtered_ms("SU", 2, 20000, dataset_dir=tmp_path) == 4.0


def test_filtered_exact_threshold_overrides_model_flip(tmp_path):
    cost_predict._reset_cache()
    _write_model(tmp_path)
    assert cost_predict.predict_filtered_ms(
        "SU", 2, 5000, dataset_dir=tmp_path, exact_threshold=4000) == 4.0


def test_missing_model_raises(tmp_path):
    cost_predict._reset_cache()
    with pytest.raises(FileNotFoundError):
        cost_predict.cost_unfiltered("SU", 1, 500, dataset_dir=tmp_path)
