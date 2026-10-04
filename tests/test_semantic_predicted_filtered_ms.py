import json
from types import SimpleNamespace

import pandas as pd

# bootstrap Seekers first to avoid a circular import via SemanticUnion
from src.Operators import Seekers  # noqa: F401

from src.Semantic import cost_predict
from src.Semantic.seekers.union_base import SemanticUnionSeekerBase


def test_predicted_filtered_ms_uses_explicit_a_cols(tmp_path):
    cost_predict._reset_cache()
    (tmp_path / "optimizer").mkdir(parents=True)
    (tmp_path / "optimizer" / "semantic_cost_model.json").write_text(json.dumps({
        "SU": {"hnsw_per_col_ms": 2.0, "exact_beta_ms": 1.0,
               "exact_alpha_ms_per_col": 0.001, "flip_a_cols": 10000,
               "n_vectors": 1, "avg_cols_per_table": 1.0,
               "k_coarse": 500, "ef_search": 64, "backend": "pgvector"}
    }))
    seeker = SimpleNamespace(
        SEMANTIC_OP="SU",
        input=pd.DataFrame({"a": [1], "b": [2]}),
        _exact_threshold=None,
        _cfg=SimpleNamespace(
            dataset=SimpleNamespace(dir=lambda: tmp_path),
            vector_backend="pgvector",
        ),
    )
    assert SemanticUnionSeekerBase.predicted_filtered_ms(seeker, 5000) == 12.0


def test_predicted_filtered_ms_faiss_uses_config_exact_threshold(tmp_path):
    cost_predict._reset_cache()
    (tmp_path / "optimizer").mkdir(parents=True)
    (tmp_path / "optimizer" / "semantic_cost_model.json").write_text(json.dumps({
        "SU": {"hnsw_per_col_ms": 2.0, "exact_beta_ms": 1.0,
               "exact_alpha_ms_per_col": 0.001, "flip_a_cols": None,
               "n_vectors": 1, "avg_cols_per_table": 1.0,
               "k_coarse": 500, "ef_search": 64, "backend": "faiss"}
    }))
    seeker = SimpleNamespace(
        SEMANTIC_OP="SU",
        input=pd.DataFrame({"a": [1], "b": [2]}),
        _exact_threshold=4000,
        _cfg=SimpleNamespace(
            dataset=SimpleNamespace(dir=lambda: tmp_path),
            vector_backend="faiss",
        ),
    )
    assert SemanticUnionSeekerBase.predicted_filtered_ms(seeker, 5000) == 4.0
