import warnings, pytest
from dataclasses import replace
from src.Semantic.config import SemanticConfig
from src.Semantic.retrieve import _warn_if_ef_below_k, _WARNED_EF

def setup_function():
    _WARNED_EF.clear()

def test_warns_on_faiss_when_ef_below_k():
    cfg = replace(SemanticConfig(), vector_backend="faiss",
                  faiss_hnsw_ef_search=64, faiss_k_coarse=500)
    with pytest.warns(UserWarning, match="cannot return 500"):
        _warn_if_ef_below_k(cfg)

def test_warns_once_per_pair():
    cfg = replace(SemanticConfig(), vector_backend="faiss",
                  faiss_hnsw_ef_search=64, faiss_k_coarse=500)
    with pytest.warns(UserWarning):
        _warn_if_ef_below_k(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_ef_below_k(cfg)

def test_silent_on_faiss_when_ef_equals_k():
    cfg = replace(SemanticConfig(), vector_backend="faiss",
                  faiss_hnsw_ef_search=500, faiss_k_coarse=500)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_ef_below_k(cfg)

def test_silent_on_pgvector_even_when_ef_below_k():
    """pgvector's iterative scan fills LIMIT regardless of ef."""
    cfg = replace(SemanticConfig(), vector_backend="pgvector",
                  faiss_hnsw_ef_search=64, faiss_k_coarse=500)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_ef_below_k(cfg)
