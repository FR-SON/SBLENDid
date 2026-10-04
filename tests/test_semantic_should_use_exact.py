from src.Semantic.retrieve import should_use_exact


def test_faiss_dispatches_exact_below_threshold():
    assert should_use_exact(10, 1000) is True
    assert should_use_exact(10, 1000, vector_backend="faiss") is True


def test_above_threshold_is_false():
    assert should_use_exact(2000, 1000, vector_backend="faiss") is False


def test_pgvector_never_dispatches_exact_even_with_override():
    assert should_use_exact(10, 1000, vector_backend="pgvector") is False
    assert should_use_exact(1, 10**9, vector_backend="pgvector") is False
