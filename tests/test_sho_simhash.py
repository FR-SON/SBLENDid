import numpy as np

from src.Semantic.simhash import (
    SHO_ENCODER_MODEL, clean_cell, codes_from_embeddings, get_hyperplanes,
    is_numeric_column, sample_row_indices, table_seed,
)


def test_model_name_matches_semdisc():
    assert SHO_ENCODER_MODEL == "paraphrase-distilroberta-base-v1"


def test_hyperplanes_seeded_and_shaped():
    a = get_hyperplanes(18, 768, seed=0)
    b = get_hyperplanes(18, 768, seed=0)
    assert a.shape == (18, 768)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, get_hyperplanes(18, 768, seed=1))


def test_codes_match_smoke_test_bitstring_path():
    # SemDisc reference implementation: codes must match for same seed+bits
    def ref_get_hyperplanes(b, dim, seed):
        return np.random.default_rng(seed).standard_normal((b, dim))

    def ref_simhash_codes(embeddings, hyperplanes):
        dot = np.dot(embeddings, hyperplanes.T)
        bits = (dot > 0).astype(int)
        return [''.join(map(str, map(int, row))) for row in bits]

    rng = np.random.default_rng(42)
    emb = rng.standard_normal((5, 16)).astype(np.float32)
    hp = get_hyperplanes(6, 16, seed=0)
    np.testing.assert_array_equal(hp, ref_get_hyperplanes(6, 16, seed=0))
    expected = [int(s, 2) for s in ref_simhash_codes(emb, hp)]
    assert codes_from_embeddings(emb, hp).tolist() == expected


def test_sample_row_indices_identity_and_seeded():
    np.testing.assert_array_equal(sample_row_indices(5, 10, seed=1), np.arange(5))
    s1 = sample_row_indices(100, 10, seed=7)
    s2 = sample_row_indices(100, 10, seed=7)
    np.testing.assert_array_equal(s1, s2)
    assert len(s1) == 10 and len(set(s1.tolist())) == 10
    assert list(s1) == sorted(s1)


def test_table_seed_deterministic_and_distinct():
    assert table_seed(0, "a.csv") == table_seed(0, "a.csv")
    assert table_seed(0, "a.csv") != table_seed(0, "b.csv")
    assert table_seed(0, "a.csv") != table_seed(1, "a.csv")


def test_is_numeric_column_semdisc_rule():
    assert is_numeric_column(["123", "456", "789"]) is True
    assert is_numeric_column(["alpha", "beta", "gamma"]) is False
    assert is_numeric_column([]) is False


def test_clean_cell():
    assert clean_cell("foo") == "foo"
    assert clean_cell("") is None
    assert clean_cell("nan") is None
    assert clean_cell("NaN") is None
    assert clean_cell(7) == "7"
