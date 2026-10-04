import numpy as np
import pytest

from src.Semantic.emb_store import (
    MissingEmbeddings, PersistentEmbeddingStore, resolve_embeddings,
)


def _vec(seed, dim=8):
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_store_roundtrip_and_reload(tmp_path):
    store = PersistentEmbeddingStore(tmp_path / "s", model="m")
    n = store.append({"a": _vec(1), "b": _vec(2)})
    assert n == 2 and store.count == 2 and store.dim == 8
    assert "a" in store and "zz" not in store
    np.testing.assert_allclose(store.get("b"), _vec(2), rtol=1e-6)
    store2 = PersistentEmbeddingStore(tmp_path / "s", model="m")
    assert store2.count == 2
    np.testing.assert_allclose(store2.get("a"), _vec(1), rtol=1e-6)


def test_store_model_mismatch_raises(tmp_path):
    PersistentEmbeddingStore(tmp_path / "s", model="m").append({"a": _vec(1)})
    with pytest.raises(ValueError, match="model mismatch"):
        PersistentEmbeddingStore(tmp_path / "s", model="other")


def test_resolve_strict_raises_on_miss(tmp_path):
    store = PersistentEmbeddingStore(tmp_path / "s", model="m")
    store.append({"a": _vec(1)})
    with pytest.raises(MissingEmbeddings) as ei:
        resolve_embeddings(["a", "b", "c"], store, encode_fn=None)
    assert set(ei.value.values) == {"b", "c"}


def test_resolve_encodes_misses_and_appends(tmp_path):
    store = PersistentEmbeddingStore(tmp_path / "s", model="m")
    store.append({"a": _vec(1)})
    calls = []

    def fake_encode(strings):
        calls.append(list(strings))
        return np.stack([_vec(hash(s) % 1000, dim=8) for s in strings])

    out = resolve_embeddings(["a", "b", "a", "c"], store, encode_fn=fake_encode)
    assert out.shape == (4, 8)
    np.testing.assert_allclose(out[0], out[2])
    assert sorted(sum(calls, [])) == ["b", "c"]
    assert "b" in store and "c" in store


def test_resolve_persist_false_does_not_append(tmp_path):
    store = PersistentEmbeddingStore(tmp_path / "s", model="m")
    store.append({"a": _vec(1)})

    def fake_encode(strings):
        return np.stack([_vec(9) for _ in strings])

    out = resolve_embeddings(["a", "b"], store, encode_fn=fake_encode, persist=False)
    assert out.shape == (2, 8)
    assert "b" not in store and store.count == 1
