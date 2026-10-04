import numpy as np
import pandas as pd

from src.Semantic.emb_store import PersistentEmbeddingStore


def _fake_encode(strings):
    return np.stack([np.full(4, (len(s) + 1) / 10.0, dtype=np.float32) for s in strings])


def test_embed_values_fills_store_and_resumes(tmp_path):
    from scripts.embed_sho_values import embed_values
    vp = tmp_path / "values.parquet"
    pd.DataFrame({"value_id": [0, 1, 2], "value": ["a", "bb", "ccc"]}).to_parquet(vp)
    store_dir = tmp_path / "emb_store"

    stats = embed_values(vp, store_dir, model="m", device="cpu",
                         batch=2, encode_fn=_fake_encode)
    assert stats["n_encoded"] == 3 and stats["n_total"] == 3
    store = PersistentEmbeddingStore(store_dir, model="m")
    assert store.count == 3 and "bb" in store

    stats2 = embed_values(vp, store_dir, model="m", device="cpu",
                          batch=2, encode_fn=_fake_encode)
    assert stats2["n_encoded"] == 0


def test_full_resume_never_builds_encoder(tmp_path):
    from scripts.embed_sho_values import embed_values
    vp = tmp_path / "values.parquet"
    pd.DataFrame({"value_id": [0, 1, 2], "value": ["a", "bb", "ccc"]}).to_parquet(vp)
    store_dir = tmp_path / "emb_store"
    embed_values(vp, store_dir, model="m", device="cpu",
                 batch=2, encode_fn=_fake_encode)

    stats = embed_values(vp, store_dir, model="m", device="cpu",
                         batch=2, encode_fn=None)
    assert stats["n_encoded"] == 0
