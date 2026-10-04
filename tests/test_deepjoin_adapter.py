"""Adapter behavior with a stub model — never loads sentence-transformers."""
import pickle

import numpy as np
import pytest

from src.Semantic.encoders import deepjoin as dj
from src.Semantic.registry import ColumnRef


class StubModel:
    def __init__(self, dim=768):
        self._dim = dim
        self.encoded: list[list[str]] = []

    def get_sentence_embedding_dimension(self):
        return self._dim

    def encode(self, sentences, **kw):
        batch = [sentences] if isinstance(sentences, str) else list(sentences)
        self.encoded.append(batch)
        return np.ones((len(batch), self._dim), dtype=np.float32)


def _ref(gid, table, col_idx, col_name, source):
    return ColumnRef(global_id=gid, table_id=table, col_idx=col_idx,
                     col_name=col_name, n_rows=2, source_path=source)


@pytest.fixture
def lake(tmp_path):
    (tmp_path / "t1.csv").write_text("a,b\n1,x\n2,y\n")
    return tmp_path


def _adapter(lake, sentences_path=None, dim=768):
    ad = dj.DeepJoinAdapter(
        lake_root=lake, sentences_path=sentences_path,
        encoder_version="deepjoin@test",
    )
    ad._model = StubModel(dim=dim)
    if sentences_path is not None:
        ad._cache = pickle.loads(sentences_path.read_bytes())
    return ad


def test_encode_one_uses_cache(lake, tmp_path):
    cache = {("t1.csv", 0): "CACHED SENTENCE"}
    p = tmp_path / "c.flat.pkl"
    p.write_bytes(pickle.dumps(cache))
    ad = _adapter(lake, sentences_path=p)
    vec = ad.encode_one(_ref(0, "t1.csv", 0, "a", "t1.csv"))
    assert vec.dtype == np.float32 and vec.shape == (768,)
    assert ad._model.encoded[-1] == ["CACHED SENTENCE"]


def test_encode_one_fallback_serializes_from_csv(lake, capsys):
    ad = _adapter(lake)
    ad.encode_one(_ref(0, "t1.csv", 1, "b", "t1.csv"))
    sent = ad._model.encoded[-1][0]
    assert sent.startswith("b contains 2 values")
    assert "FALLBACK" not in capsys.readouterr().err


def test_cache_miss_banner_once_and_counter(lake, tmp_path, capsys):
    cache = {("other.csv", 0): "x"}
    p = tmp_path / "c.flat.pkl"
    p.write_bytes(pickle.dumps(cache))
    ad = _adapter(lake, sentences_path=p)
    refs = [_ref(0, "t1.csv", 0, "a", "t1.csv"),
            _ref(1, "t1.csv", 1, "b", "t1.csv")]
    out = dict(ad.encode(refs))
    assert set(out) == {0, 1}
    err = capsys.readouterr().err
    assert err.count("[DEEPJOIN][SENTENCE-FALLBACK]") == 1
    assert "2/2" in err
    assert ad.n_fallback_total == 2


def test_encode_batches(lake):
    ad = _adapter(lake)
    ad._batch_size = 2
    refs = [_ref(i, "t1.csv", i % 2, "ab"[i % 2], "t1.csv") for i in range(5)]
    out = dict(ad.encode(refs))
    assert len(out) == 5
    assert [len(b) for b in ad._model.encoded] == [2, 2, 1]


def test_load_dim_mismatch_raises(lake, tmp_path, monkeypatch):
    ad = dj.DeepJoinAdapter(lake_root=lake, sentences_path=None)

    class FakeST:
        def __init__(self, *a, **kw): ...
        def get_sentence_embedding_dimension(self):
            return 384

    import sys, types
    fake_mod = types.SimpleNamespace(SentenceTransformer=FakeST)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)
    with pytest.raises(ValueError, match="768"):
        ad.load(tmp_path)


def test_missing_lake_csv_actionable(lake, tmp_path):
    ad = _adapter(lake)
    with pytest.raises(FileNotFoundError, match="gone.csv"):
        ad.encode_one(_ref(0, "gone.csv", 0, "a", "gone.csv"))


def test_na_cell_is_nan_string():
    assert dj.DeepJoinAdapter.na_cell == "nan"


def test_encode_one_cells_matches_csv_fallback(lake):
    ad = _adapter(lake)
    ref = _ref(0, "t1.csv", 1, "b", "t1.csv")
    ad.encode_one(ref)                                   # fallback reads column b of t1.csv
    from_csv = ad._model.encoded[-1][0]
    ad.encode_one(ref, cells=["x", "y"])
    from_cells = ad._model.encoded[-1][0]
    assert from_cells == from_csv
    assert ad.n_fallback_total == 1                      # the cells call never touched the lake


def test_encode_one_cells_bypasses_cache(lake, tmp_path):
    p = tmp_path / "c.flat.pkl"
    p.write_bytes(pickle.dumps({("t1.csv", 0): "CACHED SENTENCE"}))
    ad = _adapter(lake, sentences_path=p)
    ad.encode_one(_ref(0, "t1.csv", 0, "a", "t1.csv"), cells=["1", "2"])
    sent = ad._model.encoded[-1][0]
    assert sent != "CACHED SENTENCE" and sent.startswith("a contains 2 values")


def test_encode_batch_cells_one_model_call(lake):
    ad = _adapter(lake)
    refs = [_ref(0, "q.csv", 0, "a", "q.csv"), _ref(1, "q.csv", 1, "b", "q.csv")]
    out = ad.encode_batch(refs, cells_per_ref={0: ["1", "2"], 1: ["x", "y"]})
    assert set(out) == {0, 1}
    assert all(v.shape == (768,) and v.dtype == np.float32 for v in out.values())
    assert len(ad._model.encoded) == 1 and len(ad._model.encoded[0]) == 2


def test_encode_batch_cells_key_mismatch_raises(lake):
    ad = _adapter(lake)
    with pytest.raises(ValueError, match="cells_per_ref keys"):
        ad.encode_batch([_ref(0, "q.csv", 0, "a", "q.csv")], cells_per_ref={7: ["1"]})
