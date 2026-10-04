from __future__ import annotations
from pathlib import Path

import numpy as np
import pytest
import torch

from snoopy.model import Scorpion

from src.Semantic.encoders.snoopy import ScorpionAdapter, _normalize_cells
from src.Semantic.registry import ColumnRef
from tests.fixtures.semantic_join._fake_embedder import fake_embedder


FIXTURE = Path(__file__).parent / "fixtures" / "semantic_join"


def _make_adapter(tmp_path: Path, *, row_cap: int = 8, cell_char_cap: int = 64,
                  dataset: str = "fixture") -> ScorpionAdapter:
    enc = ScorpionAdapter(
        n_proxy_sets=4, n_elements=2, d=4, device="cpu",
        lake_root=tmp_path,
        cell_embedder=fake_embedder(4),
        cell_char_cap=cell_char_cap,
        row_cap=row_cap,
        row_cap_seed=0,
        dataset=dataset,
        eval_chunk_size=4,
        encoder_version="snoopy@test",
    )
    enc.load(FIXTURE / "ckpt" / "tiny.pth")
    return enc


def _ref(source_path: str, n_rows: int = 5) -> ColumnRef:
    return ColumnRef(
        global_id=0, table_id=source_path, col_idx=0,
        col_name="c0", n_rows=n_rows, source_path=source_path,
    )


def test_snoopy_adapter_dim_matches_n_proxy_sets(tmp_path):
    enc = _make_adapter(tmp_path)
    assert enc.dim == 4
    assert enc.natively_normalized is True


def test_snoopy_adapter_reads_cells_from_lake_root(tmp_path):
    csv = tmp_path / "t.csv"
    csv.write_text("c0\nfoo\nbar\n")
    enc = _make_adapter(tmp_path)
    spy_calls: list[list[str]] = []
    def spy(cells):
        spy_calls.append(list(cells))
        return fake_embedder(4)(cells)
    enc._embed = spy
    vec = enc.encode_one(_ref("t.csv", n_rows=2))
    assert vec.shape == (4,)
    assert spy_calls and spy_calls[0] == ["foo", "bar"]


def test_snoopy_adapter_applies_row_cap_deterministically(tmp_path):
    csv = tmp_path / "t.csv"
    csv.write_text("c0\n" + "\n".join(f"r{i}" for i in range(20)) + "\n")
    enc = _make_adapter(tmp_path, row_cap=4)
    v1 = enc.encode_one(_ref("t.csv", n_rows=20))
    v2 = enc.encode_one(_ref("t.csv", n_rows=20))
    np.testing.assert_array_equal(v1, v2)


def test_snoopy_adapter_different_basename_different_sample(tmp_path):
    body = "c0\n" + "\n".join(f"r{i}" for i in range(20)) + "\n"
    (tmp_path / "a.csv").write_text(body)
    (tmp_path / "b.csv").write_text(body)
    enc = _make_adapter(tmp_path, row_cap=4)
    va = enc.encode_one(_ref("a.csv", n_rows=20))
    vb = enc.encode_one(_ref("b.csv", n_rows=20))
    assert not np.array_equal(va, vb)


def test_snoopy_adapter_applies_char_cap(tmp_path):
    long_cell = "x" * 200
    csv = tmp_path / "t.csv"
    csv.write_text(f"c0\n{long_cell}\n")
    enc = _make_adapter(tmp_path, cell_char_cap=10, row_cap=8)
    spy_calls: list[list[str]] = []
    def spy(cells):
        spy_calls.append(list(cells))
        return fake_embedder(4)(cells)
    enc._embed = spy
    enc.encode_one(_ref("t.csv", n_rows=1))
    assert spy_calls[0] == ["x" * 10]


def test_normalize_cells_strips_newlines():
    out = _normalize_cells(["hi\nthere", "ok"], cap=100)
    assert out == ["hi there", "ok"]


def test_snoopy_adapter_load_strict_raises_on_arch_mismatch(tmp_path):
    import argparse
    args = argparse.Namespace(n_proxy_sets=8, n_elements=2, d=4, device="cpu")
    bad_model = Scorpion(args)
    bad_path = tmp_path / "bad.pth"
    torch.save(bad_model.state_dict(), bad_path)
    enc = _make_adapter(tmp_path)
    with pytest.raises(RuntimeError):
        enc.load(bad_path)


def test_na_cell_is_empty_string():
    assert ScorpionAdapter.na_cell == ""


def test_encode_one_cells_matches_csv_path(tmp_path):
    cells = [f"v{i}" for i in range(12)]            # 12 > row_cap 8: the seeded sample must agree
    (tmp_path / "t.csv").write_text("c0\n" + "\n".join(cells) + "\n")
    enc = _make_adapter(tmp_path, row_cap=8)
    ref = _ref("t.csv", n_rows=12)
    from_csv = enc.encode_one(ref)
    from_cells = enc.encode_one(ref, cells=cells)
    assert np.array_equal(from_csv, from_cells)


def test_encode_one_cells_never_reads_lake(tmp_path):
    enc = _make_adapter(tmp_path)                    # tmp_path holds no t.csv
    vec = enc.encode_one(_ref("t.csv", n_rows=2), cells=["foo", "bar"])
    assert vec.shape == (4,) and vec.dtype == np.float32


def test_encode_batch_cells_per_ref_key_mismatch_raises(tmp_path):
    enc = _make_adapter(tmp_path)
    with pytest.raises(ValueError, match="cells_per_ref keys"):
        enc.encode_batch([_ref("t.csv", n_rows=1)], cells_per_ref={99: ["x"]})


def test_encode_batch_cells_length_mismatch_raises(tmp_path):
    enc = _make_adapter(tmp_path)
    with pytest.raises(ValueError, match="length must match ref.n_rows"):
        enc.encode_batch([_ref("t.csv", n_rows=2)], cells_per_ref={0: ["x"]})
