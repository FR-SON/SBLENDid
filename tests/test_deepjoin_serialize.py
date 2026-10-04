"""Serialization parity units for the DeepJoin sentence builder."""
import nltk
import pytest

pytest.importorskip("nltk")
try:
    nltk.word_tokenize("probe")
except LookupError:
    pytest.skip("nltk tokenizer data unavailable", allow_module_level=True)

from src.Semantic.encoders.deepjoin import (
    read_lake_cells, serialize_column_from_cells,
)


def test_sentence_format_and_frequency_sort():
    cells = ["a", "b", "b", "cc"]
    s = serialize_column_from_cells("mycol", cells)
    assert s.startswith("mycol contains 3 values")
    assert "( 2 , 1 , " in s or "(2, 1," in s.replace(" ,", ",")
    assert s.index("b") < s.index("cc")


def test_sentence_token_cap():
    cells = [f"value{i}" for i in range(2000)]
    s = serialize_column_from_cells("c", cells)
    assert len(s.split(" ")) <= 512


def test_empty_cells():
    s = serialize_column_from_cells("c", [])
    assert s.startswith("c contains 0 values")


def test_read_lake_cells_nan_becomes_literal_nan(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("x,y\n1,\n2,foo\n")
    assert read_lake_cells(p, 1) == ["nan", "foo"]


def test_read_lake_cells_positional(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("x,y\n1,a\n2,b\n")
    assert read_lake_cells(p, 0) == ["1", "2"]
