"""Index-time and query-time cell tokenization must both go through tokenize_cell."""

from __future__ import annotations

import pytest

from src.DBHandler import DBHandler
from src.Index.tokenize import tokenize_cell


_SAMPLES = [
    "Hello World",
    "  weird  spacing  ",
    "Don't Stop",
    "WITH/SLASH",
    "Pothole/Roadway Surface Repair",
    "tab\there",
    'quote"inside',
    "x" * 250,
    "NaN", "None", "nan", "none", "",
    "   ",
    "MixedCase ÄÖÜ",
    42, 3.14, None,
]


@pytest.mark.parametrize("v", _SAMPLES)
def test_query_path_matches_index_path(v):
    """Query-side tokens equal tokenize_cell output; empty tokens are dropped at query time."""
    indexed = tokenize_cell(v)
    queried = DBHandler.clean_value_collection([v])
    if indexed == "":
        assert queried == []
    else:
        assert queried == [indexed], (
            f"DRIFT: index stored {indexed!r} but query produced {queried!r}"
        )


def test_no_inline_tokenizer_in_repo():
    import pathlib
    repo = pathlib.Path(__file__).resolve().parent.parent
    needle = "lower().replace('\\\\', '')"
    # legacy upstream Vertica ingest keeps its original inline tokenizer
    allow_list = {"scripts/create_index.py"}
    offenders = []
    for path in (repo / "src").rglob("*.py"):
        if path.name == "tokenize.py":
            continue
        if needle in path.read_text():
            offenders.append(path)
    for path in (repo / "scripts").rglob("*.py"):
        rel = path.relative_to(repo).as_posix()
        if rel in allow_list:
            continue
        if needle in path.read_text():
            offenders.append(path)
    assert not offenders, (
        f"inline tokenizer chain found in {offenders}; "
        f"import tokenize_cell from src/Index/tokenize.py instead"
    )
