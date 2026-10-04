"""Cache-vs-on-the-fly DeepJoin sentence parity on split_13; only the stats header is compared (tie order varies across pandas)."""
import pickle
from pathlib import Path

import pytest

DATASET = Path("datasets/opendata-split_13")
CACHE = DATASET / "semantic/deepjoin/default/sentences/opendata.flat.pkl"
CSVS = DATASET / "csvs"
N_SAMPLE = 50

if not (CACHE.is_file() and CSVS.is_dir()):
    pytest.skip("split_13 deepjoin artifacts not staged", allow_module_level=True)

nltk = pytest.importorskip("nltk")
try:
    nltk.word_tokenize("probe")
except LookupError:
    pytest.skip("nltk tokenizer data unavailable", allow_module_level=True)

import pandas as pd

from src.Semantic.encoders.deepjoin import (
    read_lake_cells, serialize_column_from_cells,
)


_HEADER_SEP = " ) : "


def _header(sentence: str) -> str:
    return sentence.partition(_HEADER_SEP)[0]


def test_cache_header_matches_on_the_fly_serialization():
    cache = pickle.loads(CACHE.read_bytes())
    keys = sorted(cache)[:: max(1, len(cache) // N_SAMPLE)][:N_SAMPLE]
    mismatches = []
    for table_id, col_idx in keys:
        csv_path = CSVS / table_id
        if not csv_path.is_file():
            mismatches.append((table_id, col_idx, "csv missing"))
            continue
        cols = pd.read_csv(csv_path, nrows=0).columns
        col_name = str(cols[col_idx])
        got = serialize_column_from_cells(
            col_name, read_lake_cells(csv_path, col_idx))
        if _header(got) != _header(cache[(table_id, col_idx)]):
            mismatches.append((table_id, col_idx, "header differs"))
    assert not mismatches, (
        f"{len(mismatches)}/{len(keys)} header mismatches; first: "
        f"{mismatches[:3]}"
    )
