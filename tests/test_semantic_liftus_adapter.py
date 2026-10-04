import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.Semantic.encoders.liftus import LiftusAdapter
from src.Semantic.registry import ColumnRef


FIXTURE = Path(__file__).parent / "fixtures" / "semantic"


@pytest.fixture(scope="module")
def adapter():
    sidecar = json.loads((FIXTURE / "ckpt" / "opendata_judit_opendata_balanced.json").read_text())
    enc = LiftusAdapter(
        left_aspects=sidecar["left"],
        right_aspects=sidecar["right"],
        hidden_size=sidecar["hidden_size"],
        num_heads=sidecar["num_heads"],
        aspects_dir=FIXTURE / "aspects",
        dataset=sidecar["train_dataset"],
        device="cpu",
        encoder_version="liftus@test",
    )
    enc.load(FIXTURE / "ckpt" / "opendata_judit_opendata_balanced.pth")
    return enc


def test_encode_one_yields_128_fp32(adapter):
    csv = FIXTURE / "csvs" / "SG_CSV0000000000000007.csv"
    df = pd.read_csv(csv, dtype=str, keep_default_na=False)
    ref = ColumnRef(
        global_id=0, table_id=csv.name, col_idx=0,
        col_name=df.columns[0], n_rows=len(df),
        source_path=csv.name,
    )
    vec = adapter.encode_one(ref)
    assert vec.shape == (128,)
    assert vec.dtype == np.float32


def test_encode_streams_all_columns(adapter):
    refs = []
    gid = 0
    for csv in sorted((FIXTURE / "csvs").glob("*.csv")):
        df = pd.read_csv(csv, dtype=str, keep_default_na=False)
        for i, name in enumerate(df.columns):
            refs.append(ColumnRef(
                global_id=gid, table_id=csv.name, col_idx=i,
                col_name=name, n_rows=len(df), source_path=csv.name,
            ))
            gid += 1
    out = dict(adapter.encode(refs))
    assert set(out.keys()) == {r.global_id for r in refs}
    assert all(v.shape == (128,) for v in out.values())


def test_encode_one_cells_not_supported(adapter):
    ref = ColumnRef(global_id=0, table_id="q.csv", col_idx=0, col_name="c",
                    n_rows=2, source_path="q.csv")
    with pytest.raises(NotImplementedError, match="batch pipeline"):
        adapter.encode_one(ref, cells=["a", "b"])


def test_na_cell_is_empty_string():
    assert LiftusAdapter.na_cell == ""
