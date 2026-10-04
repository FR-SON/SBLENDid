import pandas as pd
import pytest


def _fixture(tmp_path):
    csvs = tmp_path / "csvs"
    csvs.mkdir()
    (csvs / "alpha.csv").write_text("name,num\nfoo,1\nbar,2\nfoo,3\n")
    (csvs / "beta.csv").write_text("name\nbaz\nqux\n")
    sidecar = tmp_path / "blend_index_basenames.parquet"
    pd.DataFrame({"table_int_id": [0, 1], "basename": ["alpha.csv", "beta.csv"]}
                 ).to_parquet(sidecar)
    return csvs, sidecar


def test_sample_writes_values_and_cells(tmp_path):
    from scripts.create_sho_index import run_sample
    csvs, sidecar = _fixture(tmp_path)
    out = tmp_path / "sho"
    stats = run_sample(csvs, out, sidecar, row_cap=1000, seed=0, skip_numeric=True)
    values = pd.read_parquet(out / "values.parquet")
    cells = pd.read_parquet(out / "cells.parquet")
    assert set(values["value"]) == {"foo", "bar", "baz", "qux"}
    assert stats["n_cols_skipped_numeric"] == 1
    a0 = cells[(cells["tableid"] == 0) & (cells["colid"] == 0)]
    assert len(a0) == 3 and sorted(a0["rowid"]) == [0, 1, 2]
    vid = dict(zip(values["value"], values["value_id"]))
    assert (a0["value_id"] == [vid["foo"], vid["bar"], vid["foo"]]).all()
    import json
    meta = json.loads((out / "sample_meta.json").read_text())
    assert meta == {"row_cap": 1000, "seed": 0, "skip_numeric": True}


def test_sample_deterministic(tmp_path):
    from scripts.create_sho_index import run_sample
    csvs, sidecar = _fixture(tmp_path)
    run_sample(csvs, tmp_path / "a", sidecar, row_cap=2, seed=0, skip_numeric=True)
    run_sample(csvs, tmp_path / "b", sidecar, row_cap=2, seed=0, skip_numeric=True)
    for name in ("values.parquet", "cells.parquet"):
        pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "a" / name),
                                      pd.read_parquet(tmp_path / "b" / name))


def test_sample_missing_sidecar_fails(tmp_path):
    from scripts.create_sho_index import run_sample
    csvs, _ = _fixture(tmp_path)
    with pytest.raises(FileNotFoundError):
        run_sample(csvs, tmp_path / "o", tmp_path / "nope.parquet",
                   row_cap=10, seed=0, skip_numeric=True)
