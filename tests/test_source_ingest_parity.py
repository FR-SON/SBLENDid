from pathlib import Path
import pandas as pd
from src.Index.source_ingest import (
    list_csv_paths, build_basenames_df, tokenize_one, iter_tokenized_tables,
)

FIX = Path("tests/fixtures/semantic/csvs")

def test_paths_sorted_and_filtered():
    paths = list_csv_paths(FIX)
    assert paths == sorted(paths, key=lambda p: p.as_posix())
    assert all(p.suffix == ".csv" and not p.name.startswith("._") for p in paths)

def test_basenames_df_shape():
    df = build_basenames_df(list_csv_paths(FIX))
    assert list(df.columns) == ["table_int_id", "basename"]
    assert df["table_int_id"].tolist() == list(range(len(df)))

def test_tokenize_one_columns():
    paths = list_csv_paths(FIX)
    rows = tokenize_one((0, str(paths[0])))
    assert list(rows.columns) == ["tokenized", "tableid", "colid", "rowid", "super_key", "quadrant"]
    assert (rows["tableid"] == 0).all()

def test_pool_matches_serial():
    from src.Index.source_ingest import build_jobs
    jobs = build_jobs(list_csv_paths(FIX))
    sort_cols = ["tableid", "colid", "rowid"]
    serial = pd.concat(iter_tokenized_tables(jobs, workers=1), ignore_index=True)
    pooled = pd.concat(iter_tokenized_tables(jobs, workers=2), ignore_index=True)
    serial = serial.sort_values(sort_cols).reset_index(drop=True)
    pooled = pooled.sort_values(sort_cols).reset_index(drop=True)
    pd.testing.assert_frame_equal(serial, pooled)


def test_resume_keeps_tids_stable():
    from src.Index.source_ingest import build_jobs
    jobs = build_jobs(list_csv_paths(FIX))
    remaining = [j for j in jobs if j[0] != 0]
    rows = pd.concat(iter_tokenized_tables(remaining, workers=1), ignore_index=True)
    assert 0 not in set(rows["tableid"])
    assert set(rows["tableid"]) == {j[0] for j in remaining}
