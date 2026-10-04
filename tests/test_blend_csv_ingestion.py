from pathlib import Path

import duckdb
import pandas as pd

from scripts.create_blend_csv_index import ingest_csv_dir


FIXTURE = Path(__file__).parent / "fixtures" / "semantic" / "csvs"


def test_ingest_csv_dir_creates_index_and_sidecar(tmp_path):
    db = tmp_path / "blend.duckdb"
    sidecar, n = ingest_csv_dir(csv_dir=FIXTURE, duckdb_path=db)

    assert n == 3
    assert db.is_file()
    assert sidecar == tmp_path / "blend_index_basenames.parquet"
    assert sidecar.is_file()

    sc = pd.read_parquet(sidecar)
    assert list(sc.columns) == ["table_int_id", "basename"]
    assert sc["table_int_id"].dtype == "int64"
    expected = sorted(p.name for p in FIXTURE.glob("*.csv"))
    assert sc["basename"].tolist() == expected
    assert sc["table_int_id"].tolist() == [0, 1, 2]

    con = duckdb.connect(str(db), read_only=True)
    try:
        n_rows, = con.execute("SELECT COUNT(*) FROM blend_index").fetchone()
        assert n_rows > 0
        distinct_tids = [r[0] for r in con.execute(
            "SELECT DISTINCT tableid FROM blend_index ORDER BY tableid"
        ).fetchall()]
        assert distinct_tids == [0, 1, 2]
        cols = {r[0]: r[1] for r in con.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name='blend_index'"
        ).fetchall()}
        assert set(cols) == {"tokenized", "tableid", "colid", "rowid",
                             "super_key", "quadrant"}
        assert cols["tableid"].upper() == "INTEGER"
    finally:
        con.close()


def test_ingest_csv_dir_empty_dir_errors(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    db = tmp_path / "blend.duckdb"
    try:
        ingest_csv_dir(csv_dir=empty, duckdb_path=db)
    except SystemExit as exc:
        assert "no csv files" in str(exc)
    else:
        raise AssertionError("expected SystemExit on empty csv_dir")
