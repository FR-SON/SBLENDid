import duckdb
from scripts.create_blend_csv_index import ingest_csv_dir


def _write_csvs(d):
    (d / "csvs").mkdir()
    (d / "csvs" / "alpha.csv").write_text("a,b\nx,1\ny,2\nz,3\n")
    (d / "csvs" / "beta.csv").write_text("a,b\np,4\nq,5\n")
    return d / "csvs"


def test_two_table_builds_both_with_parity(tmp_path):
    csvs = _write_csvs(tmp_path)
    db = tmp_path / "t.duckdb"
    ingest_csv_dir(csvs, db, sidecar_path=tmp_path / "sc.parquet", workers=1, layout="two_table")
    con = duckdb.connect(str(db), read_only=True)
    tables = {r[0] for r in con.execute("select table_name from information_schema.tables").fetchall()}
    assert "blend_index_token" in tables and "blend_index_tableid" in tables
    assert "blend_index" not in tables
    n_t = con.execute("select count(*) from blend_index_token").fetchone()[0]
    n_i = con.execute("select count(*) from blend_index_tableid").fetchone()[0]
    assert n_t == n_i and n_t > 0
    assert con.execute("(select * from blend_index_token) except (select * from blend_index_tableid)").fetchall() == []
    desc = con.execute("select count(*) from (select tableid, lag(tableid) over () p from blend_index_tableid) where tableid < p").fetchone()[0]
    assert desc == 0
