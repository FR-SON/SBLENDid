import duckdb

from scripts.create_blend_csv_index import ingest_csv_dir
from src.DBHandler import DBHandler
from src.Optimizer.freqs import build_freqs_dict
from src.Optimizer.materialize import build_rowid_slice


def _csvs(d):
    (d / "csvs").mkdir(parents=True)
    (d / "csvs" / "alpha.csv").write_text("a,b\nx,1\ny,2\nx,3\n")
    (d / "csvs" / "beta.csv").write_text("a,b\np,4\nx,5\nq,6\n")
    return d / "csvs"


def _build(d, layout):
    db = d / f"{layout}.duckdb"
    ingest_csv_dir(_csvs(d), db, sidecar_path=d / f"sc_{layout}.parquet", workers=1, layout=layout)
    return db


def _handler(duckdb_path):
    cfg = duckdb_path.parent / "cfg.ini"
    cfg.write_text(
        f"[Dataset]\nname = {duckdb_path.parent.name}\nroot = {duckdb_path.parent.parent}\n\n"
        f"[Database]\ndbms = duckdb\ndb_filename = {duckdb_path.name}\nindex_table = blend_index\n"
    )
    return DBHandler.for_dataset(duckdb_path.parent.name, config_path=cfg)


def _freqs_map(path):
    rows = duckdb.connect().execute(
        f"SELECT tokenized, frequency FROM read_csv_auto('{path}')"
    ).fetchall()
    return {str(t): int(f) for t, f in rows}


def test_build_freqs_parity_two_table_vs_single(tmp_path):
    single_db = _build(tmp_path / "s", "single")
    tt_db = _build(tmp_path / "t", "two_table")
    single_out = build_freqs_dict(_handler(single_db), out_path=tmp_path / "freqs_single.csv")
    tt_out = build_freqs_dict(_handler(tt_db), out_path=tmp_path / "freqs_tt.csv")
    assert _freqs_map(tt_out) == _freqs_map(single_out)


def test_build_slice_shape_two_table(tmp_path):
    tt_db = _build(tmp_path / "t", "two_table")
    slice_db = build_rowid_slice(tt_db, tmp_path / "slice.duckdb")
    con = duckdb.connect(str(slice_db), read_only=True)
    tables = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    assert "blend_index" in tables
    assert "blend_index_token" not in tables and "blend_index_tableid" not in tables
    assert con.execute("SELECT count(*) FROM blend_index WHERE rowid >= 256").fetchone()[0] == 0
    descents = con.execute(
        "SELECT count(*) FROM (SELECT tableid, lag(tableid) OVER () p FROM blend_index) WHERE tableid < p"
    ).fetchone()[0]
    assert descents == 0
    con.close()


def test_single_layout_still_works(tmp_path):
    single_db = _build(tmp_path / "s", "single")
    freqs_out = build_freqs_dict(_handler(single_db), out_path=tmp_path / "freqs.csv")
    assert freqs_out.exists()
    slice_db = build_rowid_slice(single_db, tmp_path / "slice.duckdb")
    con = duckdb.connect(str(slice_db), read_only=True)
    tables = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    assert "blend_index" in tables
    con.close()
