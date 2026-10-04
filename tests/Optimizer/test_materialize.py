import duckdb
from pathlib import Path
from scripts.create_blend_csv_index import ingest_csv_dir
from src.Optimizer.materialize import build_rowid_slice

FIXTURE = Path(__file__).parent.parent / "fixtures" / "semantic" / "csvs"


def test_slice_keeps_only_low_rowids(tmp_path):
    src = tmp_path / "blend.duckdb"
    ingest_csv_dir(csv_dir=FIXTURE, duckdb_path=src)
    out = build_rowid_slice(src, tmp_path / "slice.duckdb", max_rowid=5)

    con = duckdb.connect(str(out), read_only=True)
    mx, = con.execute("SELECT max(rowid) FROM blend_index").fetchone()
    n_slice, = con.execute("SELECT count(*) FROM blend_index").fetchone()
    assert mx < 5

    csrc = duckdb.connect(str(src), read_only=True)
    n_expected, = csrc.execute("SELECT count(*) FROM blend_index WHERE rowid < 5").fetchone()
    assert n_slice == n_expected
    cols = [r[0] for r in con.execute("DESCRIBE blend_index").fetchall()]
    assert cols == [r[0] for r in csrc.execute("DESCRIBE blend_index").fetchall()]
