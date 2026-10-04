import duckdb
import pandas as pd
from pathlib import Path
from scripts.create_blend_csv_index import ingest_csv_dir
from src.DBHandler import DBHandler
from src.Optimizer.freqs import build_freqs_dict

FIXTURE = Path(__file__).parent.parent / "fixtures" / "semantic" / "csvs"


def _duckdb_handler(tmp_path):
    """Real DBHandler over a freshly ingested duckdb."""
    lake_dir = tmp_path / "lake"
    lake_dir.mkdir()
    dbp = lake_dir / "blend.duckdb"
    ingest_csv_dir(csv_dir=FIXTURE, duckdb_path=dbp)
    cfg = tmp_path / "config.ini"
    cfg.write_text(
        f"[Dataset]\nname = lake\nroot = {tmp_path}\n\n"
        f"[Database]\ndbms = duckdb\ndb_filename = blend.duckdb\nindex_table = blend_index\n"
    )
    h = DBHandler()
    h.load_config(cfg)
    return h, dbp


def test_build_freqs_dict_counts_tokens_on_duckdb(tmp_path):
    h, dbp = _duckdb_handler(tmp_path)
    out = tmp_path / "freqs_dict.csv"

    build_freqs_dict(h, out_path=out)

    df = pd.read_csv(out)
    assert list(df.columns) == ["tokenized", "frequency"]
    assert (df["frequency"] >= 1).all()
    con = duckdb.connect(str(dbp), read_only=True)
    n_distinct, = con.execute("SELECT COUNT(DISTINCT tokenized) FROM blend_index").fetchone()
    assert len(df) == n_distinct
