from pathlib import Path
import pytest
from src.Benchmark.db import open_dataset_db

_ROOT = Path(__file__).resolve().parents[1]

def test_open_dataset_db_missing_raises(tmp_path, monkeypatch):
    cfg = tmp_path / "config.ini"
    cfg.write_text(
        "[Dataset]\nname = santos\n\n"
        "[Database]\ndbms = duckdb\ndb_filename = blend.duckdb\nindex_table = blend_index\n"
    )
    monkeypatch.setenv("BLEND_CONFIG", str(cfg))
    with pytest.raises(FileNotFoundError):
        open_dataset_db("does-not-exist-dataset")

@pytest.mark.skipif(not (_ROOT / "datasets/santos/blend.duckdb").exists(),
                    reason="santos dataset not present")
def test_open_dataset_db_santos_counts_rows():
    db = open_dataset_db("santos")
    try:
        db.cursor.execute("SELECT COUNT(*) FROM blend_index")
        assert db.cursor.fetchone()[0] > 0
        assert db.index_table == "blend_index"
    finally:
        db.close()
