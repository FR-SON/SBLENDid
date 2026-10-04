import duckdb
import pandas as pd

from scripts.create_blend_csv_index import ingest_csv_dir
from src.DBHandler import DBHandler
from src.Operators.Seekers.MultiColumnOverlap import MultiColumnOverlap

_ROWS = [("New York", "USA"), ("Berlin", "Germany"), ("Paris", "France"),
         ("Rome", "Italy"), ("Madrid", "Spain")]


def _db(tmp_path):
    csvs = tmp_path / "csvs"
    csvs.mkdir()
    pd.DataFrame(_ROWS, columns=["City", "Country"]).to_csv(csvs / "cand.csv", index=False)
    path = tmp_path / "blend.duckdb"
    ingest_csv_dir(csv_dir=csvs, duckdb_path=path)
    db = DBHandler.__new__(DBHandler)
    db.dbms, db.index_table, db.layout = "duckdb", "blend_index", "single"
    db.connection = duckdb.connect(str(path), read_only=True)
    db.cursor = db.connection.cursor()
    return db


def test_mc_matches_untokenized_input(tmp_path):
    db = _db(tmp_path)
    mc = MultiColumnOverlap(pd.DataFrame(_ROWS, columns=["City", "Country"]), k=10)
    sql = mc.create_sql_query(db)
    assert "1=0" not in sql
    assert [r[0] for r in db.execute_and_fetchall(sql)] == [0]


def test_mc_input_is_tokenized_at_construction(tmp_path):
    mc = MultiColumnOverlap(pd.DataFrame(_ROWS, columns=["City", "Country"]), k=10)
    assert mc.input.iloc[0].tolist() == ["new york", "usa"]


def test_mc_tokenized_input_is_unchanged(tmp_path):
    db = _db(tmp_path)
    rows = [(k.lower(), v.lower()) for k, v in _ROWS]
    mc = MultiColumnOverlap(pd.DataFrame(rows, columns=["City", "Country"]), k=10)
    assert [r[0] for r in db.execute_and_fetchall(mc.create_sql_query(db))] == [0]
