import duckdb

from src.Benchmark.candidate_store import CandidateStore
from src.DBHandler import DBHandler

_SQL = "SELECT colid, rowid, tokenized FROM AllTables WHERE tableid = {tableid}"

_ROWS = ("('paris',1,0,0),('france',1,1,0),"
         "('rome',1,0,1),('italy',1,1,1),"
         "('x',2,0,0)")


def _bare(dbms, cursor):
    db = DBHandler.__new__(DBHandler)
    db.dbms, db.cursor, db.index_table, db.layout = dbms, cursor, "blend_index", "single"
    return db


def _db():
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE blend_index"
                "(tokenized VARCHAR, tableid INTEGER, colid INTEGER, rowid INTEGER)")
    con.execute(f"INSERT INTO blend_index VALUES {_ROWS}")
    return _bare("duckdb", con.cursor())


def test_get_pivots_to_wide_tokenized():
    s = CandidateStore(_db(), sql=_SQL)
    df = s.get(1)
    assert list(df.columns) == [0, 1]
    assert df.loc[0, 0] == "paris" and df.loc[0, 1] == "france"
    assert df.loc[1, 0] == "rome" and df.loc[1, 1] == "italy"


def test_missing_table_returns_none():
    s = CandidateStore(_db(), sql=_SQL)
    assert s.get(999) is None


def test_repeat_get_is_cached_same_object():
    s = CandidateStore(_db(), sql=_SQL)
    assert s.get(1) is s.get(1)


def test_lru_evicts_oldest():
    s = CandidateStore(_db(), sql=_SQL, max_tables=1)
    s.get(1)
    s.get(2)
    assert 1 not in s._cache and 2 in s._cache


def test_col_sets_built_and_cached():
    s = CandidateStore(_db(), sql=_SQL)
    cs = s.col_sets(1)
    assert cs[0] == {"paris", "rome"} and cs[1] == {"france", "italy"}
    assert s.col_sets(1) is cs


def test_col_sets_missing_returns_none():
    s = CandidateStore(_db(), sql=_SQL)
    assert s.col_sets(999) is None


def test_get_and_col_sets_share_one_cache_entry():
    s = CandidateStore(_db(), sql=_SQL)
    s.get(1)
    s.col_sets(1)
    assert list(s._cache) == [1]


def test_reads_through_dbhandler_with_inlined_tableid():
    # no dialect placeholder: `?` (duckdb) vs `%s` (postgres) differ.
    seen = []

    class _Rec:
        dbms = "duckdb"

        def execute_and_fetchall(self, sql):
            seen.append(sql)
            return []

    CandidateStore(_Rec(), sql=_SQL).get(7)
    assert seen == ["SELECT colid, rowid, tokenized FROM AllTables WHERE tableid = 7"]


def test_postgres_pivots_the_same(pg_conn):
    with pg_conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS blend_index")
        cur.execute("CREATE TABLE blend_index"
                    "(tokenized text, tableid int, colid int, rowid int)")
        cur.execute(f"INSERT INTO blend_index VALUES {_ROWS}")
    try:
        s = CandidateStore(_bare("postgres", pg_conn.cursor()), sql=_SQL)
        df = s.get(1)
        assert list(df.columns) == [0, 1]
        assert df.loc[0, 0] == "paris" and df.loc[1, 1] == "italy"
        assert s.get(999) is None
    finally:
        with pg_conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS blend_index")
