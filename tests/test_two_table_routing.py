from src.index_routing import route_suffix, physical_table


def test_route_sc_token_only():
    assert route_suffix("SELECT TableId FROM AllTables WHERE CellValue IN ('a') ") == "token"

def test_route_sc_with_pushdown():
    assert route_suffix("... WHERE CellValue IN ('a')  AND TableId IN (1,2) ") == "tableid"

def test_route_mc_refetch_lowercase_in():
    assert route_suffix("SELECT x FROM AllTables WHERE TableId in ('1','2') and RowId in ('3')") == "tableid"

def test_route_c_qualified_join_is_not_a_filter():
    sql = "... ON categorical.TableId = numerical.TableId AND categorical.RowId = numerical.RowId ..."
    assert route_suffix(sql) == "token"

def test_route_unqualified_equality():
    assert route_suffix("SELECT colid, rowid, tokenized FROM AllTables WHERE tableid = ?") == "tableid"
    assert route_suffix("SELECT CellValue FROM AllTables WHERE TableId = 7") == "tableid"

def test_route_not_in_is_token():
    assert route_suffix("... WHERE CellValue IN ('a')  AND TableId NOT IN (1,2) ") == "token"

def test_route_empty_scan_is_token():
    assert route_suffix("SELECT TableId FROM AllTables WHERE 1=0") == "token"

def test_route_ge_le_not_matched():
    assert route_suffix("... WHERE tableid >= 5 AND tableid <= 9") == "token"

def test_route_ignores_tableid_text_inside_token_values():
    assert route_suffix("SELECT TableId FROM AllTables WHERE CellValue IN ('foo' , 'tableid in (1)') ") == "token"
    assert route_suffix("SELECT TableId FROM AllTables WHERE CellValue IN ('tableid = 5') ") == "token"

def test_route_real_pushdown_survives_literal_strip():
    assert route_suffix("... WHERE CellValue IN ('tableid = 5')  AND TableId IN (1,2) ") == "tableid"

def test_physical_table_single_vs_two():
    q_push = "... WHERE TableId IN (1)"
    assert physical_table("blend_index", "single", q_push) == "blend_index"
    assert physical_table("blend_index", "two_table", q_push) == "blend_index_tableid"
    assert physical_table("blend_index", "two_table", "... CellValue IN ('a')") == "blend_index_token"


def test_route_simhash_code_any_layout():
    q = "SELECT TableId FROM AllTables WHERE simhash_code IN (3, 9) GROUP BY TableId, ColumnId"
    assert physical_table("blend_index", "single", q) == "blend_index_code"
    assert physical_table("blend_index", "two_table", q) == "blend_index_code"


def test_route_simhash_code_wins_over_pushdown():
    q = "... WHERE simhash_code IN (3)  AND TableId IN (1,2) "
    assert physical_table("blend_index", "two_table", q) == "blend_index_code"


def test_route_simhash_code_text_inside_literal_does_not_route():
    q = "SELECT TableId FROM AllTables WHERE CellValue IN ('simhash_code') "
    assert physical_table("blend_index", "two_table", q) == "blend_index_token"


from src.DBHandler import DBHandler


def _bare_handler(layout):
    h = DBHandler.__new__(DBHandler)
    h.index_table = "blend_index"
    h.layout = layout
    return h


def test_clean_query_single_unchanged():
    h = _bare_handler("single")
    assert h.clean_query("SELECT * FROM AllTables WHERE TableId IN (1)") == \
        "SELECT * FROM blend_index WHERE TableId IN (1)"


def test_clean_query_two_table_routes():
    h = _bare_handler("two_table")
    assert h.clean_query("SELECT * FROM AllTables WHERE TableId IN (1)") == \
        "SELECT * FROM blend_index_tableid WHERE TableId IN (1)"
    assert h.clean_query("SELECT * FROM AllTables WHERE CellValue IN ('a')") == \
        "SELECT * FROM blend_index_token WHERE CellValue IN ('a')"


def test_clean_query_two_table_equality_routes_tableid():
    h = _bare_handler("two_table")
    assert h.clean_query("SELECT CellValue FROM AllTables WHERE TableId = 7") == \
        "SELECT CellValue FROM blend_index_tableid WHERE TableId = 7"


def test_detect_layout_from_schema(tmp_path):
    import duckdb
    p1 = tmp_path / "single.duckdb"; c = duckdb.connect(str(p1))
    c.execute("CREATE TABLE blend_index(tableid INT)"); c.close()
    h1 = DBHandler.__new__(DBHandler); h1.dbms = "duckdb"; h1.index_table = "blend_index"
    h1.cursor = duckdb.connect(str(p1), read_only=True).cursor()
    assert h1._detect_layout() == "single"
    p2 = tmp_path / "two.duckdb"; c = duckdb.connect(str(p2))
    c.execute("CREATE TABLE blend_index_token(tableid INT)")
    c.execute("CREATE TABLE blend_index_tableid(tableid INT)"); c.close()
    h2 = DBHandler.__new__(DBHandler); h2.dbms = "duckdb"; h2.index_table = "blend_index"
    h2.cursor = duckdb.connect(str(p2), read_only=True).cursor()
    assert h2._detect_layout() == "two_table"


def test_candidate_store_takes_routed_sql():
    import inspect, src.Benchmark.candidate_store as cs
    assert "sql" in inspect.signature(cs.CandidateStore.__init__).parameters
    assert not hasattr(cs, "_SQL")


def test_candidate_store_sql_routes_to_tableid_in_two_table():
    h = _bare_handler("two_table")
    assert h.clean_query("SELECT colid, rowid, tokenized FROM AllTables WHERE tableid = {tableid}") == \
        "SELECT colid, rowid, tokenized FROM blend_index_tableid WHERE tableid = {tableid}"


import pathlib, re

def test_no_hardcoded_index_table_in_source():
    root = pathlib.Path(__file__).resolve().parent.parent
    pat = re.compile(r"FROM\s+blend_index(_token|_tableid|_code)?\b", re.IGNORECASE)
    offenders = []
    for base in ("src", "scripts"):
        for p in (root / base).rglob("*.py"):
            # *_pg.py loaders query psycopg directly, outside clean_query routing
            if p.name.endswith("_pg.py"):
                continue
            for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if pat.search(line):
                    offenders.append(f"{p.relative_to(root)}:{i}")
    assert not offenders, f"hardcoded FROM blend_index (use AllTables): {offenders}"


def _spy(handler):
    routed = []
    orig = handler.clean_query
    def wrapped(q):
        r = orig(q)
        routed.append(r)
        return r
    handler.clean_query = wrapped
    return routed, orig


def test_open_dataset_db_detects_two_table_and_routes_both_legs(tmp_path, monkeypatch):
    import shutil
    from pathlib import Path
    from scripts.create_blend_csv_index import ingest_csv_dir
    from src.Benchmark.db import open_dataset_db
    from src.Benchmark.correctness.arms import run_arm
    from src.Operators import Seekers

    repo = Path(__file__).resolve().parent.parent
    ds = repo / "datasets" / "_tt_routing_test"
    if ds.exists():
        shutil.rmtree(ds)
    (ds / "csvs").mkdir(parents=True)
    (ds / "csvs" / "alpha.csv").write_text("a\nfoo\nbar\nfoo\n")
    (ds / "csvs" / "beta.csv").write_text("a\nfoo\nbaz\n")
    try:
        ingest_csv_dir(ds / "csvs", ds / "blend.duckdb",
                       sidecar_path=ds / "sc.parquet", workers=1, layout="two_table")
        cfg = tmp_path / "config.ini"
        cfg.write_text(
            f"[Dataset]\nname = _tt_routing_test\nroot = {repo / 'datasets'}\n\n"
            f"[Database]\ndbms = duckdb\ndb_filename = blend.duckdb\nindex_table = blend_index\n"
        )
        monkeypatch.setenv("BLEND_CONFIG", str(cfg))
        db = open_dataset_db("_tt_routing_test")
        assert db.layout == "two_table"
        routed, orig = _spy(db)
        legs = [Seekers.SC(["foo"], k=10), Seekers.SC(["bar", "foo"], k=10)]
        result = run_arm(legs, "intersection", "cost", db, 10, guard="none")
        db.clean_query = orig
        assert any("blend_index_token" in q for q in routed), routed
        assert any("blend_index_tableid" in q for q in routed), routed
        assert isinstance(result, list)
        db.close()
    finally:
        shutil.rmtree(ds, ignore_errors=True)
