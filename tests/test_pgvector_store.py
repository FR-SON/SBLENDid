from pathlib import Path
import numpy as np
import pytest
import src.Semantic.pgvector_store as store


def test_read_db_params(tmp_path: Path):
    ini = tmp_path / "config.ini"
    ini.write_text(
        "[Database]\ndbms=postgres\nhost=localhost\nport=5432\n"
        "user=blend\npassword=secret\ndbname=blend\nindex_table=blend_index\n"
    )
    p = store.read_db_params(ini)
    assert p == {"host": "localhost", "port": "5432", "user": "blend",
                 "password": "secret", "dbname": "blend"}


def test_ddl_carries_all_columns():
    cols = ["tokenized", "tableid", "colid", "rowid", "super_key", "quadrant"]
    blob = " ".join(store.BLEND_INDEX_INDEXES)
    for c in cols:
        assert c in blob
    assert "to_tokenized" in blob and "to_tableid" in blob
    sql = store.create_semantic_columns_sql("semantic_columns__liftus__default", 128)
    assert "vector(128)" in sql
    table = "semantic_columns__liftus__default"
    default_idx = store.semantic_columns_indexes(table)
    assert any("hnsw" in s for s in default_idx)
    assert any(f"{table}_table_int_id" in s for s in default_idx)


def test_semantic_columns_table_name():
    assert store.semantic_columns_table("liftus", "default") == \
        "semantic_columns__liftus__default"
    assert store.semantic_columns_table("snoopy", "default") == \
        "semantic_columns__snoopy__default"


def test_semantic_columns_table_rejects_bad_ident():
    with pytest.raises(ValueError):
        store.semantic_columns_table("li;drop", "default")


def test_create_sql_and_indexes_use_table_and_dim():
    table = store.semantic_columns_table("liftus", "default")
    sql = store.create_semantic_columns_sql(table, 384)
    assert table in sql
    assert "vector(384)" in sql
    idx = store.semantic_columns_indexes(table)
    assert any("hnsw" in s and table in s for s in idx)
    assert any(f"{table}_table_int_id" in s for s in idx)


def test_live_schema_and_ddl(pg_conn):
    table = "semantic_columns__t__t"
    with pg_conn.cursor() as cur:
        cur.execute(store.CREATE_BLEND_INDEX_SQL)
        for ddl in store.BLEND_INDEX_INDEXES:
            cur.execute(ddl)
        cur.execute(store.create_semantic_columns_sql(table, 8))
        for ddl in store.semantic_columns_indexes(table):
            cur.execute(ddl)
        cur.execute(f"SELECT count(*) FROM {table}")
        assert cur.fetchone()[0] == 0


def _by_gid(res):
    scores, gids = res
    return {int(g): float(x) for g, x in zip(gids[0], scores[0]) if g >= 0}


def test_search_vec_matches_search_gid(pg_semantic_index):
    """Same hits and scores whether the query comes from a stored gid or a vector literal (ties may reorder)."""
    dataset, approach, index_name = pg_semantic_index
    s = store.PgVectorStore(dataset, approach, index_name)
    try:
        gid, k = 0, s.vector_count
        vec = s.get_vector(gid)
        with s.conn.cursor() as cur:
            cur.execute(f"SELECT DISTINCT table_int_id FROM {s.table} ORDER BY 1 LIMIT 2")
            allowed = {r[0] for r in cur.fetchall()}
        cases = [
            (s.search_gid(gid, k), s.search_vec(vec, k)),
            (s.search_gid(gid, k, exact=True), s.search_vec(vec, k, exact=True)),
            (s.search_gid(gid, k, allowed_int_ids=allowed), s.search_vec(vec, k, allowed_int_ids=allowed)),
            (s.search_gid(gid, k), s.search_vec(vec * 3.0, k)),          # literal is re-normalised
        ]
        for by_gid, by_vec in cases:
            a, b = _by_gid(by_gid), _by_gid(by_vec)
            assert a and set(a) == set(b)
            assert all(abs(a[g] - b[g]) < 1e-5 for g in a)
        assert set(_by_gid(cases[2][1])) < set(_by_gid(cases[0][1]))      # the filter prunes rows
    finally:
        s.close()


def test_build_search_sql_vec_literal_vs_gid_subselect():
    table = "semantic_columns__liftus__default"
    vec_sql = store._build_search_sql(table, with_sql="", qv_sql="%(qv)s::vector",
                                      where="WHERE table_int_id = ANY(%(ids)s)", exact=False)
    assert vec_sql.count("%(qv)s::vector") == 2 and "global_id = %(gid)s" not in vec_sql
    assert "WHERE table_int_id = ANY(%(ids)s)" in vec_sql and "LIMIT %(k)s" in vec_sql
    gid_sql = store._build_search_sql(
        table, with_sql=f"WITH q AS MATERIALIZED (SELECT embedding AS v FROM {table} WHERE global_id = %(gid)s) ",
        qv_sql="(SELECT v FROM q)", where="", exact=False)
    assert gid_sql.startswith("WITH q AS MATERIALIZED") and "<=>" in gid_sql
    exact_sql = store._build_search_sql(table, with_sql="", qv_sql="%(qv)s::vector", where="", exact=True)
    assert "<#>" in exact_sql and "ORDER BY embedding <#> %(qv)s::vector, global_id" in exact_sql
