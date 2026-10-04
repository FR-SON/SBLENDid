"""PG-gated integration tests for load_blend_index_pg and load_semantic_index_pg."""
from pathlib import Path
import shutil

FIX = Path("tests/fixtures/semantic/csvs")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SIDECAR_DIR = _PROJECT_ROOT / "datasets" / "pytest_pg"


def test_load_blend_index_pg(pg_conn):
    from scripts.load_blend_index_pg import load_blend_index_pg

    try:
        n = load_blend_index_pg(FIX, "pytest_pg", workers=1, resume=False)
        assert n == len(list(FIX.glob("*.csv")))
        with pg_conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM blend_index")
            assert cur.fetchone()[0] > 0
            cur.execute(
                "SELECT indexname FROM pg_indexes"
                " WHERE tablename='blend_index' AND schemaname='pytest_pg'"
            )
            names = {r[0] for r in cur.fetchall()}
        assert {"blend_index_to_tokenized", "blend_index_to_tableid"} <= names
        with pg_conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM blend_index WHERE quadrant IS NULL")
            assert cur.fetchone()[0] > 0
    finally:
        shutil.rmtree(_SIDECAR_DIR, ignore_errors=True)


def test_load_semantic_index_pg(pg_conn, pg_semantic_index):
    import src.Semantic.pgvector_store as store
    _, approach, index_name = pg_semantic_index
    table = store.semantic_columns_table(approach, index_name)
    with pg_conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {table}")
        assert cur.fetchone()[0] > 0
        cur.execute(f"SELECT vector_dims(embedding) FROM {table} LIMIT 1")
        assert cur.fetchone()[0] == 128


def test_second_load_does_not_drop_first(pg_conn, pg_semantic_index):
    from scripts.load_semantic_index_pg import load_semantic_index_pg
    import src.Semantic.pgvector_store as store
    dataset, approach, index_name = pg_semantic_index
    sentinel = store.semantic_columns_table("other", "demo")
    with pg_conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {sentinel}")
        cur.execute(store.create_semantic_columns_sql(sentinel, 4))
    load_semantic_index_pg(dataset, approach, index_name)
    with pg_conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (f"pytest_pg.{sentinel}",))
        assert cur.fetchone()[0] is not None
