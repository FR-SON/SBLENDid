import pandas as pd


def test_load_codes_into_pg_roundtrip(pg_conn, tmp_path):
    from scripts.load_sho_index_pg import load_codes_into_pg
    conn = pg_conn
    cp = tmp_path / "codes.parquet"
    pd.DataFrame({"simhash_code": [3, 3, 9], "tableid": [0, 1, 0],
                  "colid": [0, 0, 1], "rowid": [0, 0, 2]}
                 ).astype("int32").to_parquet(cp)
    n = load_codes_into_pg(cp, "pytest_pg")
    assert n == 3
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM blend_index_code")
        assert cur.fetchone()[0] == 3
