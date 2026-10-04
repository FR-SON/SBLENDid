import hashlib
import json

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.Semantic.emb_store import MissingEmbeddings, PersistentEmbeddingStore
from src.Semantic.simhash import codes_from_embeddings, get_hyperplanes

DIM = 8


def fake_encode(strings):
    out = []
    for s in strings:
        seed = int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "little")
        v = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
        out.append(v / np.linalg.norm(v))
    return np.stack(out)


def _sampled(tmp_path):
    from scripts.create_sho_index import run_sample
    csvs = tmp_path / "csvs"
    csvs.mkdir()
    (csvs / "alpha.csv").write_text("name\nfoo\nbar\nfoo\n")
    (csvs / "beta.csv").write_text("name\nbaz\nfoo\n")
    sidecar = tmp_path / "blend_index_basenames.parquet"
    pd.DataFrame({"table_int_id": [0, 1], "basename": ["alpha.csv", "beta.csv"]}
                 ).to_parquet(sidecar)
    out = tmp_path / "sho"
    run_sample(csvs, out, sidecar, row_cap=1000, seed=0, skip_numeric=True)
    return out


def test_hash_strict_raises_without_store(tmp_path):
    from scripts.create_sho_index import run_hash
    out = _sampled(tmp_path)
    with pytest.raises(MissingEmbeddings):
        run_hash(out, bits=6, seed=0, encode_fn=None)


def test_hash_writes_codes_and_manifest(tmp_path):
    from scripts.create_sho_index import run_hash
    out = _sampled(tmp_path)
    stats = run_hash(out, bits=6, seed=0, encode_fn=fake_encode)
    codes = pd.read_parquet(out / "codes.parquet")
    assert list(codes.columns) == ["simhash_code", "tableid", "colid", "rowid"]
    assert len(codes) == 5
    PersistentEmbeddingStore(out / "emb_store", model=stats["encoder_model"])
    hp = get_hyperplanes(6, DIM, seed=0)
    expected_foo = int(codes_from_embeddings(fake_encode(["foo"]), hp)[0])
    foo_rows = codes[(codes["tableid"] == 0) & (codes["rowid"].isin([0, 2]))]
    assert set(foo_rows["simhash_code"]) == {expected_foo}
    man = json.loads((out / "manifest.json").read_text())
    for key in ("bits", "seed", "sample_seed", "row_cap", "skip_numeric",
                "encoder_model", "dim", "n_tables", "n_codes"):
        assert key in man
    assert man["bits"] == 6 and man["dim"] == DIM


def test_materialize_duckdb_sorted(tmp_path):
    from scripts.create_sho_index import materialize_duckdb, run_hash
    out = _sampled(tmp_path)
    run_hash(out, bits=6, seed=0, encode_fn=fake_encode)
    db = tmp_path / "blend.duckdb"
    materialize_duckdb(db, out / "codes.parquet")
    con = duckdb.connect(str(db), read_only=True)
    rows = con.execute("SELECT count(*) FROM blend_index_code").fetchone()[0]
    desc = con.execute(
        "SELECT count(*) FROM (SELECT simhash_code, lag(simhash_code) OVER () p "
        "FROM blend_index_code) WHERE simhash_code < p").fetchone()[0]
    con.close()
    assert rows == 5 and desc == 0
