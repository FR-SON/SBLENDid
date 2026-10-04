
from src.Semantic.registry import (
    LakeTableEntry, build_registry, load_registry, hash_registry,
)


def test_build_registry_round_trip(tmp_path):
    csv = tmp_path / "t.csv"
    csv.write_text("a,b\n1,2\n3,4\n")
    out = tmp_path / "registry.parquet"
    n = build_registry(
        [LakeTableEntry(table_path=csv, source_path="t.csv")],
        out,
    )
    assert n == 2

    refs = list(load_registry(out))
    assert [r.global_id for r in refs] == [0, 1]
    assert all(r.table_id == "t.csv" for r in refs)
    assert [r.col_name for r in refs] == ["a", "b"]
    assert all(r.n_rows == 2 for r in refs)
    assert all(r.source_path == "t.csv" for r in refs)


def test_hash_registry_stable(tmp_path):
    csv = tmp_path / "t.csv"
    csv.write_text("a\n1\n")
    out = tmp_path / "registry.parquet"
    build_registry([LakeTableEntry(table_path=csv, source_path="t.csv")], out)
    h1 = hash_registry(out)
    h2 = hash_registry(out)
    assert h1 == h2 and h1.startswith("sha256:")
