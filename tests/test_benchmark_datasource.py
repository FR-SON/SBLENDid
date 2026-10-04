import csv
import pandas as pd
from src.Benchmark import datasource as ds

def _write(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        csv.writer(f).writerows(rows)

def test_union_loaders(tmp_path, monkeypatch):
    dsdir = tmp_path / "datasets" / "toy"
    _write(dsdir / "query" / "toy_union_query.csv", [["query_table"], ["a.csv"], ["b.csv"]])
    _write(dsdir / "groundtruth" / "toy_union_ground_truth.csv",
           [["query_table", "candidate_table"], ["a.csv", "a.csv"], ["a.csv", "c.csv"]])
    monkeypatch.setenv("BLEND_DATASETS_DIR", str(tmp_path / "datasets"))
    assert ds.load_union_queries("toy") == ["a.csv", "b.csv"]
    assert ds.load_union_gt("toy") == {"a.csv": {"c.csv"}}

def test_join_loaders(tmp_path, monkeypatch):
    dsdir = tmp_path / "datasets" / "toy"
    _write(dsdir / "query" / "toy_join_query.csv",
           [["query_table", "query_column"], ["a.csv", "c0"], ["a.csv", "c1"]])
    _write(dsdir / "groundtruth" / "toy_join_ground_truth.csv",
           [["query_table", "candidate_table", "query_column", "candidate_column"],
            ["a.csv", "z.csv", "c0", "c0"], ["a.csv", "a.csv", "c0", "c1"]])
    monkeypatch.setenv("BLEND_DATASETS_DIR", str(tmp_path / "datasets"))
    assert ds.load_join_queries("toy") == [("a.csv", "c0"), ("a.csv", "c1")]
    assert ds.load_join_gt("toy") == {("a.csv", "c0"): {"z.csv"}}

def test_load_sidecar(tmp_path, monkeypatch):
    dsdir = tmp_path / "datasets" / "toy"
    dsdir.mkdir(parents=True)
    pd.DataFrame({"table_int_id": [1, 2], "basename": ["a.csv", "b.csv"]}) \
        .to_parquet(dsdir / "blend_index_basenames.parquet")
    monkeypatch.setenv("BLEND_DATASETS_DIR", str(tmp_path / "datasets"))
    i2b, lake = ds.load_sidecar("toy")
    assert i2b == {1: "a.csv", 2: "b.csv"}
    assert lake == {"a.csv", "b.csv"}
