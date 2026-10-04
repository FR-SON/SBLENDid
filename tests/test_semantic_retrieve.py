import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.Semantic.config import SemanticConfig
from src.Semantic.index_build import build_semantic_index
from src.Semantic.retrieve import IndexHandle, QueryColumnNotIndexed, search, reset_semantic_cache


FIXTURE = Path(__file__).parent / "fixtures" / "semantic"


def _write_basenames(parquet_path: Path) -> None:
    basenames = sorted(p.name for p in (FIXTURE / "csvs").glob("*.csv"))
    pd.DataFrame(
        [(i, b) for i, b in enumerate(basenames)],
        columns=["table_int_id", "basename"],
    ).astype({"table_int_id": "int64", "basename": "string"}).to_parquet(
        parquet_path, index=False
    )


def _stage_approach(root: Path, dataset: str = "test_ds") -> Path:
    approach_dir = root / dataset / "semantic" / "liftus" / "demo"
    (approach_dir / "ckpt").mkdir(parents=True)
    shutil.copytree(FIXTURE / "ckpt", approach_dir / "ckpt", dirs_exist_ok=True)
    shutil.copytree(FIXTURE / "aspects", approach_dir / "aspects")
    return root / dataset


@pytest.fixture()
def index_dir(tmp_path):
    root = tmp_path / "data"
    dataset_dir = _stage_approach(root)
    _write_basenames(dataset_dir / "blend_index_basenames.parquet")

    cfg = SemanticConfig.load(
        path=tmp_path / "missing.ini",
        overrides={
            "dataset": "test_ds",
            "dataset_root": str(root),
            "faiss_quant": "flat",
        },
    )
    reset_semantic_cache()
    build_semantic_index(FIXTURE / "csvs", "liftus", "demo", cfg)
    return cfg


def test_open_index_handle_caches(index_dir):
    cfg = index_dir
    h1 = IndexHandle.open(cfg, "liftus", "demo")
    h2 = IndexHandle.open(cfg, "liftus", "demo")
    assert h1 is h2
    assert h1.manifest["approach"] == "liftus"
    assert h1.encoder.dim == 128
    assert set(h1.table_to_int_id) == set(h1.gid_to_table.values())


def test_search_returns_int_table_ids(index_dir):
    cfg = index_dir
    handle = IndexHandle.open(cfg, "liftus", "demo")
    df = pd.read_csv(
        FIXTURE / "csvs" / "SG_CSV0000000000000007.csv",
        dtype=str, keep_default_na=False,
    )
    ids = search(handle, df, k=2, k_coarse=20,
                 query_table_id="SG_CSV0000000000000007.csv")
    assert isinstance(ids, list)
    assert 0 < len(ids) <= 2
    assert all(isinstance(t, int) for t in ids)
    assert set(ids).issubset({0, 1, 2})


def test_search_excludes_query_table_from_results(index_dir):
    cfg = index_dir
    handle = IndexHandle.open(cfg, "liftus", "demo")
    qtid = "SG_CSV0000000000000007.csv"
    df = pd.read_csv(
        FIXTURE / "csvs" / qtid, dtype=str, keep_default_na=False,
    )
    self_int_id = handle.table_to_int_id[qtid]
    ids = search(handle, df, k=10, k_coarse=20, query_table_id=qtid)
    assert self_int_id not in ids


def test_open_handle_errors_on_missing_basenames_file(tmp_path):
    root = tmp_path / "data"
    _stage_approach(root)

    cfg = SemanticConfig.load(
        path=tmp_path / "missing.ini",
        overrides={
            "dataset": "test_ds",
            "dataset_root": str(root),
            "faiss_quant": "flat",
        },
    )
    reset_semantic_cache()
    build_semantic_index(FIXTURE / "csvs", "liftus", "demo", cfg)
    reset_semantic_cache()
    with pytest.raises(FileNotFoundError, match="blend_index_basenames.parquet"):
        IndexHandle.open(cfg, "liftus", "demo")


def test_open_handle_errors_on_basename_drift(tmp_path):
    root = tmp_path / "data"
    dataset_dir = _stage_approach(root)

    pd.DataFrame(
        [(0, "SG_CSV0000000000000007.csv"), (1, "SG_CSV0000000000000008.csv")],
        columns=["table_int_id", "basename"],
    ).astype({"table_int_id": "int64", "basename": "string"}).to_parquet(
        dataset_dir / "blend_index_basenames.parquet", index=False
    )

    cfg = SemanticConfig.load(
        path=tmp_path / "missing.ini",
        overrides={
            "dataset": "test_ds",
            "dataset_root": str(root),
            "faiss_quant": "flat",
        },
    )
    reset_semantic_cache()
    build_semantic_index(FIXTURE / "csvs", "liftus", "demo", cfg)
    reset_semantic_cache()
    with pytest.raises(KeyError, match="missing"):
        IndexHandle.open(cfg, "liftus", "demo")


class _FakeEncoder:
    na_cell = ""
    dim = 128

    def __init__(self):
        self.calls: list[list[str]] = []

    def encode_batch(self, refs, *, cells_per_ref=None):
        self.calls.append([r.col_name for r in refs])
        return {r.global_id: np.random.default_rng(r.global_id + len(cells_per_ref[r.global_id]))
                .standard_normal(128).astype(np.float32) for r in refs}


def _fake_handle(cfg):
    handle = IndexHandle.open(cfg, "liftus", "demo")
    fake = _FakeEncoder()
    handle.__dict__["encoder"] = fake          # the cached_property slot
    return handle, fake


def test_search_unindexed_table_encodes_and_warns_once(index_dir, capsys):
    handle, fake = _fake_handle(index_dir)
    df = pd.DataFrame({"name": ["alice", "bob"], "age": ["1", "2"]})
    ids = search(handle, df, k=2, k_coarse=20, query_table_id="external.csv", query_encode="auto")
    assert 0 < len(ids) <= 2 and set(ids) <= {0, 1, 2}
    assert fake.calls == [["name", "age"]]
    err = capsys.readouterr().err
    assert err.count("[SEMANTIC][QUERY-ENCODE]") == 1
    assert "approach=liftus/demo" in err and "table='external.csv'" in err and "cols=2" in err
    search(handle, df, k=2, k_coarse=20, query_table_id="external.csv", query_encode="auto")
    assert fake.calls == [["name", "age"]]                       # vector cache hit
    assert "[SEMANTIC][QUERY-ENCODE]" not in capsys.readouterr().err


def test_search_recomputes_when_cells_change(index_dir):
    handle, fake = _fake_handle(index_dir)
    df1 = pd.DataFrame({"name": ["alice", "bob"]})
    df2 = pd.DataFrame({"name": ["carol", "dan"]})
    search(handle, df1, k=2, k_coarse=20, query_table_id="external.csv", query_encode="auto")
    search(handle, df2, k=2, k_coarse=20, query_table_id="external.csv", query_encode="auto")
    assert fake.calls == [["name"], ["name"]]


def test_search_partial_enrolment_encodes_only_missing(index_dir):
    handle, fake = _fake_handle(index_dir)
    qtid = "SG_CSV0000000000000007.csv"
    df = pd.read_csv(FIXTURE / "csvs" / qtid, dtype=str, keep_default_na=False)
    df["brand_new"] = "x"
    ids = search(handle, df, k=2, k_coarse=20, query_table_id=qtid, query_encode="auto")
    assert fake.calls == [["brand_new"]]
    assert handle.table_to_int_id[qtid] not in ids


def test_search_enrolled_table_with_edited_cells_uses_lookup(index_dir, capsys):
    handle, fake = _fake_handle(index_dir)
    qtid = "SG_CSV0000000000000007.csv"
    df = pd.read_csv(FIXTURE / "csvs" / qtid, dtype=str, keep_default_na=False)
    df.iloc[:, 0] = "edited"
    ids = search(handle, df, k=2, k_coarse=20, query_table_id=qtid)
    assert fake.calls == []
    assert 0 < len(ids) <= 2
    assert "[SEMANTIC][QUERY-ENCODE]" not in capsys.readouterr().err


def test_search_skips_empty_unindexed_columns(index_dir, capsys):
    handle, fake = _fake_handle(index_dir)
    df = pd.DataFrame({"blank": ["", None], "name": ["alice", "bob"]})
    search(handle, df, k=2, k_coarse=20, query_table_id="external.csv", query_encode="auto")
    assert fake.calls == [["name"]]
    assert "skipped=['blank']" in capsys.readouterr().err


def test_search_all_empty_returns_nothing(index_dir):
    handle, fake = _fake_handle(index_dir)
    df = pd.DataFrame({"blank": ["", None]})
    assert search(handle, df, k=2, k_coarse=20, query_table_id="external.csv", query_encode="auto") == []
    assert fake.calls == []


def test_search_off_raises_for_unindexed(index_dir):
    handle, fake = _fake_handle(index_dir)
    df = pd.DataFrame({"name": ["alice"]})
    with pytest.raises(QueryColumnNotIndexed, match="not present in the liftus/demo registry"):
        search(handle, df, k=2, k_coarse=20, query_table_id="external.csv")      # default: off
    assert fake.calls == []


def test_search_without_table_id(index_dir):
    handle, fake = _fake_handle(index_dir)
    df = pd.DataFrame({"name": ["alice", "bob"]})
    ids = search(handle, df, k=2, k_coarse=20, query_encode="auto")   # auto: sentinel table id
    assert 0 < len(ids) <= 2
    with pytest.raises(ValueError, match="query_table_id"):
        search(handle, df, k=2, k_coarse=20)                               # default: off


def test_search_recomputes_when_dtype_changes(index_dir):
    handle, fake = _fake_handle(index_dir)
    df_bool = pd.DataFrame({"flag": [True, False]})
    df_int = pd.DataFrame({"flag": [1, 0]})          # same bits, different cell strings
    search(handle, df_bool, k=2, k_coarse=20, query_table_id="external.csv", query_encode="auto")
    search(handle, df_int, k=2, k_coarse=20, query_table_id="external.csv", query_encode="auto")
    assert fake.calls == [["flag"], ["flag"]]
