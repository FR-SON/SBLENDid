import shutil
from pathlib import Path

import pandas as pd
import pytest

from src.Semantic.config import SemanticConfig, SemanticOp, OperatorConfig
from src.Semantic.index_build import build_semantic_index
from src.Semantic.retrieve import reset_semantic_cache
from src.Optimizer import semantic_sweep as sw

FIXTURE = Path(__file__).parent.parent / "fixtures" / "semantic"


@pytest.fixture()
def cfg(tmp_path):
    root = tmp_path / "data"
    ad = root / "test_ds" / "semantic" / "liftus" / "default"
    (ad / "ckpt").mkdir(parents=True)
    shutil.copytree(FIXTURE / "ckpt", ad / "ckpt", dirs_exist_ok=True)
    shutil.copytree(FIXTURE / "aspects", ad / "aspects")
    names = sorted(p.name for p in (FIXTURE / "csvs").glob("*.csv"))
    pd.DataFrame(
        [(i, b) for i, b in enumerate(names)],
        columns=["table_int_id", "basename"],
    ).astype({"table_int_id": "int64", "basename": "string"}).to_parquet(
        root / "test_ds" / "blend_index_basenames.parquet", index=False
    )
    csv_dst = root / "test_ds" / "csvs"
    shutil.copytree(FIXTURE / "csvs", csv_dst)
    cfg = SemanticConfig.load(
        overrides={
            "dataset": "test_ds",
            "dataset_root": str(root),
            "faiss_quant": "flat",
            "vector_backend": "faiss",
        },
        operators={
            SemanticOp.SU: OperatorConfig(approach="liftus", index_name="default"),
            SemanticOp.SJ: OperatorConfig(approach="liftus", index_name="default"),
        },
    )
    reset_semantic_cache()
    build_semantic_index(FIXTURE / "csvs", "liftus", "default", cfg)
    return cfg


def test_sample_su_queries_are_tables_with_ncols(cfg):
    qs = sw.sample_su_queries(cfg, n=2, seed=0)
    assert 1 <= len(qs) <= 2
    assert all(q.col_name is None and q.n_cols >= 1 for q in qs)


def test_sample_sj_queries_are_single_columns(cfg):
    qs = sw.sample_sj_queries(cfg, n=3, seed=0)
    assert all(q.col_name is not None and q.n_cols == 1 for q in qs)


def test_build_gids_filter_reaches_target(cfg):
    import numpy as np
    from src.Semantic.retrieve import IndexHandle
    h = IndexHandle.open(cfg, "liftus", "default")
    total_gids = sum(len(v) for v in h.table_to_gids.values())
    rng = np.random.default_rng(0)
    int_ids, n_gids = sw.build_gids_filter(h, target_gids=1, rng=rng)
    assert n_gids >= 1 and len(int_ids) >= 1
    int_ids2, n_gids2 = sw.build_gids_filter(h, target_gids=10 ** 9, rng=rng)
    assert n_gids2 == total_gids
    assert all(isinstance(i, int) for i in int_ids2)


def test_run_exact_curve_writes_rows(cfg, tmp_path):
    qs = sw.sample_su_queries(cfg, n=2, seed=0)
    out = sw.run_exact_curve(cfg, "SU", qs, gids_grid=[1, 4], out_csv=tmp_path / "e.csv")
    df = pd.read_csv(out)
    assert set(df.columns) >= {"seeker", "n_query_cols", "m_tables", "n_gids", "exact_ms"}
    assert len(df) == 2 * 2 and (df["exact_ms"] >= 0).all()
    assert (df["n_gids"] >= 1).all()


def test_run_hnsw_surface_writes_rows(cfg, tmp_path):
    ef_grid = [16, 64]
    kc_grid = [10]
    qs = sw.sample_su_queries(cfg, n=2, seed=0)
    out = sw.run_hnsw_surface(
        cfg, "SU", qs, ef_grid=ef_grid, kc_grid=kc_grid,
        out_csv=tmp_path / "h.csv",
    )
    df = pd.read_csv(out)
    assert set(df.columns) >= {
        "seeker", "efSearch", "k_coarse", "n_cols",
        "faiss_ms", "total_ms", "overhead_ms", "plan_label",
    }
    assert len(df) == len(ef_grid) * len(kc_grid) * len(qs)
    assert (df["total_ms"] >= 0).all()
    assert df["faiss_ms"].notna().all() and (df["faiss_ms"] >= 0).all()


def test_run_hnsw_surface_diagonal_measures_only_ef_equals_kc(cfg, tmp_path):
    qs = sw.sample_su_queries(cfg, n=2, seed=0)
    out = sw.run_hnsw_surface(cfg, "SU", qs, ef_grid=[5, 10], kc_grid=[5, 10],
                              out_csv=tmp_path / "h.csv", diagonal=True)
    df = pd.read_csv(out)
    assert set(zip(df["efSearch"], df["k_coarse"])) == {(5, 5), (10, 10)}
    assert len(df) == 2 * len(qs)


def test_run_hnsw_surface_diagonal_without_a_shared_value_fails(cfg, tmp_path):
    qs = sw.sample_su_queries(cfg, n=1, seed=0)
    with pytest.raises(SystemExit, match="no ef == k_coarse cell"):
        sw.run_hnsw_surface(cfg, "SU", qs, ef_grid=[5], kc_grid=[10],
                            out_csv=tmp_path / "h.csv", diagonal=True)


def test_run_hnsw_surface_labels_faiss_plan(cfg, tmp_path):
    qs = sw.sample_su_queries(cfg, n=1, seed=0)
    out = sw.run_hnsw_surface(cfg, "SU", qs, ef_grid=[16], kc_grid=[10],
                              out_csv=tmp_path / "h.csv")
    assert (pd.read_csv(out)["plan_label"] == "faiss").all()


def test_query_df_falls_back_to_index_without_csv(cfg):
    import shutil
    from src.Semantic.retrieve import IndexHandle
    shutil.rmtree(cfg.dataset.dir() / "csvs")
    h = IndexHandle.open(cfg, "liftus", "default")
    tid = sorted(h.table_to_gids)[0]
    q = sw.QueryRef(tid, None, len(h.table_to_gids[tid]))
    df = sw._query_df(cfg, q)
    assert df.attrs["table_id"] == tid
    assert len(df) == 1
    assert list(df.columns) == [h.gid_to_col_name[g] for g in h.table_to_gids[tid]]


def test_run_hnsw_surface_corpus_gates_k_coarse(cfg, tmp_path, capsys):
    qs = sw.sample_su_queries(cfg, n=1, seed=0)
    out = sw.run_hnsw_surface(cfg, "SU", qs, ef_grid=[16], kc_grid=[10, 100000],
                              out_csv=tmp_path / "h.csv")
    df = pd.read_csv(out)
    assert set(df["k_coarse"]) == {10}
    assert "100000" in capsys.readouterr().out
