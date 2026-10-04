import configparser
import shutil
from pathlib import Path

import pandas as pd
import pytest

from scripts.create_blend_csv_index import ingest_csv_dir
from src.Semantic.config import OperatorConfig, SemanticConfig, SemanticOp
from src.Semantic.index_build import build_semantic_index
from src.Semantic.retrieve import reset_semantic_cache
from tests.conftest import drop_blend_modules

FIXTURE = Path(__file__).parent / "fixtures" / "semantic"
PREFIX = "bench"


def _write_qgt(base: Path, basenames: list[str]) -> None:
    q, gt = base / "query", base / "groundtruth"
    q.mkdir(parents=True, exist_ok=True)
    gt.mkdir(parents=True, exist_ok=True)
    a, b, c = basenames
    pd.DataFrame({"query_table": [a, c]}).to_csv(
        q / f"{PREFIX}_union_query.csv", index=False)
    pd.DataFrame({
        "query_table": [a, a, c, c],
        "candidate_table": [b, c, a, b],
    }).to_csv(gt / f"{PREFIX}_union_ground_truth.csv", index=False)
    col = "course_name"
    pd.DataFrame({"query_table": [a, c], "query_column": [col, col]}).to_csv(
        q / f"{PREFIX}_join_query.csv", index=False)
    pd.DataFrame({
        "query_table": [a, a, c, c],
        "query_column": [col, col, col, col],
        "candidate_table": [b, c, a, b],
        "candidate_column": [col, col, col, col],
    }).to_csv(gt / f"{PREFIX}_join_ground_truth.csv", index=False)


@pytest.fixture()
def bench_fixture(tmp_path, monkeypatch):
    """Self-contained duckdb+faiss dataset with union+join query/GT; yields (cfg, prefix)."""
    root = tmp_path / "data"
    ds_name = "bench_ds"
    ds_dir = root / ds_name
    (ds_dir / "csvs").mkdir(parents=True)
    for p in sorted((FIXTURE / "csvs").glob("*.csv")):
        shutil.copy(p, ds_dir / "csvs" / p.name)

    ingest_csv_dir(ds_dir / "csvs", ds_dir / "blend.duckdb",
                   sidecar_path=ds_dir / "blend_index_basenames.parquet")

    approach_dir = ds_dir / "semantic" / "liftus" / "demo"
    (approach_dir / "ckpt").mkdir(parents=True)
    shutil.copytree(FIXTURE / "ckpt", approach_dir / "ckpt", dirs_exist_ok=True)
    shutil.copytree(FIXTURE / "aspects", approach_dir / "aspects")
    build_cfg = SemanticConfig.load(
        path=tmp_path / "missing.ini",
        overrides={"dataset": ds_name, "dataset_root": str(root)},
        operators={SemanticOp.SU: OperatorConfig("liftus", "demo"),
                   SemanticOp.SJ: OperatorConfig("liftus", "demo")},
    )
    reset_semantic_cache()
    build_semantic_index(FIXTURE / "csvs", "liftus", "demo", build_cfg)

    basenames = sorted(p.name for p in (FIXTURE / "csvs").glob("*.csv"))
    _write_qgt(ds_dir, basenames)

    ini = tmp_path / "config.ini"
    cp = configparser.ConfigParser()
    cp["Dataset"] = {"name": ds_name, "root": str(root)}
    cp["Database"] = {"dbms": "duckdb", "db_filename": "blend.duckdb",
                      "index_table": "blend_index"}
    cp["Semantic"] = {"sidecar_filename": "blend_index_basenames.parquet",
                      "faiss_quant": "flat", "faiss_k_coarse": "50", "device": "cpu"}
    cp["Semantic.SU"] = {"approach": "liftus", "index_name": "demo"}
    cp["Semantic.SJ"] = {"approach": "liftus", "index_name": "demo"}
    with open(ini, "w") as f:
        cp.write(f)
    monkeypatch.setenv("BLEND_CONFIG", str(ini))
    drop_blend_modules()
    reset_semantic_cache()
    yield SemanticConfig.load(), PREFIX
    reset_semantic_cache()
    drop_blend_modules()


def _read_headers(cfg, prefix=None) -> dict[str, list[str]]:
    csvs = cfg.dataset.dir() / "csvs"
    return {
        p.name: pd.read_csv(p, dtype=str, keep_default_na=False, nrows=0).columns.tolist()
        for p in sorted(csvs.glob("*.csv"))
    }


def test_load_union_qgt_filters_by_reg(bench_fixture):
    from src.Benchmark.judit import loading

    cfg, prefix = bench_fixture
    full = {t: set(c) for t, c in _read_headers(cfg, prefix).items()}
    qgt_full = loading.load_union_qgt(cfg, prefix, full)
    assert qgt_full.queries_kept
    for qt in qgt_full.queries_kept:
        assert qt in qgt_full.header_cols

    q0 = qgt_full.queries_kept[0]
    reduced = {t: c for t, c in full.items() if t != q0}
    qgt = loading.load_union_qgt(cfg, prefix, reduced)
    assert q0 not in qgt.queries_kept
    assert set(qgt.queries_kept) <= set(reduced)
    assert qgt.n_dropped_q >= 1


def _oracle_union_hits(cfg, prefix):
    from src.Benchmark.judit import loading

    headers = _read_headers(cfg, prefix)
    reg = {t: set(c) for t, c in headers.items()}
    qgt = loading.load_union_qgt(cfg, prefix, reg)
    hits = {}
    for qt in qgt.queries_kept:
        rel = sorted(qgt.relevant_by_q.get(qt, set()))
        for col in headers[qt]:
            hits[(qt, col)] = (list(rel), [1.0] * len(rel), False)
    return hits


class _FakeUnionBackend:
    method = approach = "fake"
    index_name = "demo"

    def __init__(self, reg, hits):
        self._reg, self._hits = reg, hits

    def available(self):
        return True

    def reg(self):
        return self._reg

    def search_union_column(self, qt, col, kc, kf=None, *, ef=None):
        return self._hits.get((qt, col), ([], [], False))

    def search_join_column(self, qt, qc, kc, kf=None, *, ef=None):
        return ([], [], False)


def test_run_union_leg_records_runtime_and_metrics(bench_fixture):
    from src.Benchmark.judit.seekers import base

    cfg, prefix = bench_fixture
    reg = {qt: set(cols) for qt, cols in _read_headers(cfg, prefix).items()}
    hits = _oracle_union_hits(cfg, prefix)
    be = _FakeUnionBackend(reg, hits)
    leg, rows = base.run_union_leg(be, cfg, prefix, ks=(1, 5), k_coarse=10, k_vote=10)
    assert leg["task"] == "union"
    assert rows and "runtime_ms" in rows[0]
    assert leg["recall_at_k"]["5"] > 0.0


def test_semantic_backend_serves_union_and_join(bench_fixture):
    from src.Benchmark.judit.seekers import base
    from src.Benchmark.judit.seekers.semantic import SemanticBackend

    cfg, prefix = bench_fixture
    be = SemanticBackend(cfg, "liftus", "demo")
    assert be.available() is True
    u_leg, u_rows = base.run_union_leg(be, cfg, prefix, ks=(1, 5), k_coarse=50, k_vote=50)
    j_leg, j_rows = base.run_join_leg(be, cfg, prefix, ks=(1, 5), k_coarse=50, k_vote=50)
    assert u_leg["task"] == "union" and j_leg["task"] == "join"
    assert u_rows and j_rows
    assert "precision_col_at_k" in j_leg


class _DepthRecordingBackend:
    method = approach = "fake"
    index_name = "-"

    def __init__(self, reg):
        self._reg = reg
        self.kcs = []

    def available(self):
        return True

    def reg(self):
        return self._reg

    def search_union_column(self, qt, col, kc, *, ef=None):
        self.kcs.append(kc)
        return [], [], False

    def search_join_column(self, qt, qc, kc, *, ef=None):
        self.kcs.append(kc)
        return [], [], False


def test_join_leg_column_metrics_use_deepest_fetch_whatever_the_k_order(bench_fixture):
    from dataclasses import replace

    from src.Benchmark.judit.seekers import base

    cfg, prefix = bench_fixture
    cfg = replace(cfg, faiss_k_coarse=None)
    reg = {qt: set(cols) for qt, cols in _read_headers(cfg, prefix).items()}

    plans = {k: base._plan(k, None, None, 2.0, cfg) for k in (1, 50)}
    deepest = max(p[0] for p in plans.values())
    assert len({p[0] for p in plans.values()}) == 2, "ks must produce distinct fetches"

    for ks in ((1, 50), (50, 1)):
        be = _DepthRecordingBackend(reg)
        base.run_join_leg(be, cfg, prefix, ks=ks, k_coarse=None, k_vote=None)
        assert be.kcs[-1] == deepest, f"ks={ks} left a k_coarse={be.kcs[-1]} fetch in cols"


def test_semantic_backend_unavailable_for_missing_index(bench_fixture):
    from src.Benchmark.judit.seekers.semantic import SemanticBackend

    cfg, _ = bench_fixture
    assert SemanticBackend(cfg, "deepjoin", "default").available() is False


def test_token_backend_union_and_join(bench_fixture):
    from src.Benchmark.judit.seekers import base
    from src.Benchmark.judit.seekers.token import TokenBackend

    cfg, prefix = bench_fixture
    be = TokenBackend(cfg)
    assert be.available() is True
    try:
        u_leg, u_rows = base.run_union_leg(be, cfg, prefix, ks=(1, 5), k_coarse=50, k_vote=50)
        j_leg, j_rows = base.run_join_leg(be, cfg, prefix, ks=(1, 5), k_coarse=50, k_vote=50)
        assert u_leg["task"] == "union" and u_rows
        assert j_leg["task"] == "join" and j_rows
    finally:
        be.close()


def test_sho_backend_unavailable_without_codes(bench_fixture):
    from src.Benchmark.judit.seekers.sho import ShoBackend

    cfg, _ = bench_fixture
    assert ShoBackend(cfg).available() is False


def test_tiebreak_bench_runs_the_sj_leg(bench_fixture):
    import json
    from src.Benchmark.tiebreak.bench import run_tiebreak_bench
    from src.Optimizer.semantic_sweep import _FakeDB
    from src.optimizer_paths import semantic_cost_model_path

    cfg, prefix = bench_fixture
    model = semantic_cost_model_path(cfg.dataset.dir(), "duckdb-faiss-single")
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_text(json.dumps({
        op: {"plan_label": "faiss", "hnsw_per_col_ms": 0.1,
             "unfiltered": {"form": form,
                            "coef": ({"a": 0.5, "c_kc": 0.01} if op == "SJ" else
                                     {"a": 0.5, "b_ncols": 0.02, "c_kc": 0.01, "d": 0.001})}}
        for op, form in (("SU", "su_bilinear"), ("SJ", "sj_linear_kc"))}))

    rows, summary = run_tiebreak_bench(
        cfg.dataset.name, k=5, k_coarse_grid=[10, 20], queries=[], db=_FakeDB(),
        repeats=1, seed=0, sample_queries=4, max_pairs=50, op="SJ")

    assert summary["op"] == "SJ"
    assert rows, "SJ leg produced no pairs"
    assert summary["placeholder_ties"] == len(rows)
    assert any(r["new"] != "tie" for r in rows)


def test_query_legs_are_streamed_not_materialised(bench_fixture):
    import inspect
    from src.Benchmark.tiebreak import bench

    assert inspect.isgeneratorfunction(bench._iter_query_items)

    cfg, _ = bench_fixture
    skipped = {"missing_csv": 0, "unusable_query": 0}
    gen = bench._iter_query_items(cfg, cfg.dataset.name, [], 3, 0, "SJ", skipped)
    first = next(gen)
    assert first[1].shape[1] == 1, "an SJ leg is exactly one column"
    gen.close()
