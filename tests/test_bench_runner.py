import json

import pandas as pd

from src.Benchmark.judit import runner


def test_median_tree_nested():
    a = {"precision_at_k": {"1": 1.0}, "wall_clock_seconds": 10.0, "task": "union"}
    b = {"precision_at_k": {"1": 0.0}, "wall_clock_seconds": 20.0, "task": "union"}
    m = runner.median_tree([a, b])
    assert m["precision_at_k"]["1"] == 0.5
    assert m["wall_clock_seconds"] == 15.0
    assert m["task"] == "union"


def test_median_tree_passes_none_through():
    a = {"x": None, "y": 1.0}
    b = {"x": None, "y": 3.0}
    m = runner.median_tree([a, b])
    assert m["x"] is None and m["y"] == 2.0


def test_run_matrix_writes_outputs_and_skips(bench_fixture, tmp_path):
    cfg, prefix = bench_fixture
    out = runner.run_matrix(
        cfg, prefix, tasks=["union", "join"], methods=["liftus:demo", "sho"],
        repeat=2, ks=(1, 5), k_coarse=50, k_vote=50, out_dir=tmp_path / "run")
    agg = json.loads((out / "aggregate.json").read_text())
    labels = {(l["task"], l["method"]) for l in agg["legs"]}
    assert ("union", "liftus") in labels and ("join", "liftus") in labels
    assert any(s["method"] == "sho" for s in agg["skipped"])

    for leg in agg["legs"]:
        assert "runtime_ms_per_query" in leg["median"]
        assert leg["median"]["precision_at_k"]["1"] is not None

    per_repeat = json.loads((out / "per_repeat.json").read_text())
    assert all(len(v) == 2 for v in per_repeat.values())

    pq = pd.read_parquet(out / "per_query.parquet")
    assert pq["repeat"].nunique() == 2 and len(pq) > 0
    assert "runtime_ms" in pq.columns


def test_run_matrix_surfaces_leg_failure(bench_fixture, tmp_path, monkeypatch):
    from src.Benchmark.judit.seekers import make_backend as real_make

    class _Boom:
        method = approach = "boom"
        index_name = "default"

        def available(self):
            return True

        def reg(self):
            raise RuntimeError("kaboom")

    monkeypatch.setattr(
        runner, "make_backend",
        lambda m, c: _Boom() if m == "boom" else real_make(m, c))
    cfg, prefix = bench_fixture
    out = runner.run_matrix(
        cfg, prefix, tasks=["union"], methods=["boom", "liftus:demo"],
        repeat=1, ks=(1,), k_coarse=10, k_vote=10, out_dir=tmp_path / "run")
    agg = json.loads((out / "aggregate.json").read_text())
    assert [f["method"] for f in agg["failed"]] == ["boom"]
    assert [l["method"] for l in agg["legs"]] == ["liftus"]
    pq = pd.read_parquet(out / "per_query.parquet")
    assert set(pq["method"]) == {"liftus"}


def _stub(rank_rollup):
    class _B:
        uses_rank_rollup = rank_rollup
    return _B()


def test_leg_vote_factor_only_reaches_rank_rollup_backends(tmp_path):
    from src.Semantic.config import SemanticConfig, SemanticOp

    ini = tmp_path / "config.ini"
    ini.write_text(
        "[Semantic]\n"
        "vote_depth_factor = 3.0\n"
        "[Semantic.SU]\n"
        "approach = liftus\n"
        "index_name = default\n"
        "vote_depth_factor = 5.0\n"
        "[Semantic.SJ]\n"
        "approach = snoopy\n"
        "index_name = default\n"
    )
    cfg = SemanticConfig.load(path=ini)
    assert cfg.vote_factor_for(SemanticOp.SU) == 5.0

    sem, exact = _stub(True), _stub(False)
    assert runner._leg_vote_factor(cfg, "union", sem, None) == 5.0
    assert runner._leg_vote_factor(cfg, "join", sem, None) == 3.0
    assert runner._leg_vote_factor(cfg, "union", sem, 1.5) == 1.5
    assert runner._leg_vote_factor(cfg, "union", exact, None) == 2.0
    assert runner._leg_vote_factor(cfg, "union", exact, 1.5) == 2.0


def test_run_matrix_vote_factor_reaches_derived_depths(bench_fixture, tmp_path):
    from dataclasses import replace
    from types import MappingProxyType

    from src.Semantic.config import OperatorConfig, SemanticOp

    cfg, prefix = bench_fixture
    cfg = replace(cfg, faiss_k_coarse=None, operators=MappingProxyType({
        SemanticOp.SU: OperatorConfig("liftus", "demo", vote_depth_factor=5.0),
        SemanticOp.SJ: OperatorConfig("liftus", "demo", vote_depth_factor=3.0),
    }))
    out = runner.run_matrix(
        cfg, prefix, tasks=["union", "join"], methods=["liftus:demo", "sc"],
        repeat=1, ks=(10,), k_coarse=None, k_vote=None, out_dir=tmp_path / "run")
    agg = json.loads((out / "aggregate.json").read_text())
    assert agg["vote_factor"] is None
    by = {(l["task"], l["method"]): l for l in agg["legs"]}

    assert by[("union", "liftus")]["vote_factor"] == 5.0
    assert by[("join", "liftus")]["vote_factor"] == 3.0
    assert by[("union", "sc")]["vote_factor"] == 2.0

    def kv(leg):
        return list(leg["median"]["k_vote_used"].values())[0]

    assert kv(by[("union", "liftus")]) == 50
    assert kv(by[("join", "liftus")]) == 30
    assert kv(by[("union", "sc")]) == 20
