class _CfgStub:
    class dataset:
        name = "opendata-lake"
    vector_backend = "faiss"


def test_bench_dispatch_runs_matrix(monkeypatch, tmp_path):
    # patch via module objects: drop_blend_modules can strip submodule attrs from `src`
    from src.Benchmark import cli
    from src.Benchmark.judit import runner
    from src.Semantic.config import SemanticConfig

    called = {}

    def fake_run_matrix(cfg, prefix, **kw):
        called.update(prefix=prefix, **kw)
        return tmp_path

    monkeypatch.setattr(runner, "run_matrix", fake_run_matrix)
    monkeypatch.setattr(SemanticConfig, "load", staticmethod(lambda **_: _CfgStub()))
    rc = cli.main(["bench", "--dataset", "opendata", "--tasks", "union",
                   "--seekers", "sc,liftus", "--repeat", "3", "--k", "1,25"])
    assert rc == 0
    assert called["prefix"] == "opendata"
    assert called["tasks"] == ["union"]
    assert called["methods"] == ["sc", "liftus"]
    assert called["repeat"] == 3 and called["ks"] == (1, 25)


def test_bench_cli_smoke(bench_fixture, tmp_path):
    from src.Benchmark import cli

    cfg, prefix = bench_fixture
    rc = cli.main(["bench", "--dataset", prefix, "--tasks", "union",
                   "--seekers", "liftus:demo", "--repeat", "1", "--k", "1,5",
                   "--out", str(tmp_path / "o")])
    assert rc == 0
    assert (tmp_path / "o" / "aggregate.json").exists()


def test_bench_cli_all_skipped_nonzero(bench_fixture, tmp_path):
    from src.Benchmark import cli

    cfg, prefix = bench_fixture
    rc = cli.main(["bench", "--dataset", prefix, "--tasks", "union",
                   "--seekers", "sho", "--repeat", "1", "--k", "1,5",
                   "--out", str(tmp_path / "o")])
    assert rc == 1
    assert (tmp_path / "o" / "aggregate.json").exists()


def test_bench_vote_factor_flag_and_slug(monkeypatch, tmp_path):
    from src.Benchmark import cli
    from src.Benchmark.judit import runner
    from src.Semantic.config import SemanticConfig

    called = {}
    monkeypatch.setattr(runner, "run_matrix",
                        lambda cfg, prefix, **kw: (called.update(kw), tmp_path)[1])
    monkeypatch.setattr(SemanticConfig, "load", staticmethod(lambda **_: _CfgStub()))

    rc = cli.main(["bench", "--dataset", "opendata", "--tasks", "union",
                   "--seekers", "liftus", "--vote-factor", "5"])
    assert rc == 0 and called["vote_factor"] == 5.0

    called.clear()
    rc = cli.main(["bench", "--dataset", "opendata", "--tasks", "union",
                   "--seekers", "liftus"])
    assert rc == 0 and called["vote_factor"] is None


def test_bench_vote_factor_absent_from_slug_by_default():
    from src.Benchmark.runspec import Field, render_slug

    fields = (Field("vote_factor", prefix="vf", default=None),)
    assert render_slug(fields, {"vote_factor": None}) == ""
    assert render_slug(fields, {"vote_factor": 5.0}) == "vf5p0"


def test_bench_vote_factor_rejects_non_positive(capsys):
    import pytest

    from src.Benchmark import cli

    for bad in ("0", "-2"):
        with pytest.raises(SystemExit):
            cli.main(["bench", "--dataset", "d", "--vote-factor", bad])
        assert "must be > 0" in capsys.readouterr().err
