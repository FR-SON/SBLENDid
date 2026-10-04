from src.Optimizer import cli


def test_parser_has_all_subcommands():
    parser = cli.build_parser()
    sub = parser.parse_args(["build-freqs", "--lake", "L"])
    assert sub.cmd == "build-freqs" and sub.lake == "L"
    for cmd in ("sample", "train", "eval"):
        ns = parser.parse_args([cmd, "--lake", "L"])
        assert ns.cmd == cmd
        assert ns.lake == "L"


def test_sample_defaults():
    ns = cli.build_parser().parse_args(["sample", "--lake", "L"])
    assert ns.n == 1000 and ns.seed == 0


def test_train_repeats_and_seeker_types():
    p = cli.build_parser()
    ns = p.parse_args(["train", "--lake", "L"])
    assert ns.repeats == 3
    assert ns.seeker_types == ["SC", "Keyword", "C", "MC"]
    ns = p.parse_args(["train", "--lake", "L", "--repeats", "1", "--seeker-types", "C", "MC"])
    assert ns.repeats == 1 and ns.seeker_types == ["C", "MC"]


def test_sample_seeker_types_subset():
    ns = cli.build_parser().parse_args(["sample", "--lake", "L", "--seeker-types", "C", "--n", "200"])
    assert ns.seeker_types == ["C"] and ns.n == 200


def test_measure_and_sweep_subcommands_parse():
    p = cli.build_parser()
    ns = p.parse_args(["measure", "--lake", "santos"])
    assert ns.cmd == "measure" and ns.lake == "santos"
    ns = p.parse_args(["sweep-semantic", "--lake", "santos", "--ops", "SU", "SJ"])
    assert ns.cmd == "sweep-semantic" and ns.ops == ["SU", "SJ"]
    assert ns.ef_grid == [16, 32, 64, 128, 256]
    assert ns.kc_grid == [100, 500, 1000]


def test_report_subcommand_parses():
    ns = cli.build_parser().parse_args(["report", "--lake", "santos"])
    assert ns.cmd == "report" and ns.lake == "santos"
    assert ns.ef == 64 and ns.kc == 500


def _sweep_calls(monkeypatch, backend):
    from src.Semantic.config import SemanticConfig
    from src.Optimizer import semantic_sweep as ssw

    class _Cfg:
        vector_backend = backend
    calls = {"hnsw": 0, "exact": 0, "filtered": 0}
    monkeypatch.setattr(SemanticConfig, "load", lambda **kw: _Cfg())
    monkeypatch.setattr(cli, "_profile_of", lambda lake: f"{backend}-x-single")
    monkeypatch.setattr(ssw, "sample_su_queries", lambda *a, **k: [])
    monkeypatch.setattr(ssw, "sample_sj_queries", lambda *a, **k: [])
    monkeypatch.setattr(ssw, "run_hnsw_surface",
                        lambda *a, **k: calls.__setitem__("hnsw", calls["hnsw"] + 1))
    monkeypatch.setattr(ssw, "run_exact_curve",
                        lambda *a, **k: calls.__setitem__("exact", calls["exact"] + 1))
    monkeypatch.setattr(ssw, "run_filtered_curve",
                        lambda *a, **k: calls.__setitem__("filtered", calls["filtered"] + 1))
    cli.main(["sweep-semantic", "--lake", "x", "--ops", "SU"])
    return calls


def test_sweep_faiss_runs_exact_curve(monkeypatch):
    assert _sweep_calls(monkeypatch, "faiss") == {"hnsw": 1, "exact": 1, "filtered": 0}


def test_sweep_pgvector_skips_exact_curve(monkeypatch):
    assert _sweep_calls(monkeypatch, "pgvector") == {"hnsw": 1, "exact": 0, "filtered": 1}
