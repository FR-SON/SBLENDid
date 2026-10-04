from src.Benchmark.cli import build_parser


def test_parser_run_union():
    args = build_parser().parse_args(["run", "--dataset", "santos", "--task", "union", "--k", "10"])
    assert args.cmd == "run" and args.task == "union" and args.k == 10


def test_parser_requires_task():
    import pytest
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "--dataset", "santos"])


def test_parser_compare():
    from src.Benchmark.cli import build_parser
    args = build_parser().parse_args(
        ["compare", "--dataset", "santos", "--task", "imputation", "--k-grid", "10,50"])
    assert args.cmd == "compare" and args.k_grid == [10, 50]


def test_parser_tiebreak_bench():
    args = build_parser().parse_args(
        ["tiebreak-bench", "--dataset", "santos", "--k-coarse-grid", "100,500,1000"])
    assert args.cmd == "tiebreak-bench"
    assert args.k_coarse_grid == [100, 500, 1000]
    assert args.k == 10 and args.repeats == 3


def test_tiebreak_parser_sample_queries():
    from src.Benchmark.cli import build_parser
    p = build_parser()
    assert p.parse_args(["tiebreak-bench", "--dataset", "x"]).sample_queries is None
    assert p.parse_args(
        ["tiebreak-bench", "--dataset", "x", "--sample-queries", "7"]
    ).sample_queries == 7
