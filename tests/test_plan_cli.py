import pytest

from src.Benchmark.cli import build_parser


def test_plan_list_parses():
    a = build_parser().parse_args(["plan", "list"])
    assert a.cmd == "plan" and a.plan_cmd == "list"


def test_plan_run_parses_required_and_defaults():
    a = build_parser().parse_args(
        ["plan", "run", "semantic_oracle_repair",
         "--dataset", "opendata-split-11", "--task", "union", "--k", "10"])
    assert a.cmd == "plan" and a.plan_cmd == "run"
    assert a.name == "semantic_oracle_repair"
    assert a.dataset == "opendata-split-11" and a.task == "union" and a.k == 10
    assert a.seed == 0 and a.limit is None and a.out is None


def test_plan_run_parses_optionals():
    a = build_parser().parse_args(
        ["plan", "run", "p", "--dataset", "d", "--task", "join", "--k", "5",
         "--seed", "7", "--limit", "3", "--out", "/tmp/x"])
    assert a.task == "join" and a.seed == 7 and a.limit == 3 and a.out == "/tmp/x"


def test_plan_run_requires_task():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["plan", "run", "p", "--dataset", "d", "--k", "10"])


def test_git_sha_returns_str():
    from src.Benchmark.runspec import git_sha
    assert isinstance(git_sha(), str)
