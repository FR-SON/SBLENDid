from src.Optimizer.cli import build_parser


def test_fit_semantic_cost_subcommand_parses():
    args = build_parser().parse_args(
        ["fit-semantic-cost", "--lake", "santos", "--ops", "SU", "--ef", "64", "--kc", "500"]
    )
    assert args.cmd == "fit-semantic-cost"
    assert args.lake == "santos"
    assert args.ops == ["SU"]
    assert args.ef == 64 and args.kc == 500


def test_fit_semantic_cost_defaults():
    args = build_parser().parse_args(["fit-semantic-cost", "--lake", "santos"])
    assert args.ops == ["SU", "SJ"]
    assert args.ef == 64 and args.kc == 500
