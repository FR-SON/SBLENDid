from src.Optimizer.cli import build_parser, _calibrate_stages, _run_pipeline


def _args(argv):
    return build_parser().parse_args(argv)


def test_calibrate_parses_defaults():
    a = _args(["calibrate", "--lake", "santos"])
    assert a.cmd == "calibrate" and a.lake == "santos"
    assert a.k == 10 and a.seed == 0
    assert a.skip == [] and a.rowid_slice is False and a.no_warmup is False


def test_full_plan_order_excludes_build_slice_by_default():
    a = _args(["calibrate", "--lake", "santos"])
    names = [n for n, _ in _calibrate_stages(a)]
    assert names == ["build-freqs", "sample", "train", "measure",
                     "sweep-semantic", "fit-semantic-cost", "report"]


def test_rowid_slice_adds_build_slice_and_train_flag():
    a = _args(["calibrate", "--lake", "santos", "--rowid-slice"])
    stages = _calibrate_stages(a)
    names = [n for n, _ in stages]
    assert names[:3] == ["build-freqs", "build-slice", "sample"]
    assert "--rowid-slice" in dict(stages)["train"]


def test_skip_omits_stages_keeps_order():
    a = _args(["calibrate", "--lake", "santos",
               "--skip", "sweep-semantic", "fit-semantic-cost"])
    names = [n for n, _ in _calibrate_stages(a)]
    assert names == ["build-freqs", "sample", "train", "measure", "report"]


def test_no_warmup_propagates_to_train_and_measure():
    a = _args(["calibrate", "--lake", "santos", "--no-warmup"])
    stages = dict(_calibrate_stages(a))
    assert "--no-warmup" in stages["train"]
    assert "--no-warmup" in stages["measure"]


def test_seed_propagates_to_sample_and_sweep():
    a = _args(["calibrate", "--lake", "santos", "--seed", "7"])
    stages = dict(_calibrate_stages(a))
    assert stages["sample"][:5] == ["sample", "--lake", "santos", "--seed", "7"]
    assert "--seed" in stages["sweep-semantic"] and "7" in stages["sweep-semantic"]


def test_report_is_last_and_writes():
    a = _args(["calibrate", "--lake", "santos"])
    stages = _calibrate_stages(a)
    assert stages[-1] == ("report", ["report", "--lake", "santos", "--write",
                                    "--ef", "64", "--kc", "500"])


def test_run_pipeline_invokes_runner_in_order():
    a = _args(["calibrate", "--lake", "santos",
               "--skip", "build-freqs", "sample", "train", "measure",
               "sweep-semantic", "fit-semantic-cost"])
    calls = []
    _run_pipeline(_calibrate_stages(a), runner=lambda argv: calls.append(argv))
    assert calls == [["report", "--lake", "santos", "--write", "--ef", "64", "--kc", "500"]]
