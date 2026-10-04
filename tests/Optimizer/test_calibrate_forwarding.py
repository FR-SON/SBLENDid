import pytest

from src.Optimizer import cli

FORWARDED = [
    ("n", "sample"), ("max_cat_card", "sample"), ("seeker_types", "sample"),
    ("repeats", "measure"), ("statement_timeout", "measure"),
    ("n_su", "sweep-semantic"), ("n_sj", "sweep-semantic"), ("ops", "sweep-semantic"),
    ("sweep_repeats", "sweep-semantic", "repeats"),
    ("ef", "report"), ("kc", "report"),
]


def _subparsers():
    p = cli.build_parser()
    return next(a.choices for a in p._subparsers._group_actions if a.choices)


def _ns(*extra):
    return cli.build_parser().parse_args(["calibrate", "--lake", "L", *extra])


def test_parse_n_type_maps_pairs():
    assert cli.parse_n_type(["C=50", "MC=25"]) == {"C": 50, "MC": 25}
    assert cli.parse_n_type([]) == {}


def test_parse_n_type_rejects_bad_input():
    for bad in (["C"], ["C=x"], ["Nope=5"]):
        with pytest.raises(SystemExit):
            cli.parse_n_type(bad)


def test_sample_uses_per_type_n(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(cli, "_runs_base", lambda: tmp_path)
    monkeypatch.setattr(cli, "lake_csv_dir", lambda lake: tmp_path)
    def _rec(d, types, *, n_by_type, n_default, seed, max_cat_card):
        seen.update({t: n_by_type.get(t, n_default) for t in types})
        return {t: [] for t in types}
    monkeypatch.setattr(cli.smp, "sample_many", _rec)
    monkeypatch.setattr(cli.smp, "save_specs", lambda specs, path: None)
    cli.main(["sample", "--lake", "L", "--n", "500", "--n-type", "C=50", "MC=25"])
    assert seen == {"SC": 500, "Keyword": 500, "C": 50, "MC": 25}


def test_sample_rejects_n_type_outside_seeker_types(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_runs_base", lambda: tmp_path)
    monkeypatch.setattr(cli, "lake_csv_dir", lambda lake: tmp_path)
    with pytest.raises(SystemExit):
        cli.main(["sample", "--lake", "L", "--seeker-types", "SC", "--n-type", "C=50"])


def _argv(stages, name):
    return next(a for n, a in stages if n == name)


def test_calibrate_forwards_stage_knobs():
    stages = cli._calibrate_stages(_ns(
        "--n", "500", "--n-type", "C=50", "MC=25", "--repeats", "5",
        "--statement-timeout", "60", "--n-su", "300", "--n-sj", "400",
        "--sweep-repeats", "3", "--ef", "128", "--kc", "1000"))
    s = _argv(stages, "sample")
    assert s[s.index("--n") + 1] == "500"
    assert s[s.index("--n-type") + 1:s.index("--n-type") + 3] == ["C=50", "MC=25"]
    m = _argv(stages, "measure")
    assert m[m.index("--repeats") + 1] == "5"
    assert m[m.index("--statement-timeout") + 1] == "60"
    w = _argv(stages, "sweep-semantic")
    assert w[w.index("--n-su") + 1] == "300" and w[w.index("--n-sj") + 1] == "400"
    assert w[w.index("--repeats") + 1] == "3"
    assert _argv(stages, "measure")[_argv(stages, "measure").index("--repeats") + 1] == "5"
    for name in ("report", "fit-semantic-cost"):
        a = _argv(stages, name)
        assert a[a.index("--ef") + 1] == "128" and a[a.index("--kc") + 1] == "1000"
    assert "--write" in _argv(stages, "report")


def test_calibrate_omits_n_type_when_unset():
    assert "--n-type" not in _argv(cli._calibrate_stages(_ns()), "sample")


def test_every_generated_stage_argv_parses():
    stages = cli._calibrate_stages(_ns("--rowid-slice", "--no-warmup",
                                       "--n-type", "C=50"))
    assert {n for n, _ in stages} == set(cli._CALIBRATE_ORDER)
    for name, argv in stages:
        assert cli.build_parser().parse_args(argv).cmd == name


def test_calibrate_defaults_match_the_stages():
    subs = _subparsers()
    cal = _ns()
    for entry in FORWARDED:
        dest, stage = entry[0], entry[1]
        stage_dest = entry[2] if len(entry) > 2 else dest
        assert getattr(cal, dest) == subs[stage].get_default(stage_dest), dest
