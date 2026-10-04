import pytest

from src.Optimizer import cli
from src.Optimizer.materialize import require_duckdb_slice, warn_rowid_slice


def test_require_duckdb_slice_rejects_postgres():
    with pytest.raises(SystemExit) as e:
        require_duckdb_slice("postgres", "build-slice")
    msg = str(e.value)
    assert "build-slice" in msg and "duckdb" in msg and "postgres" in msg


def test_require_duckdb_slice_allows_duckdb():
    assert require_duckdb_slice("duckdb", "build-slice") is None


def test_warn_names_the_stage_and_goes_to_stderr(capsys):
    warn_rowid_slice("train")
    err = capsys.readouterr().err
    assert "[OPTIMIZER][ROWID-SLICE]" in err and "train" in err


def test_build_slice_rejects_postgres(monkeypatch):
    monkeypatch.setattr(cli, "lake_dbms", lambda: "postgres")
    monkeypatch.setattr(cli, "build_rowid_slice",
                        lambda *a, **kw: pytest.fail("must not build under pg"))
    with pytest.raises(SystemExit):
        cli.main(["build-slice", "--lake", "L"])


def test_build_slice_warns_on_duckdb(monkeypatch, capsys):
    monkeypatch.setattr(cli, "lake_dbms", lambda: "duckdb")
    monkeypatch.setattr(cli, "lake_duckdb_path", lambda lake: "src.duckdb")
    monkeypatch.setattr(cli, "lake_rowid_slice_path", lambda lake: "slice.duckdb")
    monkeypatch.setattr(cli, "build_rowid_slice", lambda src, out: out)
    cli.main(["build-slice", "--lake", "L"])
    assert "[OPTIMIZER][ROWID-SLICE]" in capsys.readouterr().err


def test_train_rowid_slice_rejects_postgres_before_connecting(monkeypatch):
    monkeypatch.setattr(cli, "lake_dbms", lambda: "postgres")
    monkeypatch.setattr(cli, "use_lake",
                        lambda *a, **kw: pytest.fail("must not connect under pg"))
    with pytest.raises(SystemExit):
        cli.main(["train", "--lake", "L", "--rowid-slice"])


def test_calibrate_rejects_postgres_before_any_stage(monkeypatch):
    monkeypatch.setattr(cli, "lake_dbms", lambda: "postgres")
    monkeypatch.setattr(cli, "_run_pipeline",
                        lambda *a, **kw: pytest.fail("must not run stages under pg"))
    with pytest.raises(SystemExit):
        cli.main(["calibrate", "--lake", "L", "--rowid-slice"])


def test_train_without_slice_flag_is_unguarded(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "lake_dbms", lambda: "postgres")
    monkeypatch.setattr(cli, "load_freqs", lambda p: None)
    monkeypatch.setattr(cli, "freqs_path", lambda d: "f.csv")
    monkeypatch.setattr(cli, "lake_dir", lambda lake: "d")
    monkeypatch.setattr(cli, "lake_csv_dir", lambda lake: "csvs")
    monkeypatch.setattr(cli, "use_lake",
                        lambda lake: type("DB", (), {"optimizer_profile_str": lambda s: "p"})())
    monkeypatch.setattr("src.Optimizer.training.train_all",
                        lambda *a, **kw: seen.setdefault("type_db", kw["type_db"]) or {})
    cli.main(["train", "--lake", "L"])
    assert seen["type_db"] is None
