import configparser
import time
from pathlib import Path

import pandas as pd
import pytest

from scripts.create_blend_csv_index import ingest_csv_dir
from src.DBHandler import DBHandler
from src.Optimizer import training as tr, sampling as smp
from src.Optimizer.db import open_lake_db, CONFIG_PATH


@pytest.fixture(autouse=True)
def _ml_off(monkeypatch):
    monkeypatch.setattr(DBHandler, "USE_ML_OPTIMIZER", False)


def test_build_seeker_per_type():
    from src.Operators.Seekers.SingleColumnOverlap import SingleColumnOverlap
    from src.Operators.Seekers.MultiColumnOverlap import MultiColumnOverlap
    from src.Operators.Seekers.Correlation import Correlation
    from src.Operators.Seekers.Keyword import Keyword
    assert isinstance(tr.build_seeker("SC", [["a", "b"]], 10), SingleColumnOverlap)
    assert isinstance(tr.build_seeker("Keyword", [["a", "b"]], 10), Keyword)
    assert isinstance(tr.build_seeker("C", [["a", "b"], ["1", "2"]], 10), Correlation)
    assert isinstance(tr.build_seeker("MC", [["a", "b"], ["c", "d"]], 10), MultiColumnOverlap)


def test_cost_columns_matches_ml_cost_inputs(monkeypatch):
    for stype, cols in [
        ("SC", [["a", "b", "a"]]),
        ("Keyword", [["a", "b"]]),
        ("C", [["x", "y", "x"], ["1", "2", "3"]]),
        ("MC", [["a", "b"], ["c", "d"]]),
    ]:
        seeker = tr.build_seeker(stype, cols, 10)
        captured = {}
        monkeypatch.setattr(
            type(seeker), "_predict_runtime",
            lambda self, columns, db: captured.setdefault("cols", columns) or 0.0,
        )
        seeker._cached_predicted_runtime = None
        seeker.ml_cost(db=None)
        assert tr.cost_columns(stype, seeker) == captured["cols"]


def test_model_path_points_at_profile_dir(tmp_path):
    p = tr.model_path("SC", tmp_path, "duckdb-faiss-single")
    assert p.name == "SingleColumnOverlap_model.json"
    assert p.parent.as_posix().endswith("optimizer/duckdb-faiss-single/models")


def test_train_and_dump_roundtrip(tmp_path):
    rows = [([float(i), float(i * 2), 1.0], float(i)) for i in range(1, 20)]
    model = tr.train_model(rows)
    pred = model.predict([[5.0, 10.0, 1.0]])
    assert pred.shape == (1,)
    out = tr.dump_model(model, "SC", tmp_path, "duckdb-faiss-single")
    assert out.exists()
    assert out == tmp_path / "optimizer" / "duckdb-faiss-single" / "models" / "SingleColumnOverlap_model.json"


def test_measure_single_seekers_writes_runtimes(tmp_path, monkeypatch):
    monkeypatch.setattr(DBHandler, "USE_ML_OPTIMIZER", False)

    FIXTURE = Path(__file__).parent.parent / "fixtures" / "semantic" / "csvs"

    ds = tmp_path / "datasets" / "testlake"
    (ds / "csvs").mkdir(parents=True)
    for p in FIXTURE.glob("*.csv"):
        (ds / "csvs" / p.name).write_bytes(p.read_bytes())
    ingest_csv_dir(csv_dir=ds / "csvs", duckdb_path=ds / "blend.duckdb")

    cp = configparser.ConfigParser()
    cp.read(CONFIG_PATH)
    cp["Dataset"]["name"] = "testlake"
    cp["Dataset"]["root"] = str(tmp_path / "datasets")
    cp["Database"]["dbms"] = "duckdb"
    tmp_cfg = tmp_path / "config.ini"
    with open(tmp_cfg, "w") as f:
        cp.write(f)

    db = open_lake_db("testlake", config_path=tmp_cfg)
    specs = smp.sample_queries(ds / "csvs", "SC", n=3, seed=0)
    smp.save_specs(specs, tmp_path / "samples_SC.jsonl")

    out_dir = tmp_path / "measure"
    paths = tr.measure_single_seekers(db, ds / "csvs", out_dir,
                                      seeker_types=["SC"], repeats=1, warmup=False,
                                      samples_dir=tmp_path)
    sc_csv = out_dir / "single_seeker_runtimes_SC.csv"
    assert paths == [sc_csv]
    df = pd.read_csv(sc_csv)
    assert list(df.columns) == ["seeker_type", "csv", "cols", "runtime_s", "outcome"]
    assert (df["outcome"] == "ok").all()
    assert (df["runtime_s"] > 0).all() and len(df) >= 1

    kw_specs = smp.sample_queries(ds / "csvs", "Keyword", n=3, seed=0)
    smp.save_specs(kw_specs, tmp_path / "samples_Keyword.jsonl")
    tr.measure_single_seekers(db, ds / "csvs", out_dir, seeker_types=["Keyword"],
                              repeats=1, warmup=False, samples_dir=tmp_path)
    assert sc_csv.exists(), "second measure run clobbered the first type's CSV"
    assert len(pd.read_csv(sc_csv)) >= 1
    assert (out_dir / "single_seeker_runtimes_Keyword.csv").exists()


_SLOW_SQL = "SELECT count(*) FROM range(40000000000) t(i) WHERE i % 7 = 0"


class _DuckStub:
    """Mirrors the DBHandler surface: duckdb's cursor() is a separate connection."""
    dbms = "duckdb"

    def __init__(self):
        import duckdb
        self.connection = duckdb.connect()
        self.cursor = self.connection.cursor()


def test_duckdb_query_is_cancelled_at_the_cap():
    db = _DuckStub()
    with pytest.raises(tr.QueryTimeout):
        with tr.query_deadline(db, 1):
            db.connection.execute(_SLOW_SQL).fetchall()
    assert db.connection.execute("SELECT 42").fetchone() == (42,)


def test_a_query_on_the_cursor_is_cancelled_too():
    """duckdb's cursor is a separate connection; interrupting only the parent leaves it running."""
    db = _DuckStub()
    with pytest.raises(tr.QueryTimeout):
        with tr.query_deadline(db, 1):
            db.cursor.execute(_SLOW_SQL).fetchall()
    assert db.cursor.execute("SELECT 42").fetchone() == (42,)
    assert db.connection.execute("SELECT 7").fetchone() == (7,)


def test_deadline_does_not_disturb_a_query_under_the_cap():
    db = _DuckStub()
    with tr.query_deadline(db, 30):
        assert db.connection.execute("SELECT 1").fetchone() == (1,)
    assert db.connection.execute("SELECT 2").fetchone() == (2,)


def test_a_real_error_is_not_relabelled_as_a_timeout():
    db = _DuckStub()
    with pytest.raises(Exception) as e:
        with tr.query_deadline(db, 30):
            db.connection.execute("SELECT * FROM does_not_exist").fetchall()
    assert not isinstance(e.value, tr.QueryTimeout)


def test_deadline_is_a_noop_off_duckdb_and_when_disabled():
    class _PG:
        dbms = "postgres"
        connection = None
    with tr.query_deadline(_PG(), 1):
        pass
    with tr.query_deadline(_DuckStub(), 0):
        pass


def test_measure_runtime_caps_every_execution():
    """The cap applies per execution, matching postgres per-statement semantics."""
    db = _DuckStub()
    seen = []

    class _Seeker:
        def run(self):
            seen.append(1)
            db.cursor.execute(_SLOW_SQL).fetchall()

    with pytest.raises(tr.QueryTimeout):
        tr.measure_runtime(_Seeker(), repeats=3, warmup=True, db=db, timeout=1)
    assert seen == [1]


def test_apply_statement_timeout_still_skips_duckdb():
    tr.apply_statement_timeout(_DuckStub(), 5)


def test_a_pure_python_loop_is_cancelled_too():
    """MultiColumnOverlap's run_filter runs in Python after the SQL returns."""
    db = _DuckStub()
    t0 = time.perf_counter()
    with pytest.raises(tr.QueryTimeout):
        with tr.query_deadline(db, 1):
            acc = 0
            for a in range(10 ** 9):
                for b in range(10):
                    acc += a * b
    assert time.perf_counter() - t0 < 10


def test_no_async_exception_leaks_past_the_block():
    db = _DuckStub()
    with pytest.raises(tr.QueryTimeout):
        with tr.query_deadline(db, 1):
            while True:
                pass
    time.sleep(1.5)
    total = sum(range(10000))
    assert total == 49995000
    assert db.cursor.execute("SELECT 1").fetchone() == (1,)


def test_work_finishing_after_the_cap_still_counts_as_over_budget():
    """Fired but the work returned before the exception landed."""
    db = _DuckStub()
    with pytest.raises(tr.QueryTimeout):
        with tr.query_deadline(db, 1):
            time.sleep(2)


class _QueryCanceled(Exception):
    """Stands in for psycopg.errors.QueryCanceled, matched by type name."""


def test_is_timeout_recognises_both_caps():
    assert tr.is_timeout(tr.QueryTimeout("cancelled after 60s wall cap"))
    assert tr.is_timeout(_QueryCanceled("canceling statement due to statement timeout"))
    assert tr.is_timeout(RuntimeError("canceling statement due to statement timeout"))


def test_is_timeout_rejects_a_genuine_error():
    assert not tr.is_timeout(KeyError("no such column"))
    assert not tr.is_timeout(ValueError("could not parse"))


class _PgStub:
    """Postgres handle: no interrupt() — the engine cancels its own statement."""
    dbms = "postgres"

    def __init__(self):
        self.interrupts = 0

    def interrupt(self):
        self.interrupts += 1


def _burn(seconds):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


def test_wall_cap_stops_a_python_loop_on_postgres():
    db = _PgStub()
    t0 = time.perf_counter()
    with pytest.raises(tr.QueryTimeout):
        with tr.query_deadline(db, 0.3):
            _burn(3.0)
    assert time.perf_counter() - t0 < 1.5
    assert db.interrupts == 0, "postgres must not be interrupted client-side"


def test_wall_cap_is_idle_when_the_work_finishes():
    db = _PgStub()
    with tr.query_deadline(db, 2.0):
        _burn(0.05)


def test_wall_cap_off_when_seconds_falsy():
    db = _PgStub()
    with tr.query_deadline(db, 0):
        _burn(0.2)


def test_capped_python_phase_reads_as_censored():
    db = _PgStub()
    try:
        with tr.query_deadline(db, 0.3):
            _burn(3.0)
    except tr.QueryTimeout as exc:
        assert tr.is_timeout(exc)
