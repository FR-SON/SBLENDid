import pandas as pd

from src.Optimizer.freqs import build_freqs_dict


class _FakeDB:
    def __init__(self, rows):
        self.rows = rows
        self.captured = None

    def execute_and_fetchall(self, query):
        self.captured = query
        return self.rows


def test_build_freqs_writes_csv_from_handler(tmp_path):
    db = _FakeDB([("alpha", 3), ("beta", 1)])
    out = tmp_path / "freqs_dict.csv"
    p = build_freqs_dict(db, out_path=out)
    assert p == out
    df = pd.read_csv(out)
    assert list(df.columns) == ["tokenized", "frequency"]
    assert dict(zip(df["tokenized"].astype(str), df["frequency"])) == {"alpha": 3, "beta": 1}


def test_build_freqs_runs_portable_group_by_no_backend_plumbing(tmp_path):
    db = _FakeDB([("a", 1)])
    build_freqs_dict(db, index_table="blend_index", out_path=tmp_path / "f.csv")
    q = db.captured.lower()
    assert "count(*)" in q and "group by" in q
    assert "alltables" in q and "tokenized" in q
    assert "pragma" not in q and "copy" not in q


def test_build_freqs_empty_result_writes_header_only(tmp_path):
    out = tmp_path / "f.csv"
    build_freqs_dict(_FakeDB([]), out_path=out)
    df = pd.read_csv(out)
    assert list(df.columns) == ["tokenized", "frequency"]
    assert len(df) == 0
