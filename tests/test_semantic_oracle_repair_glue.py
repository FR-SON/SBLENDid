"""Regression tests for _seeker_ids and _keyword_ids glue in semantic_oracle_repair."""
import types

from src.Benchmark.plans.semantic_oracle_repair import _seeker_ids, _keyword_ids, _KW_K


class FakePlan:
    DB = types.SimpleNamespace(close=lambda: None)

    def __init__(self):
        self._operators = {}

    def add(self, name, op):
        self._operators[name] = op

    def run(self):
        return [0, 2]


def test_seeker_ids_maps_int_ids_to_basenames(monkeypatch):
    monkeypatch.setattr("src.Plan.Plan", FakePlan)
    monkeypatch.setattr("src.Benchmark.db.bind_plan", lambda plan, db: None)
    result = _seeker_ids(object(), object(), {0: "a.csv", 1: "b.csv", 2: "c.csv"})
    assert result == ["a.csv", "c.csv"]


def test_keyword_ids_empty_terms_short_circuits(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("_seeker_ids must not be called for empty terms")

    monkeypatch.setattr("src.Benchmark.plans.semantic_oracle_repair._seeker_ids", boom)
    ctx = types.SimpleNamespace(db=object())
    assert _keyword_ids(None, ctx, {}) == ([], 0.0)
    assert _keyword_ids([], ctx, {}) == ([], 0.0)


def test_keyword_ids_builds_keyword_seeker_and_maps(monkeypatch):
    import src.Operators.Seekers as Seekers

    captured = {}

    def fake_seeker_ids(seeker, db, i2b):
        captured["seeker"] = seeker
        return ["x.csv"]

    monkeypatch.setattr(
        "src.Benchmark.plans.semantic_oracle_repair._seeker_ids", fake_seeker_ids
    )
    ctx = types.SimpleNamespace(db=object())
    ids, elapsed = _keyword_ids(["quebec", "montreal"], ctx, {})
    assert ids == ["x.csv"]
    assert elapsed >= 0.0
    kw = captured["seeker"]
    assert isinstance(kw, Seekers.Keyword)
    assert kw.input == {"quebec", "montreal"}
    assert kw.k == _KW_K
