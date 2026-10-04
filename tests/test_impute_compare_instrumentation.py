import pandas as pd
import pytest

import src.Benchmark.impute_compare as IC


class _Plan:
    def __init__(self, ids):
        self._ids = ids

    def run(self):
        return self._ids


@pytest.fixture
def stubbed(monkeypatch):
    src = pd.DataFrame({"city": ["paris", "rome", "nice", "lyon", "metz", "bonn", "kiel"],
                        "country": ["france"] * 7})
    ctx = IC._QueryCtx("ds", "src.csv", "city", "country", src,
                       src.head(5)[["city", "country"]], src.iloc[5:]["city"],
                       {"bonn", "kiel"}, {"bonn": "france", "kiel": "france"})
    cand = pd.DataFrame({0: ["bonn", "kiel"], 1: ["france", "france"]})

    class _Store:
        def __init__(self, *a, **kw):
            self.misses = 0

        def get(self, tid):
            self.misses += 1
            return cand

        def col_sets(self, tid):
            return {c: set(cand[c]) for c in cand.columns}

    monkeypatch.setattr(IC, "load_sidecar", lambda d: ({1: "src.csv"}, {"src.csv"}))
    monkeypatch.setattr(IC, "load_queries", lambda d: [{"source_table": "src.csv",
                                                        "key_col": "city", "val_col": "country"}])
    monkeypatch.setattr(IC, "build_ctx", lambda d, q: ctx)
    monkeypatch.setattr(IC, "open_dataset_db", lambda d: type("DB", (), {"close": lambda s: None})())
    monkeypatch.setattr(IC, "CandidateStore", _Store)
    monkeypatch.setattr(IC, "bind_plan", lambda p, db: None)
    return {"leg": lambda ctx, k, kc: _Plan([2, 3, 4])}


def _counting_legs(counter, fail_first=False):
    class _P:
        def run(self):
            counter.append(1)
            if fail_first and len(counter) == 1:
                raise RuntimeError("cold index blew up")
            return [2, 3, 4]
    return {"leg": lambda ctx, k, kc: _P()}


def test_warmup_runs_once_per_cell_and_is_not_timed(stubbed):
    calls = []
    out = IC.run_compare("ds", k_grid=[10], legs=_counting_legs(calls), warmup=1)
    assert len(calls) == 2
    assert out["warmup"] == 1
    assert out["results"]["leg"]["10"]["n_retrieved"] == 3.0


def test_warmup_zero_keeps_the_old_behavior(stubbed):
    calls = []
    IC.run_compare("ds", k_grid=[10], legs=_counting_legs(calls), warmup=0)
    assert len(calls) == 1


def test_warmup_is_per_cell(stubbed):
    calls = []
    IC.run_compare("ds", k_grid=[10, 25], legs=_counting_legs(calls), warmup=1)
    assert len(calls) == 4


def test_failed_warmup_does_not_kill_the_run(stubbed, capsys):
    calls = []
    out = IC.run_compare("ds", k_grid=[10], legs=_counting_legs(calls, fail_first=True),
                         warmup=1)
    assert out["results"]["leg"]["10"]["n_retrieved"] == 3.0
    assert "warmup" in capsys.readouterr().out


def test_cell_reports_retrieved_count_and_resolve_cost(stubbed):
    out = IC.run_compare("ds", k_grid=[10], legs=stubbed)
    cell = out["results"]["leg"]["10"]
    assert cell["n_retrieved"] == 3.0
    assert cell["resolve_runtime_ms"] >= 0.0
    assert cell["resolve_cold_fetches"] == 3
    assert cell["retrieval_runtime_ms"] >= 0.0
    assert cell["retrieval_runtime_ms_median"] >= 0.0
    assert cell["retrieval_runtime_ms_p95"] >= cell["retrieval_runtime_ms_median"]


def test_p95_is_nearest_rank():
    from src.Benchmark.impute_compare import _p95
    assert _p95([]) == 0.0
    assert _p95([7.0]) == 7.0
    assert _p95(list(range(1, 21))) == 19
    assert _p95([5.0, 1.0, 3.0]) == 5.0


def test_cold_fetches_drop_when_the_store_caches(stubbed, monkeypatch):
    class _Cached(IC.CandidateStore):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._seen = set()

        def get(self, tid):
            if tid not in self._seen:
                self._seen.add(tid)
                self.misses += 1
            return pd.DataFrame({0: ["bonn"], 1: ["france"]})

        def col_sets(self, tid):
            return {0: {"bonn"}, 1: {"france"}}

    monkeypatch.setattr(IC, "CandidateStore", _Cached)
    out = IC.run_compare("ds", k_grid=[10, 25], legs=stubbed)
    assert out["results"]["leg"]["10"]["resolve_cold_fetches"] == 3
    assert out["results"]["leg"]["25"]["resolve_cold_fetches"] == 0
