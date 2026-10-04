import json

import pandas as pd

import src.Benchmark.recipes.imputation as IM

_TABLE = pd.DataFrame({
    "id":    [f"E{i:03d}" for i in range(20)],
    "code":  [f"{i:04d}" for i in range(20)],
    "city":  ["paris", "lyon", "nice", "metz"] * 5,
})


def _csv(tmp_path, name="t.csv"):
    _TABLE.to_csv(tmp_path / name, index=False)
    return tmp_path / name


def test_allowed_none_keeps_the_top_ranked_key(tmp_path):
    assert IM._select_key_value(_csv(tmp_path))[0] == "id"


def test_allowed_skips_to_the_next_eligible_key(tmp_path):
    kv = IM._select_key_value(_csv(tmp_path), allowed={"code", "city"})
    assert kv is not None and kv[0] == "code"


def test_no_allowed_key_rejects_the_table(tmp_path):
    assert IM._select_key_value(_csv(tmp_path), allowed={"city"}) is None
    assert IM._select_key_value(_csv(tmp_path), allowed=set()) is None


def _prepare(tmp_path, monkeypatch, enrolled, **kw):
    csvs = tmp_path / "csvs"
    csvs.mkdir(parents=True)
    for name in ("a.csv", "b.csv"):
        _TABLE.to_csv(csvs / name, index=False)
    monkeypatch.setattr(IM, "dataset_dir", lambda d: tmp_path)
    monkeypatch.setattr(IM, "load_sidecar", lambda d: ({0: "a.csv", 1: "b.csv"},
                                                       {"a.csv", "b.csv"}))
    monkeypatch.setattr(IM, "_enrolled_columns", lambda d: (enrolled, "snoopy/default"))
    out = tmp_path / "bench" / "imputation"
    out.mkdir(parents=True)
    IM.prepare_imputation("ds", out, n=10, seed=0, force=True, **kw)
    rows = list(pd.read_csv(out / "query.csv", dtype=str).itertuples(index=False))
    return rows, json.loads((out / "manifest.json").read_text())


def test_gate_drops_tables_whose_columns_are_not_indexed(tmp_path, monkeypatch):
    rows, mf = _prepare(tmp_path, monkeypatch, {"a.csv": {"id"}}, require_semantic=True)
    assert [r.source_table for r in rows] == ["a.csv"]
    assert mf["require_semantic"] is True
    assert mf["semantic_index"] == "snoopy/default"
    assert mf["pool_size"] == 1


def test_gate_falls_back_to_the_next_indexed_key(tmp_path, monkeypatch):
    rows, _ = _prepare(tmp_path, monkeypatch,
                       {"a.csv": {"code"}, "b.csv": {"code"}}, require_semantic=True)
    assert {r.key_col for r in rows} == {"code"}


def test_gate_off_ignores_enrollment(tmp_path, monkeypatch):
    rows, mf = _prepare(tmp_path, monkeypatch, {})
    assert {r.source_table for r in rows} == {"a.csv", "b.csv"}
    assert mf["require_semantic"] is False and mf["semantic_index"] is None


def test_gate_is_reproducible(tmp_path, monkeypatch):
    a, _ = _prepare(tmp_path, monkeypatch, {"a.csv": {"id"}, "b.csv": {"id"}},
                    require_semantic=True)
    b, _ = _prepare(tmp_path / "second", monkeypatch, {"a.csv": {"id"}, "b.csv": {"id"}},
                    require_semantic=True)
    assert [tuple(r) for r in a] == [tuple(r) for r in b]


def test_compare_skips_unindexed_query_instead_of_dying(monkeypatch):
    import src.Benchmark.impute_compare as IC
    from src.Semantic.retrieve import QueryColumnNotIndexed

    src = pd.DataFrame({"city": ["paris", "rome", "nice", "lyon", "metz", "bonn", "kiel"],
                        "country": ["france"] * 7})
    ctx = IC._QueryCtx("ds", "src.csv", "city", "country", src,
                       src.head(5)[["city", "country"]], src.iloc[5:]["city"],
                       {"bonn"}, {"bonn": "france"})

    class _P:
        def run(self):
            raise QueryColumnNotIndexed("('src.csv', 'city') not in registry")

    monkeypatch.setattr(IC, "load_sidecar", lambda d: ({1: "src.csv"}, {"src.csv"}))
    monkeypatch.setattr(IC, "load_queries", lambda d: [{}])
    monkeypatch.setattr(IC, "build_ctx", lambda d, q: ctx)
    monkeypatch.setattr(IC, "open_dataset_db", lambda d: type("DB", (), {"close": lambda s: None})())
    monkeypatch.setattr(IC, "CandidateStore", lambda *a, **kw: None)
    monkeypatch.setattr(IC, "bind_plan", lambda p, db: None)

    out = IC.run_compare("ds", k_grid=[10], legs={"leg": lambda c, k, kc: _P()}, warmup=0)
    assert out["results"]["leg"]["10"]["skipped_unindexed"] == 1
