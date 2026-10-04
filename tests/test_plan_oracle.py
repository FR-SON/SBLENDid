from src.Benchmark.plans.oracle import (
    load_oracle, repair_recall, repair_precision, repair_both,
)


def test_recall_appends_misses_preserving_head_order():
    assert repair_recall(["a", "b"], ["c", "d"], {"c"}) == ["a", "b", "c"]


def test_recall_skips_non_misses_and_existing():
    assert repair_recall(["a"], ["a", "b"], {"b"}) == ["a", "b"]
    assert repair_recall(["a"], ["x"], {"b"}) == ["a"]


def test_precision_drops_only_flagged_fps_preserving_order():
    assert repair_precision(["a", "b", "c"], ["b"], {"b", "c"}) == ["a", "c"]


def test_both_filters_then_appends():
    assert repair_both(["a", "b"], ["c"], ["b"], {"c"}, {"b"}) == ["a", "c"]


def test_load_oracle_absent_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("BLEND_DATASETS_DIR", str(tmp_path))
    assert load_oracle("nope", "union") == {}


def test_load_oracle_reads_fixture(tmp_path, monkeypatch):
    monkeypatch.setenv("BLEND_DATASETS_DIR", str(tmp_path))
    d = tmp_path / "ds" / "plans" / "oracle"
    d.mkdir(parents=True)
    (d / "union_oracle_keywords.json").write_text(
        '{"q.csv": {"terms_recall": ["x"], "terms_fp": ["y"]}}')
    got = load_oracle("ds", "union")
    assert got["q.csv"]["terms_recall"] == ["x"]
