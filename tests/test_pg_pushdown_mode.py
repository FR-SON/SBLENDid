from src.Semantic.retrieve import _pg_allowed_ids

_MAP = {"a": 1, "b": 2, "c": 3}


def test_no_filter_returns_none():
    assert _pg_allowed_ids(None, _MAP, postfilter=False, exact=False) is None
    assert _pg_allowed_ids(None, _MAP, postfilter=True, exact=False) is None


def test_prefilter_maps_ids():
    assert _pg_allowed_ids({"a", "b"}, _MAP, postfilter=False, exact=False) == {1, 2}


def test_postfilter_ann_drops_sql_filter():
    assert _pg_allowed_ids({"a", "b"}, _MAP, postfilter=True, exact=False) is None


def test_postfilter_exact_still_filters_in_sql():
    assert _pg_allowed_ids({"a", "b"}, _MAP, postfilter=True, exact=True) == {1, 2}


def test_prefilter_exact_filters_in_sql():
    assert _pg_allowed_ids({"a", "b"}, _MAP, postfilter=False, exact=True) == {1, 2}


def test_unknown_tables_dropped_from_map():
    assert _pg_allowed_ids({"a", "zzz"}, _MAP, postfilter=False, exact=False) == {1}
