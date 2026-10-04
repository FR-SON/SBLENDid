from src.Benchmark.setdiff_metrics import retention, additions, changed, rbo

def test_retention_and_additions():
    ref = [1, 2, 3, 4]
    arm = [2, 3, 5]
    assert retention(arm, ref) == 0.5
    assert additions(arm, ref) == 1
    assert changed(arm, ref) is True

def test_identity():
    ref = [1, 2, 3]
    assert retention(ref, ref) == 1.0
    assert additions(ref, ref) == 0
    assert changed(ref, ref) is False

def test_empty_ref():
    assert retention([1], []) == 1.0
    assert additions([1], []) == 1

def test_rbo_identical_is_one():
    assert rbo([1, 2, 3], [1, 2, 3], p=0.9) == 1.0

def test_rbo_disjoint_is_zero():
    assert rbo([1, 2, 3], [4, 5, 6], p=0.9) == 0.0


def test_rbo_partial_overlap_between_zero_and_one():
    assert 0.0 < rbo([1, 2, 3], [1, 2, 4], p=0.9) < 1.0


def test_rbo_order_sensitive():
    assert rbo([1, 2, 3], [3, 2, 1], p=0.9) < 1.0
