from src.Benchmark.correctness.refs import reference


class _FakeOp:
    def __init__(self, ids):
        self._ids = ids

    def run(self, additionals=""):
        return list(self._ids)


def test_reference_intersection_preserves_first_order():
    a = _FakeOp([3, 1, 2, 9])
    b = _FakeOp([2, 3, 7])
    assert reference([a, b], "intersection") == [3, 2]


def test_reference_difference():
    a = _FakeOp([3, 1, 2, 9])
    b = _FakeOp([1, 9])
    assert reference([a, b], "difference") == [3, 2]


def test_reference_unknown_combine_raises():
    import pytest
    a = _FakeOp([1, 2])
    with pytest.raises(ValueError):
        reference([a], "union")
