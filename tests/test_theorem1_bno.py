import pytest

from src.Benchmark.correctness.theorem1 import legs as L
from src.Benchmark.correctness.theorem1.shapes import SHAPES


class _Db:
    def execute_and_fetchall(self, sql):
        return sql

    def create_sql_list_numeric(self, ids):
        return ",".join(map(str, ids))


class _Leg:
    """Row list truncated at self.k; an alternate order below a k threshold simulates tie shuffle."""

    def __init__(self, full, k, tie_order=None):
        self.k = k
        self._full = list(full)
        self._tie = list(tie_order) if tie_order else None

    def run(self, additionals=""):
        return [r[0] for r in self.create_sql_query(_Db(), additionals)][: self.k]

    def create_sql_query(self, db, additionals=""):
        rows = self._full if (self._tie is None or self.k >= 10 ** 6) else self._tie
        return [(t,) for t in rows[: self.k]]


SHAPE = SHAPES["sc_sc_flat"]


def test_cut_faithful_and_distinct():
    assert L.cut([1, 1, 2, 3], 3) == [1, 1, 2]
    assert L.cut([1, 1, 2, 3], 3, "distinct") == [1, 2, 3]
    assert L.cut([1, 2], None) == [1, 2]


def test_full_rows_sets_and_restores_k_and_asserts_cap():
    leg = _Leg([1, 2, 3], k=2)
    assert L.full_rows(leg, _Db()) == [1, 2, 3]
    assert leg.k == 2
    with pytest.raises(RuntimeError):
        L.full_rows(_Leg(list(range(5)), k=2), _Db(), nonbinding_k=4)


def test_direct_and_slice_agree_without_ties(duckdb_unit_env):
    mk = lambda: [_Leg([1, 2, 3, 4], k=3), _Leg([2, 3, 5], k=3)]
    d, _ = L.bno_run(mk(), SHAPE, 2, db=_Db(), construction="direct")
    s, used = L.bno_run(mk(), SHAPE, 2, db=_Db(), construction="slice")
    assert d == s == [2, 3]
    assert used == ["slice", "slice"]


def test_direct_and_slice_differ_on_tie_fixture(duckdb_unit_env):
    mk = lambda: [_Leg([1, 2, 3, 4], k=3, tie_order=[1, 2, 4, 3]),
                  _Leg([1, 2, 3, 4], k=4)]
    d, _ = L.bno_run(mk(), SHAPE, 4, db=_Db(), construction="direct")
    s, _ = L.bno_run(mk(), SHAPE, 4, db=_Db(), construction="slice")
    assert s == [1, 2, 3]
    assert d == [1, 2, 4]
    assert d != s


def test_terminal_order_reorders(duckdb_unit_env):
    mk = lambda: [_Leg([1, 2, 3], k=3), _Leg([3, 2, 1], k=3)]
    lega, _ = L.bno_run(mk(), SHAPE, 3, db=_Db(), construction="direct")
    term, _ = L.bno_run(mk(), SHAPE, 3, db=_Db(), construction="direct",
                        bno_order="terminal",
                        terminal_ranking=[(3, 9.0), (2, 5.0), (1, 1.0)])
    assert lega == [1, 2, 3]
    assert term == [3, 2, 1]


def test_terminal_order_without_ranking_raises():
    with pytest.raises(ValueError):
        L.combine_bno([[1], [1]], 1, order="terminal")


def test_mc_leg_never_sliced():
    mc_shape = SHAPES["mc_sc_repo"]
    legs = [_Leg([1, 2], k=2), _Leg([1, 3], k=2)]
    _, used = L.bno_run(legs, mc_shape, 2, db=_Db(), construction="slice")
    assert used == ["direct", "slice"]


def test_semantic_full_fn_used_for_slice():
    su_shape = SHAPES["su_sc"]
    calls = []

    def sem_full():
        calls.append(1)
        return [7, 8, 9]

    legs = [_Leg([0], k=2), _Leg([8, 7, 1], k=3)]
    out, used = L.bno_run(legs, su_shape, 3, db=_Db(), construction="slice",
                          full_fns={0: sem_full})
    assert calls == [1]
    assert used == ["slice", "slice"]
    assert out == [7, 8]
