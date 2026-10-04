"""order_cost: author-controlled placement of a semantic seeker in an Intersection."""
import importlib
import sys

import pandas as pd

from src.Operators.OperatorBase import Operator


class _RecOp(Operator):
    def __init__(self, name, cost, order, *, order_cost="absent", approx=False):
        super().__init__(k=10)
        self.name, self._cost, self._order = name, cost, order
        self.is_approximate = approx
        if order_cost != "absent":
            self.order_cost = order_cost

    def cost(self):
        return self._cost

    def ml_cost(self, db):
        return 0.0

    def run(self, additionals=""):
        self._order.append(self.name)
        return [1, 2, 3]

    def create_sql_query(self, db, additionals=""):
        self._order.append(self.name)
        return "SELECT 1"


class _DB:
    def create_sql_list_numeric(self, ids):
        return ",".join(map(str, ids))


def _intersection(monkeypatch, guard="none"):
    importlib.import_module("src.Operators.Combiners.Intersection")
    mod = sys.modules["src.Operators.Combiners.Intersection"]
    monkeypatch.setattr(mod, "pushdown_guard", lambda **k: guard, raising=False)
    monkeypatch.setattr(mod, "pushdown_filter_width", lambda **k: 500, raising=False)
    return mod.Intersection


def _order(monkeypatch, inputs, guard="none"):
    inter = _intersection(monkeypatch, guard)(k=10)
    inter.set_inputs(inputs)
    inter.create_sql_query(_DB())
    return inter


def test_default_is_cheap_first(monkeypatch):
    order = []
    _order(monkeypatch, [
        _RecOp("SC", 4, order),
        _RecOp("SU", 2, order, approx=True),
        _RecOp("C", 6, order),
    ])
    assert order == ["SU", "SC", "C"]


def test_order_cost_places_between_neighbors(monkeypatch):
    order = []
    _order(monkeypatch, [
        _RecOp("SC", 4, order),
        _RecOp("SU", 2, order, order_cost=5, approx=True),
        _RecOp("C", 6, order),
    ])
    assert order == ["SC", "SU", "C"]


def test_order_cost_zero_runs_first(monkeypatch):
    order = []
    _order(monkeypatch, [
        _RecOp("A", 1, order),
        _RecOp("Z", 9, order, order_cost=0, approx=True),
    ])
    assert order[0] == "Z"


def test_syntactic_only_unchanged(monkeypatch):
    order = []
    _order(monkeypatch, [_RecOp("C", 6, order), _RecOp("KW", 3, order), _RecOp("SC", 4, order)])
    assert order == ["KW", "SC", "C"]


def test_order_cost_does_not_leak_into_aggregation(monkeypatch):
    order = []
    sc = _RecOp("SC", 4, order)
    su = _RecOp("SU", 2, order, order_cost=99, approx=True)
    inter = _intersection(monkeypatch)(k=10)
    inter.set_inputs([sc, su])
    assert inter.cost() == 2


def _qdf(table_id="t.csv"):
    df = pd.DataFrame({"col": ["a", "b"]})
    df.attrs["table_id"] = table_id
    return df


def test_su_stores_order_cost_and_leaves_cost_unchanged():
    from src.Operators.Seekers import SU
    from src.cost_model import resolve_cost
    assert SU(_qdf(), k=5).order_cost is None
    su = SU(_qdf(), k=5, order_cost=7)
    assert su.order_cost == 7
    assert su.cost() == resolve_cost("SU", 2)


def test_sj_stores_order_cost_and_leaves_cost_unchanged():
    from src.Operators.Seekers import SJ
    from src.cost_model import resolve_cost
    assert SJ(_qdf(), k=5).order_cost is None
    sj = SJ(_qdf(), k=5, order_cost=40)
    assert sj.order_cost == 40
    assert sj.cost() == resolve_cost("SJ", 1)
