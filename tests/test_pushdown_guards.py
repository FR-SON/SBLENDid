from src.Operators.Seekers import SC


def test_default_operator_is_not_approximate():
    assert SC.is_approximate is False


def test_semantic_seekers_are_approximate():
    from src.Semantic.seekers.union_base import SemanticUnionSeekerBase
    from src.Semantic.seekers.join_base import SemanticJoinSeekerBase
    assert SemanticUnionSeekerBase.is_approximate is True
    assert SemanticJoinSeekerBase.is_approximate is True


def test_guard_config_defaults():
    from src.cost_model import pushdown_guard, pushdown_filter_width
    assert pushdown_guard() in {"none", "A", "B"}
    assert isinstance(pushdown_filter_width(), int)


from src.Operators.OperatorBase import Operator as _Operator


class _RecOp(_Operator):
    """Records the additionals and k it was asked to run/emit with."""
    def __init__(self, name, ids, cost, approx=False):
        super().__init__(k=len(ids))
        self.name, self._ids, self._cost = name, ids, cost
        self.is_approximate = approx
        self.ran_with = []

    def cost(self): return self._cost
    def ml_cost(self, db): return 0.0

    def run(self, additionals=""):
        self.ran_with.append((additionals, self.k))
        return list(self._ids[: self.k])

    def create_sql_query(self, db, additionals=""):
        self.ran_with.append((additionals, self.k))
        return "SELECT 1"


class _DB:
    def create_sql_list_numeric(self, ids): return ",".join(map(str, ids))


def _build(guard, width, inputs, monkeypatch):
    import sys, importlib
    importlib.import_module("src.Operators.Combiners.Intersection")
    I = sys.modules["src.Operators.Combiners.Intersection"]
    monkeypatch.setattr(I, "pushdown_guard", lambda **k: guard, raising=False)
    monkeypatch.setattr(I, "pushdown_filter_width", lambda **k: width, raising=False)
    inter = I.Intersection(k=10)
    inter.set_inputs(inputs)
    inter.create_sql_query(_DB())
    return inter


def test_guard_a_widens_approximate_source(monkeypatch):
    su = _RecOp("su", list(range(100)), cost=5, approx=True)
    sc = _RecOp("sc", [1, 2, 3], cost=34)
    _build("A", 50, [su, sc], monkeypatch)
    assert any(k == 50 for _, k in su.ran_with)


def test_guard_b_forces_approximate_last(monkeypatch):
    su = _RecOp("su", list(range(100)), cost=5, approx=True)
    sc = _RecOp("sc", [1, 2, 3], cost=34)
    _build("B", 50, [su, sc], monkeypatch)
    assert sc.ran_with, "exact seeker should have run as source"
    assert su.ran_with, "approximate seeker should have run as the final input"
    assert all("SELECT" not in str(a) for a, _ in su.ran_with)


def test_guard_none_unchanged(monkeypatch):
    su = _RecOp("su", list(range(100)), cost=5, approx=True)
    sc = _RecOp("sc", [1, 2, 3], cost=34)
    _build("none", 50, [su, sc], monkeypatch)
    assert any(k == su.k for _, k in su.ran_with)


def test_guard_a_widens_difference_rhs(monkeypatch):
    import importlib, sys
    importlib.import_module("src.Operators.Combiners.Difference")
    D = sys.modules["src.Operators.Combiners.Difference"]
    monkeypatch.setattr(D, "pushdown_guard", lambda **k: "A", raising=False)
    monkeypatch.setattr(D, "pushdown_filter_width", lambda **k: 40, raising=False)
    lhs = _RecOp("lhs", [1, 2, 3], cost=34)
    rhs = _RecOp("rhs", list(range(100)), cost=5, approx=True)
    diff = D.Difference(k=10); diff.set_inputs([lhs, rhs])
    diff.create_sql_query(_DB())
    assert any(k == 40 for _, k in rhs.ran_with)
