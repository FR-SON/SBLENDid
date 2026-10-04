from src.Benchmark.correctness.arms import PinnedIntersection
from src.Operators.OperatorBase import Operator


class _RecOp(Operator):
    def __init__(self, name, ids, cost):
        super().__init__(k=len(ids))
        self.name, self._ids, self._cost = name, ids, cost
        self.is_approximate = False
        self.ran = []

    def cost(self): return self._cost
    def ml_cost(self, db): return 0.0

    def run(self, additionals=""):
        self.ran.append(additionals); return list(self._ids[: self.k])

    def create_sql_query(self, db, additionals=""):
        self.ran.append(additionals); return "SELECT 1"


class _DB:
    def create_sql_list_numeric(self, ids): return ",".join(map(str, ids))


def test_pinned_intersection_keeps_given_order():
    expensive = _RecOp("a", [1, 2], cost=100)
    cheap = _RecOp("b", [2, 3], cost=1)
    inter = PinnedIntersection(k=10); inter.set_inputs([expensive, cheap])
    inter.create_sql_query(_DB())
    assert expensive.ran and expensive.ran[0] == ""


def test_run_arm_pinned_rejects_guard():
    import pytest
    from src.Benchmark.correctness.arms import run_arm
    with pytest.raises(ValueError):
        run_arm([], "intersection", [1, 0], db=None, k=10, guard="A")
