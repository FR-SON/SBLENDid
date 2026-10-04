import contextlib
import importlib
from src.Operators.Combiners.Intersection import Intersection
from src.Operators.Combiners.Difference import Difference
from src.Plan import Plan
from src.Benchmark.db import bind_plan


class PinnedIntersection(Intersection):
    def create_sql_query(self, db, additionals: str = "") -> str:
        sorted_inputs = list(self._inputs)
        intersect_additionals = ""
        for input_ in sorted_inputs[:-1]:
            result = input_.run(additionals + intersect_additionals)
            if len(result) == 0:
                return "SELECT TableId FROM AllTables WHERE 1=0"
            intersect_additionals = f" AND TableId IN ({db.create_sql_list_numeric(result)}) "
        sorted_inputs[-1].k = self.k
        return sorted_inputs[-1].create_sql_query(db, additionals=additionals + intersect_additionals)


@contextlib.contextmanager
def forced_guard(guard: str, width: int):
    # patch the Combiners' own bound names; patching src.cost_model would not reach them
    mods = [importlib.import_module("src.Operators.Combiners.Intersection"),
            importlib.import_module("src.Operators.Combiners.Difference")]
    saved = [(m, m.pushdown_guard, m.pushdown_filter_width) for m in mods]
    for m in mods:
        m.pushdown_guard = lambda **k: guard
        m.pushdown_filter_width = lambda **k: width
    try:
        yield
    finally:
        for m, g, w in saved:
            m.pushdown_guard, m.pushdown_filter_width = g, w


def run_arm(legs, combine, order, db, k, *, guard="none", width=500) -> list[int]:
    if combine == "intersection" and order != "cost" and guard != "none":
        raise ValueError("pinned-order intersection arm supports only guard='none'")
    plan = Plan()
    names = [f"leg{i}" for i in range(len(legs))]
    for n, op in zip(names, legs):
        plan.add(n, op)
    if combine == "intersection":
        comb = Intersection(k=k) if order == "cost" else PinnedIntersection(k=k)
        ordered = names if order == "cost" else [names[i] for i in order]
        plan.add("B", comb, inputs=ordered)
    elif combine == "difference":
        plan.add("B", Difference(k=k), inputs=[names[i] for i in order])
    else:
        raise ValueError(combine)
    bind_plan(plan, db)
    with forced_guard(guard, width):
        return plan.run()
