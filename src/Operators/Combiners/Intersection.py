from src.Operators.Combiners.CombinerBase import Combiner
from src.cost_model import pushdown_guard, pushdown_filter_width
from functools import cmp_to_key

# Typing imports
from src.DBHandler import DBHandler


class Intersection(Combiner):
    def cost(self) -> int:
        return min(input_.cost() for input_ in self._inputs)

    def ml_cost(self, db: DBHandler) -> float:
        return min(input_.ml_cost(db) for input_ in self._inputs)

    def create_sql_query(self, db: DBHandler, additionals: str = "") -> str:
        guard = pushdown_guard()

        def _order_key(o):
            oc = getattr(o, "order_cost", None)
            return o.cost() if oc is None else oc

        def lazy_comparator(o1, o2):
            if guard == "B" and o1.is_approximate != o2.is_approximate:
                return 1 if o1.is_approximate else -1
            k1, k2 = _order_key(o1), _order_key(o2)
            if k1 != k2:
                return k1 - k2
            return o1.ml_cost(db) - o2.ml_cost(db)

        sorted_inputs = list(sorted(self._inputs, key=cmp_to_key(lazy_comparator)))
        width = pushdown_filter_width()
        intersect_additionals = ""
        for input_ in sorted_inputs[:-1]:
            if guard == "A" and input_.is_approximate:
                saved_k, input_.k = input_.k, width
                try:
                    result = input_.run(additionals + intersect_additionals)
                finally:
                    input_.k = saved_k
            else:
                result = input_.run(additionals + intersect_additionals)
            if len(result) == 0:
                return "SELECT TableId FROM AllTables WHERE 1=0"
            intersect_additionals = f" AND TableId IN ({db.create_sql_list_numeric(result)}) "
        sorted_inputs[-1].k = self.k
        return sorted_inputs[-1].create_sql_query(db, additionals=additionals + intersect_additionals)
