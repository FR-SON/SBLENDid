from src.Operators.Combiners.CombinerBase import Combiner
from src.cost_model import pushdown_guard, pushdown_filter_width

# Typing imports
from src.DBHandler import DBHandler


class Difference(Combiner):
    def cost(self) -> int:
        return self._inputs[1].cost()

    def ml_cost(self, db: DBHandler) -> float:
        return self._inputs[1].ml_cost(db)

    def create_sql_query(self, db: DBHandler, additionals: str = "") -> str:
        rhs = self._inputs[1]
        if pushdown_guard() in ("A", "B") and rhs.is_approximate:
            saved_k, rhs.k = rhs.k, pushdown_filter_width()
            try:
                minus_results = rhs.run(additionals)
            finally:
                rhs.k = saved_k
        else:
            minus_results = rhs.run(additionals)
        additionals += (
            f" AND TableId NOT IN ({db.create_sql_list_numeric(minus_results)}) "
            if minus_results
            else ""
        )
        self._inputs[0].k = self.k
        sql = self._inputs[0].create_sql_query(db, additionals=additionals)
        return sql
