import pandas as pd
from src.Tasks.UnionCounterSearch import UnionCounterSearch
from src.Operators.Combiners.Counter import Counter

def test_union_counter_search_wires_counter():
    df = pd.DataFrame({"a": ["x", "y"], "b": ["1", "2"]})
    plan = UnionCounterSearch(df, k=5)
    assert isinstance(plan._operators["union"], Counter)
    assert plan._operators["union"].k == 5
    assert set(plan._operators) == {"a", "b", "union"}
