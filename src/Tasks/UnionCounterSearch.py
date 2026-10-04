from src.Operators import Combiners, Seekers
from src.Plan import Plan

import pandas as pd


def UnionCounterSearch(dataset: pd.DataFrame, k: int = 10) -> Plan:
    plan = Plan()
    for clm_name in dataset.columns:
        plan.add(clm_name, Seekers.SC(dataset[clm_name], k * 10))
    plan.add("union", Combiners.Counter(k=k), inputs=dataset.columns)
    return plan
