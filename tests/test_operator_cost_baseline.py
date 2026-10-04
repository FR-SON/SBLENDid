import importlib

import pandas as pd


def test_syntactic_seeker_logical_baselines():
    Seekers = importlib.import_module("src.Operators.Seekers")

    assert Seekers.SC({"a", "b"}, k=10).cost() == 4
    assert Seekers.MC(pd.DataFrame({"c0": ["a"], "c1": ["b"]}), k=10).cost() == 10
    assert Seekers.C(["a", "b"], [1.0, 2.0], k=10).cost() == 6
    assert Seekers.Keyword({"a", "b"}, k=10).cost() == 3
