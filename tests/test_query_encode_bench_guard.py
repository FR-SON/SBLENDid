"""Benchmark and Optimizer tooling measures in-index lookups: every semantic seeker it builds pins query_encode = off."""
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1] / "src"
_PKGS = (_ROOT / "Benchmark", _ROOT / "Optimizer")
_SEEKER_REFS = ("Seekers.SU", "Seekers.SJ", "SemanticUnion(", "SemanticJoin(")
_PIN = '"query_encode": "off"'

_SITES = sorted(
    p for pkg in _PKGS for p in pkg.rglob("*.py")
    if "config_overrides=" in p.read_text() and any(r in p.read_text() for r in _SEEKER_REFS)
)


@pytest.mark.parametrize("path", _SITES, ids=lambda p: str(p.relative_to(_ROOT)))
def test_semantic_seeker_sites_pin_query_encode_off(path):
    assert _PIN in path.read_text(), (
        f"{path.relative_to(_ROOT)} builds SU/SJ without pinning {_PIN} in its config_overrides"
    )


def test_the_guard_sees_the_known_sites():
    names = {str(p.relative_to(_ROOT)) for p in _SITES}
    assert {"Benchmark/adapters/join.py", "Benchmark/adapters/union.py",
            "Optimizer/semantic_sweep.py"} <= names


def test_join_adapter_seeker_is_lookup_only():
    # Session default config carries [Semantic.SJ]; the handle opens lazily, so no index is needed.
    from src.Benchmark.adapters.join import _build_plan
    df = pd.DataFrame({"col": ["x"]})
    plan = _build_plan("sj", ["x"], df, "col", "q.csv", 5, dataset="ds-x")
    op = next(iter(plan._operators.values()))
    assert op._cfg.query_encode == "off"


def test_union_adapter_seeker_is_lookup_only():
    from src.Benchmark.adapters.union import _build_union_plan
    df = pd.DataFrame({"a": ["x"], "b": ["y"]})
    for dataset in ("ds-x", None):
        plan = _build_union_plan("su", df, 5, qtable="q.csv", dataset=dataset)
        op = next(iter(plan._operators.values()))
        assert op._cfg.query_encode == "off"
