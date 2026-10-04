"""Naming and output paths belong to runspec.py alone."""
import re
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parents[1] / "src" / "Benchmark"
_OWNER = _PKG / "runspec.py"

_SOURCES = sorted(p for p in _PKG.rglob("*.py") if p != _OWNER)

_ROOT_LITERAL = re.compile(r"bench_results(?![A-Za-z0-9_])")


@pytest.mark.parametrize("path", _SOURCES, ids=lambda p: str(p.name))
def test_no_hand_rolled_stamp_or_output_root(path):
    text = path.read_text()
    offenders = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        if "strftime" in line:
            offenders.append(f"{path.name}:{lineno} strftime — use open_run()")
        if _ROOT_LITERAL.search(line):
            offenders.append(
                f"{path.name}:{lineno} bench_results literal — "
                "use open_run() or bench_results_root()")
    assert not offenders, "\n".join(offenders)


def test_the_guard_can_see_the_owner_module():
    assert _SOURCES, "no source files found under src/Benchmark"
    assert _ROOT_LITERAL.search(_OWNER.read_text())
