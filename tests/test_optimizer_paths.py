"""Grep-guard and unit tests for the optimizer artifact path helper."""
import re
from pathlib import Path

from src import optimizer_paths as op

_REPO = Path(__file__).resolve().parents[1]
_ALLOWED = {_REPO / "src" / "optimizer_paths.py"}

# (?<!cost) admits cost-model JSONs; only XGB <Class>_model.json is forbidden
_FORBIDDEN = [
    (re.compile(r"freqs_dict\.csv"),
     "repo-tree freqs_dict.csv (use optimizer_paths.freqs_path)"),
    (re.compile(r"(?<!cost)_model\.json"),
     "hardcoded <Class>_model.json (use optimizer_paths.model_path)"),
    (re.compile(r'"optimizer"\s*/\s*"(?:costs|semantic_cost_model)\.json"'),
     "pathlib optimizer/<artifact>.json (use optimizer_paths)"),
    (re.compile(r"optimizer/(?:costs|semantic_cost_model)\.json"),
     "un-profiled optimizer/<artifact>.json literal (use optimizer_paths)"),
]


def _scan_files():
    for base in ("src", "scripts"):
        for p in (_REPO / base).rglob("*"):
            if p.suffix in (".py", ".sh") and p not in _ALLOWED:
                yield p


def test_no_hardcoded_optimizer_artifact_paths():
    offenders = []
    for p in _scan_files():
        text = p.read_text(encoding="utf-8", errors="ignore")
        for rx, why in _FORBIDDEN:
            for m in rx.finditer(text):
                line = text[: m.start()].count("\n") + 1
                offenders.append(f"{p.relative_to(_REPO)}:{line}: {why}")
    assert not offenders, "hardcoded optimizer artifact paths found:\n" + "\n".join(offenders)


def test_optimizer_profile_string():
    assert op.optimizer_profile("postgres", "pgvector", "single") == "postgres-pgvector-single"
    assert op.optimizer_profile("duckdb", "faiss", "two_table") == "duckdb-faiss-two_table"


def test_path_builders_shape():
    d = Path("/x/ds")
    assert op.freqs_path(d) == d / "optimizer" / "freqs.csv"
    assert op.profile_dir(d, "p") == d / "optimizer" / "p"
    assert op.costs_path(d, "p") == d / "optimizer" / "p" / "costs.json"
    assert op.semantic_cost_model_path(d, "p") == d / "optimizer" / "p" / "semantic_cost_model.json"
    assert op.model_path(d, "p", "SingleColumnOverlap") == (
        d / "optimizer" / "p" / "models" / "SingleColumnOverlap_model.json")
    assert op.legacy_costs_path(d) == d / "optimizer" / "costs.json"
    assert op.legacy_semantic_cost_model_path(d) == d / "optimizer" / "semantic_cost_model.json"
