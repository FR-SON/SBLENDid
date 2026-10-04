from __future__ import annotations

import json
from pathlib import Path

from src import paths


def _fixture_path(dataset: str, task: str) -> Path:
    return (paths.datasets_root() / dataset / "plans" / "oracle"
            / f"{task}_oracle_keywords.json")


def load_oracle(dataset: str, task: str) -> dict[str, dict[str, list[str]]]:
    p = _fixture_path(dataset, task)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def repair_recall(R: list[str], recovered: list[str],
                  misses: set[str]) -> list[str]:
    seen = set(R)
    add = [t for t in recovered if t in misses and t not in seen]
    return R + add


def repair_precision(R: list[str], fp_flagged: list[str],
                     fps: set[str]) -> list[str]:
    drop = fps & set(fp_flagged)
    return [t for t in R if t not in drop]


def repair_both(R: list[str], recovered: list[str], fp_flagged: list[str],
                misses: set[str], fps: set[str]) -> list[str]:
    return repair_recall(repair_precision(R, fp_flagged, fps), recovered, misses)
