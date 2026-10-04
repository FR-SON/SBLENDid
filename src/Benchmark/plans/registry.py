from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from src.DBHandler import DBHandler


@dataclass
class ArmResult:
    name: str
    rows: list[dict]
    summary: dict


@dataclass
class PlanResult:
    plan: str
    point: str
    dataset: str
    task: str
    arms: list[ArmResult]
    deltas: dict
    meta: dict


@dataclass
class PlanContext:
    dataset: str
    db: "DBHandler"
    task: str
    k: int
    seed: int
    out_dir: Path
    git_sha: str
    config_snapshot: dict
    limit: int | None = None


@dataclass(frozen=True)
class PlanSpec:
    name: str
    description: str
    point: str
    run: Callable[["PlanContext"], "PlanResult"]


PLANS: dict[str, PlanSpec] = {}


def register(spec: PlanSpec) -> PlanSpec:
    if spec.name in PLANS:
        raise ValueError(f"bench plan {spec.name!r} already registered")
    PLANS[spec.name] = spec
    return spec
