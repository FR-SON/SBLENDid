"""Theorem-1 plan-shape registry."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.Index.tokenize import tokenize_cell

BASELINE_COSTS = {"KW": 3, "SC": 4, "C": 6, "MC": 10, "SU": 2, "SJ": 1}
SEM_LAST_ORDER_COST = 8  # must sort above SC=4 and C=6, below MC=10
SCOREABLE_KINDS = frozenset({"SC", "KW", "C"})
SEMANTIC_KINDS = frozenset({"SU", "SJ"})


@dataclass(frozen=True)
class LegSpec:
    kind: str
    columns: str
    k_mult: int


@dataclass(frozen=True)
class Shape:
    name: str
    legs: tuple[LegSpec, LegSpec]
    shape_source: str
    operators_upstream_faithful: bool | None
    optimizer_reachable: bool
    arm3_only: bool
    prefix_stable: bool


SHAPES: dict[str, Shape] = {s.name: s for s in (
    Shape("sc_sc_flat", (LegSpec("SC", "c0", 1), LegSpec("SC", "c1", 1)),
          "src/Tasks/DependentDataSearch.py, flat k",
          True, True, False, True),
    Shape("sc_sc_prov", (LegSpec("SC", "c0", 30), LegSpec("SC", "c1", 30)),
          "DataImputation's k*30 multiplier on the DependentDataSearch shape",
          True, True, False, True),
    Shape("sc_kw", (LegSpec("SC", "c0", 1), LegSpec("KW", "c1", 1)),
          "constructed for this study",
          True, False, False, True),
    Shape("mc_sc_paper", (LegSpec("MC", "c0c1", 1), LegSpec("SC", "c0", 1)),
          "paper Listing 4, l.10-12 (MC is the thesis port)",
          False, True, True, False),
    Shape("mc_sc_repo", (LegSpec("MC", "c0c1", 10), LegSpec("SC", "c0", 30)),
          "src/Tasks/DataImputation.py",
          False, True, True, False),
    Shape("su_sc", (LegSpec("SU", "df", 1), LegSpec("SC", "c0", 1)),
          "thesis SU seeker",
          None, True, False, True),
    Shape("sj_sc", (LegSpec("SJ", "c0_df", 1), LegSpec("SC", "c0", 1)),
          "thesis SJ seeker",
          None, True, False, True),
)}


def usable_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if any(tokenize_cell(v) for v in df[c].tolist())]


def select_columns(df: pd.DataFrame) -> tuple[str, str] | None:
    cols = usable_columns(df)
    if len(cols) < 2:
        return None
    return str(cols[0]), str(cols[1])


def has_mc(shape: Shape) -> bool:
    return any(l.kind == "MC" for l in shape.legs)


def is_semantic(shape: Shape) -> bool:
    return any(l.kind in SEMANTIC_KINDS for l in shape.legs)


def leg_order_key(spec: LegSpec, *, sem_position: str = "first") -> int:
    if spec.kind in SEMANTIC_KINDS and sem_position == "last":
        return SEM_LAST_ORDER_COST
    return BASELINE_COSTS[spec.kind]


def terminal_leg(shape: Shape, *, sem_position: str = "first") -> LegSpec:
    # last-executed under Intersection's stable sort: ties keep declaration order
    a, b = shape.legs
    ka = leg_order_key(a, sem_position=sem_position)
    kb = leg_order_key(b, sem_position=sem_position)
    return b if kb >= ka else a
