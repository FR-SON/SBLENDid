from __future__ import annotations

import configparser
import json
import warnings
from pathlib import Path

from src.dataset import load_dataset_config
from src.optimizer_paths import costs_path, legacy_costs_path
from src import paths

CONFIG_PATH = paths.config_path()

_CACHE: dict[str, dict[str, int]] = {}
_DOC_CACHE: dict[str, dict] = {}
_WARNED_LEGACY: set[str] = set()


def _reset_cache() -> None:
    _CACHE.clear()
    _DOC_CACHE.clear()
    _WARNED_LEGACY.clear()


def cost_basis(*, config_path: Path = CONFIG_PATH) -> str:
    """Return the configured cost basis: "logical" (baseline integers) or "measured" (profile)."""
    return _cost_basis(config_path)


def _cost_basis(config_path: Path) -> str:
    parser = configparser.ConfigParser()
    parser.read(config_path)
    if parser.has_section("Optimizer"):
        return parser["Optimizer"].get("cost_basis", "logical").strip().lower()
    return "logical"


def pushdown_guard(*, config_path: Path = CONFIG_PATH) -> str:
    parser = configparser.ConfigParser()
    parser.read(config_path)
    if parser.has_section("Optimizer"):
        return parser["Optimizer"].get("pushdown_guard", "none").strip()
    return "none"


def pushdown_filter_width(*, config_path: Path = CONFIG_PATH) -> int:
    parser = configparser.ConfigParser()
    parser.read(config_path)
    if parser.has_section("Optimizer"):
        return int(parser["Optimizer"].get("pushdown_filter_width", "500"))
    return 500


def _warn_legacy_costs(legacy: Path, expected: Path) -> None:
    if str(legacy) not in _WARNED_LEGACY:
        _WARNED_LEGACY.add(str(legacy))
        warnings.warn(
            f"reading legacy un-profiled costs {legacy}; expected profile-keyed "
            f"{expected}. Re-run `report --write` to migrate.",
            stacklevel=2,
        )


def _measured_path(dataset_dir: Path, profile: str | None = None) -> Path:
    if profile:
        p = costs_path(dataset_dir, profile)
        if p.is_file():
            return p
        legacy = legacy_costs_path(dataset_dir)
        if legacy.is_file():
            _warn_legacy_costs(legacy, p)
            return legacy
        return p
    return legacy_costs_path(dataset_dir)


def _load_doc(dataset_dir: Path, profile: str | None = None) -> dict:
    path = _measured_path(dataset_dir, profile)
    key = str(path)
    if key not in _DOC_CACHE:
        if not path.is_file():
            raise FileNotFoundError(
                f"cost_basis=measured but {path} is missing. Run "
                f"`report --write` to regenerate, "
                f"or set cost_basis=logical in config.ini."
            )
        _DOC_CACHE[key] = json.loads(path.read_text())
    return _DOC_CACHE[key]


def _load_costs(dataset_dir: Path, profile: str | None = None) -> dict[str, int]:
    path = _measured_path(dataset_dir, profile)
    key = str(path)
    if key not in _CACHE:
        _CACHE[key] = _load_doc(dataset_dir, profile)["costs"]
    return _CACHE[key]


def cost_scale_anchor_s(*, dataset_dir: Path | None = None, profile: str | None = None,
                        db=None, config_path: Path = CONFIG_PATH) -> float | None:
    """Seconds per cost() unit (cheapest per-type median) of the measured profile, or None if absent."""
    if dataset_dir is None:
        dataset_dir, profile = _dataset_dir_and_profile(db, config_path)
    try:
        medians = (_load_doc(dataset_dir, profile).get("report") or {}).get("median_s") or {}
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    vals = [float(v) for v in medians.values() if isinstance(v, (int, float)) and v > 0]
    return min(vals) if vals else None


def _dataset_dir_and_profile(db, config_path: Path):
    if db is not None:
        return db.dataset_dir(), db.optimizer_profile_str()
    return load_dataset_config(config_path).dir(), None


def resolve_cost(key: str, baseline: int, *, db=None, config_path: Path = CONFIG_PATH) -> int:
    if _cost_basis(config_path) != "measured":
        return baseline
    dataset_dir, profile = _dataset_dir_and_profile(db, config_path)
    costs = _load_costs(dataset_dir, profile)
    if key not in costs:
        raise KeyError(
            f"cost key {key!r} absent from {_measured_path(dataset_dir, profile)}; "
            f"present keys: {sorted(costs)}. Re-run report --write."
        )
    return int(costs[key])
