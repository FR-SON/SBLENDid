from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve(env_var: str, default: Path) -> Path:
    raw = os.environ.get(env_var)
    return Path(raw).expanduser() if raw else default


def config_path() -> Path:
    return _resolve("BLEND_CONFIG", _REPO_ROOT / "config" / "config.ini")


def datasets_root() -> Path:
    return _resolve("BLEND_DATASETS_DIR", _REPO_ROOT / "datasets")


def runs_root() -> Path:
    return _resolve("BLEND_RUNS_DIR", _REPO_ROOT / "runs")


__all__ = ["config_path", "datasets_root", "runs_root"]
