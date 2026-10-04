"""Optimizer artifact paths: dataset-keyed freqs.csv, profile-keyed (dbms-backend-layout) costs and models."""
from __future__ import annotations

from pathlib import Path


def optimizer_profile(dbms: str, vector_backend: str, layout: str) -> str:
    return f"{dbms}-{vector_backend}-{layout}"


def optimizer_root(dataset_dir: Path) -> Path:
    return Path(dataset_dir) / "optimizer"


def freqs_path(dataset_dir: Path) -> Path:
    return optimizer_root(dataset_dir) / "freqs.csv"


def profile_dir(dataset_dir: Path, profile: str) -> Path:
    return optimizer_root(dataset_dir) / profile


def costs_path(dataset_dir: Path, profile: str) -> Path:
    return profile_dir(dataset_dir, profile) / "costs.json"


def semantic_cost_model_path(dataset_dir: Path, profile: str) -> Path:
    return profile_dir(dataset_dir, profile) / "semantic_cost_model.json"


def model_path(dataset_dir: Path, profile: str, class_name: str) -> Path:
    return profile_dir(dataset_dir, profile) / "models" / f"{class_name}_model.json"


def optimizer_lake_run_dir(runs_root: Path, lake: str) -> Path:
    """Backend-independent run artifacts (the sampled query specs)."""
    return Path(runs_root) / "optimizer" / lake


def optimizer_run_dir(runs_root: Path, lake: str, profile: str | None = None) -> Path:
    """Where a run WRITES its backend-dependent CSVs."""
    base = optimizer_lake_run_dir(runs_root, lake)
    return base / profile if profile else base


def resolve_optimizer_run_dir(runs_root: Path, lake: str, profile: str | None,
                              *, subdir: str) -> Path:
    """Where a reader finds them: the profile-keyed dir if it holds `subdir`, else the legacy path."""
    keyed = optimizer_run_dir(runs_root, lake, profile)
    if profile and (keyed / subdir).is_dir():
        return keyed
    legacy = optimizer_lake_run_dir(runs_root, lake)
    if (legacy / subdir).is_dir():
        return legacy
    return keyed


def legacy_costs_path(dataset_dir: Path) -> Path:
    return optimizer_root(dataset_dir) / "costs.json"


def legacy_semantic_cost_model_path(dataset_dir: Path) -> Path:
    return optimizer_root(dataset_dir) / "semantic_cost_model.json"


__all__ = [
    "optimizer_profile",
    "optimizer_lake_run_dir",
    "optimizer_run_dir",
    "resolve_optimizer_run_dir",
    "optimizer_root",
    "freqs_path",
    "profile_dir",
    "costs_path",
    "semantic_cost_model_path",
    "model_path",
    "legacy_costs_path",
    "legacy_semantic_cost_model_path",
]
