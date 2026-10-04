from __future__ import annotations

import configparser
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from src import paths

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class DatasetConfig:
    name: str = "default"
    root: Path = field(default_factory=paths.datasets_root)

    def dir(self) -> Path:
        return self.root / self.name


def load_dataset_config(
    config_path: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> DatasetConfig:
    raw: dict[str, Any] = {}
    if config_path is not None and Path(config_path).exists():
        parser = configparser.ConfigParser()
        parser.read(config_path)
        if parser.has_section("Dataset"):
            raw.update(dict(parser.items("Dataset")))
    name = raw.get("name", DatasetConfig().name)
    root_raw = raw.get("root")
    env_root = os.environ.get("BLEND_DATASETS_DIR")
    if env_root:
        root_raw = env_root
    if overrides:
        if "dataset" in overrides:
            name = overrides["dataset"]
        if overrides.get("dataset_root") is not None:
            root_raw = overrides["dataset_root"]
    root = (
        Path(str(root_raw)).expanduser()
        if root_raw is not None
        else paths.datasets_root()
    )
    if not root.is_absolute():
        root = (_PROJECT_ROOT / root).resolve()
    return DatasetConfig(name=str(name), root=root)


__all__ = ["DatasetConfig", "load_dataset_config"]
