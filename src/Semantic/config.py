from __future__ import annotations

import configparser
import hashlib
from dataclasses import dataclass, field, fields, replace
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping

from src.dataset import DatasetConfig, load_dataset_config
from src import paths


class SemanticOp(StrEnum):
    """Closed set of semantic operator types that own a `[Semantic.<OP>]` block."""
    SU = "SU"
    SJ = "SJ"
    SHO = "SHO"

    @property
    def ini_section(self) -> str:
        return f"Semantic.{self.value}"


@dataclass(frozen=True)
class OperatorConfig:
    approach: str
    index_name: str
    aggregator: str = "default"
    munkres_threshold: float | None = None
    vote_depth_factor: float | None = None


def _checked_vote_factor(value, source: str) -> float:
    f = float(value)
    if f <= 0:
        raise ValueError(
            f"vote_depth_factor must be > 0, got {f} from {source}; "
            "it multiplies the plan's k to give the vote depth"
        )
    return f


@dataclass(frozen=True)
class SemanticConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    operators: Mapping[SemanticOp, OperatorConfig] = field(
        default_factory=lambda: MappingProxyType({})
    )
    sidecar_filename: str = "blend_index_basenames.parquet"
    db_filename: str = "blend.duckdb"
    faiss_quant: Literal["flat", "pq"] = "flat"
    faiss_hnsw_M: int = 64
    faiss_hnsw_ef_construction: int = 200
    # faiss needs ef_search >= k_coarse to fill the width; not comparable to pgvector at equal value.
    faiss_hnsw_ef_search: int = 64
    faiss_pq_m: int | None = None
    faiss_pq_nbits: int = 8
    faiss_k_coarse: int | None = None
    vote_depth_factor: float = 2.0
    restricted_k_coarse: int = 500
    # Counts gids (columns), not tables.
    exact_threshold: int | None = None
    device: str = "cpu"
    vector_backend: Literal["faiss", "pgvector"] = "faiss"
    pg_pushdown_mode: Literal["prefilter", "postfilter"] = "prefilter"
    # Never `off`: pgvector then returns fewer rows than LIMIT whenever ef_search < LIMIT.
    pgvector_hnsw_iterative_scan: str = "relaxed_order"
    semantic_ml_cost: Literal["off", "analytic"] = "off"
    # off (default): a query column absent from the index is a hard error; auto: encode it from its cells.
    query_encode: Literal["auto", "off"] = "off"

    @classmethod
    def load(
        cls,
        path: Path | None = None,
        overrides: Mapping[str, Any] | None = None,
        operators: Mapping[SemanticOp, OperatorConfig] | None = None,
    ) -> "SemanticConfig":
        cfg = cls()
        ini_path = Path(path) if path is not None else paths.config_path()
        raw_semantic: dict[str, Any] = {}
        parsed_ops: dict[SemanticOp, OperatorConfig] = {}
        if ini_path.exists():
            parser = configparser.ConfigParser()
            parser.read(ini_path)
            if parser.has_section("Semantic"):
                raw_semantic.update(dict(parser.items("Semantic")))
            if parser.has_section("Database") and parser.has_option("Database", "db_filename"):
                raw_semantic.setdefault("db_filename", parser.get("Database", "db_filename"))
            for op in SemanticOp:
                if parser.has_section(op.ini_section):
                    sect = parser[op.ini_section]
                    parsed_ops[op] = OperatorConfig(
                        approach=sect["approach"],
                        index_name=sect["index_name"],
                        aggregator=sect.get("aggregator", "default"),
                        munkres_threshold=(
                            float(sect["munkres_threshold"])
                            if "munkres_threshold" in sect else None
                        ),
                        vote_depth_factor=(
                            float(sect["vote_depth_factor"])
                            if "vote_depth_factor" in sect else None
                        ),
                    )
        if overrides:
            raw_semantic.update(overrides)
        if operators:
            parsed_ops.update(operators)
        cfg = replace(cfg, dataset=load_dataset_config(ini_path, overrides))
        cfg = cfg._merge_semantic(raw_semantic)
        cfg = replace(cfg, operators=MappingProxyType(dict(parsed_ops)))
        if cfg.query_encode not in ("auto", "off"):
            raise ValueError(
                f"[Semantic] query_encode must be 'auto' or 'off', got {cfg.query_encode!r}"
            )
        if cfg.vector_backend == "pgvector":
            cfg = replace(cfg, exact_threshold=None)
        return cfg

    def operator(self, op: SemanticOp) -> OperatorConfig:
        try:
            return self.operators[op]
        except KeyError:
            raise KeyError(
                f"semantic op {op.value} not configured; "
                f"add section [{op.ini_section}] with approach= and index_name= "
                f"to your config.ini, or pass operators={{SemanticOp.{op.value}: "
                f"OperatorConfig(...)}} to SemanticConfig.load()"
            ) from None

    def vote_factor_for(self, op: SemanticOp, *, override: float | None = None) -> float:
        """Resolve the vote depth factor: override > [Semantic.<OP>] > [Semantic]."""
        if override is not None:
            return _checked_vote_factor(override, "override")
        per_op = self.operators.get(op)
        if per_op is not None and per_op.vote_depth_factor is not None:
            return _checked_vote_factor(per_op.vote_depth_factor, op.ini_section)
        return _checked_vote_factor(self.vote_depth_factor, "Semantic")

    @property
    def duckdb_path(self) -> Path:
        return self.dataset.dir() / self.db_filename

    def approach_dir(self, approach: str, index_name: str) -> Path:
        return self.dataset.dir() / "semantic" / approach / index_name

    def index_dir(self, approach: str, index_name: str) -> Path:
        return self.approach_dir(approach, index_name) / "index"

    @property
    def blend_basenames_path(self) -> Path:
        return self.dataset.dir() / self.sidecar_filename

    def signature(self) -> str:
        ops_material = tuple(
            (op.value, oc.approach, oc.index_name)
            for op, oc in sorted(self.operators.items(), key=lambda kv: kv[0].value)
        )
        material = "|".join(
            f"{name}={value!r}" for name, value in (
                ("dataset.name", self.dataset.name),
                ("dataset.root", self.dataset.root),
                ("operators", ops_material),
                ("sidecar_filename", self.sidecar_filename),
                ("db_filename", self.db_filename),
                ("vector_backend", self.vector_backend),
                ("pg_pushdown_mode", self.pg_pushdown_mode),
                ("pgvector_hnsw_iterative_scan", self.pgvector_hnsw_iterative_scan),
                ("faiss_hnsw_ef_search", self.faiss_hnsw_ef_search),
                ("faiss_k_coarse", self.faiss_k_coarse),
            )
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]

    def _merge_semantic(self, raw: Mapping[str, Any]) -> "SemanticConfig":
        kwargs: dict[str, Any] = {}
        skip = {"dataset", "dataset_root", "operators"}
        valid = {f.name: f.type for f in fields(self) if f.name not in skip}
        for key, value in raw.items():
            name = key.strip()
            if name in skip or name not in valid:
                continue
            target = valid[name]
            type_tokens = {t.strip() for t in str(target).split("|")}
            optional = "None" in type_tokens
            if optional and value in (None, "", "None"):
                kwargs[name] = None
            elif "int" in type_tokens:
                kwargs[name] = int(value)
            elif "float" in type_tokens:
                kwargs[name] = float(value)
            elif "bool" in type_tokens:
                kwargs[name] = str(value).lower() in ("1", "true", "yes", "on")
            else:
                kwargs[name] = str(value)
        return replace(self, **kwargs)


__all__ = ["SemanticConfig", "DatasetConfig", "SemanticOp", "OperatorConfig"]
