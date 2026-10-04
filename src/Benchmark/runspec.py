"""Run directories, slugs and manifests for every Benchmark subcommand."""
from __future__ import annotations

import csv
import json
import re
import traceback
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

from src import paths

_UNSET = object()
_LIST_CAP = 3


@dataclass(frozen=True)
class Field:
    """One slug/manifest knob. `default` is omitted from the slug when matched."""
    key: str
    prefix: str = ""
    default: Any = _UNSET
    in_slug: bool = True


def sanitize(value: Any) -> str:
    """`.` -> `p` first, so 0.01 stays one token; then non-alphanumeric -> `_`."""
    return re.sub(r"[^A-Za-z0-9]", "_", str(value).replace(".", "p"))


def _render(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        items = sorted(value) if isinstance(value, set) else list(value)
        head = "_".join(sanitize(v) for v in items[:_LIST_CAP])
        extra = len(items) - _LIST_CAP
        return f"{head}+{extra}" if extra > 0 else head
    return sanitize(value)


def render_slug(fields: Sequence[Field], values: Mapping[str, Any]) -> str:
    """`-` between fields, `_` within one. Defaults, None and empty lists drop out."""
    parts: list[str] = []
    for f in fields:
        if not f.in_slug or f.key not in values:
            continue
        val = values[f.key]
        if val is None:
            continue
        if f.default is not _UNSET and val == f.default:
            continue
        if isinstance(val, bool):
            if val:
                parts.append(f.prefix or f.key)
            continue
        if isinstance(val, (list, tuple, set)) and not val:
            continue
        rendered = _render(val)
        if rendered:
            parts.append(f"{f.prefix}{rendered}")
    return "-".join(parts)


@lru_cache(maxsize=1)
def git_sha() -> str:
    """Best-effort; empty in the container, which has no git binary."""
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


def runtime_info(dataset: str | None, db: Any = None) -> dict:
    """Backend identity; layout is the detected one only when `db` is given."""
    if db is not None:
        return {"dbms": db.dbms, "vector_backend": db.vector_backend,
                "layout": db.layout, "profile": db.optimizer_profile_str()}
    import configparser
    from src.Semantic.config import SemanticConfig
    parser = configparser.ConfigParser()
    parser.read(paths.config_path())
    cfg = SemanticConfig.load(
        overrides={"dataset": dataset} if dataset else None)
    return {"dbms": parser.get("Database", "dbms", fallback="duckdb"),
            "vector_backend": cfg.vector_backend, "layout": None, "profile": None}


def backend_slug(rt: Mapping[str, Any]) -> str:
    return f"{rt['dbms']}_{rt['vector_backend']}"


def bench_results_root(lake: str, root: Path | None = None) -> Path:
    """Root of every Benchmark output path, caches included."""
    return (root or paths.datasets_root()) / lake / "bench_results"


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, set):
        return [_jsonable(v) for v in sorted(obj)]
    if isinstance(obj, Path):
        return str(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


@dataclass
class RunDir:
    path: Path
    cmd: str
    lake: str | None
    stamp: str
    tag: str | None
    slug: str
    args: dict
    config: dict
    _t0: float = dc_field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        self._t0 = perf_counter()
        self._write_manifest("running")

    def file(self, name: str) -> Path:
        return self.path / name

    def set_config(self, **kw: Any) -> None:
        """Upgrade the manifest once a handler has an open DBHandler."""
        self.config.update({k: v for k, v in kw.items() if v is not None})

    def write_table(self, rows: list[dict], summary: dict, *, stem: str = "results") -> Path:
        csv_path = self.path / f"{stem}.csv"
        cols = sorted({key for r in rows for key in r}) if rows else ["query"]
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        self.write_json(f"{stem}.json", {
            "cmd": self.cmd, "lake": self.lake,
            "args": _jsonable(self.args), "summary": summary})
        return csv_path

    def write_json(self, name: str, obj: Any) -> Path:
        p = self.path / name
        p.write_text(json.dumps(_jsonable(obj), indent=2))
        return p

    def _write_manifest(self, status: str, error: str | None = None) -> None:
        (self.path / "manifest.json").write_text(json.dumps({
            "cmd": self.cmd,
            "created_utc": self.stamp,
            "tag": self.tag,
            "slug": self.slug,
            "lake": self.lake,
            "args": _jsonable(self.args),
            "config": _jsonable(self.config),
            "git_sha": git_sha(),
            "status": status,
            "error": error,
            "wall_ms": round((perf_counter() - self._t0) * 1000.0, 1),
        }, indent=2))

    def __enter__(self) -> "RunDir":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self._write_manifest("ok")
        else:
            self._write_manifest("failed", error="".join(
                traceback.format_exception_only(exc_type, exc)).strip())
        return False


def open_run(cmd: str, *, lake: str | None, fields: Sequence[Field],
             values: Mapping[str, Any], config: Mapping[str, Any] | None = None,
             tag: str | None = None, out: str | Path | None = None,
             root: Path | None = None) -> RunDir:
    """`<root>/<lake>/bench_results/<cmd>/<stampZ>[-<tag>][-<slug>]/`, or `out` verbatim."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = render_slug(fields, values)
    if out is not None:
        path = Path(out)
    else:
        if not lake:
            raise ValueError(f"{cmd}: cannot place a run dir without a lake; pass --out")
        name = "-".join(p for p in (stamp, sanitize(tag) if tag else "", slug) if p)
        path = bench_results_root(lake, root) / cmd / name
    return RunDir(path=path, cmd=cmd, lake=lake, stamp=stamp, tag=tag, slug=slug,
                  args={f.key: values.get(f.key) for f in fields},
                  config=dict(config or {}))


__all__ = ["Field", "RunDir", "backend_slug", "bench_results_root", "git_sha",
           "open_run", "render_slug", "runtime_info", "sanitize"]
