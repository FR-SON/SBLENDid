#!/usr/bin/env python3
"""Audit the on-disk and pgvector storage footprint of every built embedding store."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent
sys.path.insert(0, str(_REPO))

from src.dataset import load_dataset_config  # noqa: E402

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_VECTOR_AMS = {"hnsw", "diskann", "ivfflat"}
_MIB = 1024 * 1024


def _mib(nbytes: int | None) -> float | None:
    return None if nbytes is None else round(nbytes / _MIB, 3)


def _ratio(num: int | None, den: int | None) -> float | None:
    if not num or not den:
        return None
    return round(num / den, 3)


@dataclass
class IndexRef:
    """One built (dataset, approach, index_name) embedding set on disk."""

    dataset: str
    approach: str
    index_name: str
    index_dir: Path
    rows: int
    dim: int

    @property
    def pg_schema(self) -> str:
        return self.dataset

    @property
    def pg_table(self) -> str:
        return f"semantic_columns__{self.approach}__{self.index_name}"


def discover(root: Path) -> list[IndexRef]:
    """Every <root>/<dataset>/semantic/<approach>/<index>/index/ with embeddings."""
    refs: list[IndexRef] = []
    for npy in sorted(root.glob("*/semantic/*/*/index/embeddings.fp32.npy")):
        index_dir = npy.parent
        index_name = index_dir.parent.name
        approach = index_dir.parent.parent.name
        dataset = index_dir.parent.parent.parent.parent.name
        shape = np.load(npy, mmap_mode="r").shape
        rows = int(shape[0])
        dim = int(shape[1]) if len(shape) > 1 else 0
        refs.append(IndexRef(dataset, approach, index_name, index_dir, rows, dim))
    return refs


def collect_disk(ref: IndexRef, ctx: dict) -> dict:
    files = {p.name: p.stat().st_size for p in sorted(ref.index_dir.iterdir())
             if p.is_file()}
    raw = files.get("embeddings.fp32.npy", 0)
    faiss_name = next((n for n in files if n.endswith(".faiss")), None)
    ann = files.get(faiss_name, 0) if faiss_name else 0
    registry = files.get("registry.parquet", 0)
    sidecar = files.get("embeddings.sidecar.json", 0)
    total = sum(files.values())
    return {
        "raw_embeddings_bytes": raw,
        "ann_index_bytes": ann,
        "ann_index_file": faiss_name,
        "registry_bytes": registry,
        "sidecar_bytes": sidecar,
        "total_bytes": total,
        "total_mib": _mib(total),
        "files": files,
    }


def open_pg(config_path: Path):
    """Best-effort connection. Returns (conn, params, error). conn None if down."""
    try:
        import configparser

        import psycopg
    except Exception as exc:
        return None, {}, f"import failed: {exc}"
    params: dict = {}
    parser = configparser.ConfigParser()
    if config_path.exists():
        parser.read(config_path)
        if parser.has_section("Database"):
            sect = parser["Database"]
            for k in ("host", "port", "user", "password", "dbname"):
                if sect.get(k):
                    params[k] = sect.get(k)
    try:
        conn = psycopg.connect(**params)
        conn.autocommit = True
        return conn, {k: v for k, v in params.items() if k != "password"}, None
    except Exception as exc:
        return None, {k: v for k, v in params.items() if k != "password"}, str(exc)


def collect_pg(ref: IndexRef, ctx: dict) -> dict:
    conn = ctx.get("pg_conn")
    if conn is None:
        return {"present": False, "reason": ctx.get("pg_error", "postgres unavailable")}
    if not (_IDENT.match(ref.pg_schema) and _IDENT.match(ref.pg_table)):
        return {"present": False, "reason": "unsafe schema/table identifier"}
    schema, table = ref.pg_schema, ref.pg_table
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (f'"{schema}"."{table}"',))
        if cur.fetchone()[0] is None:
            return {"present": False, "schema": schema, "table": table,
                    "reason": "table not loaded in postgres"}
        cur.execute(
            """
            SELECT pg_relation_size(c.oid),
                   COALESCE(pg_relation_size(c.reltoastrelid), 0),
                   pg_total_relation_size(c.oid),
                   c.reltuples::bigint
            FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = %s AND c.relname = %s
            """,
            (schema, table),
        )
        heap, toast, total, rows = cur.fetchone()
        cur.execute(
            """
            SELECT i.relname, am.amname, pg_relation_size(i.oid)
            FROM pg_class t JOIN pg_namespace n ON n.oid = t.relnamespace
            JOIN pg_index ix ON ix.indrelid = t.oid
            JOIN pg_class i ON i.oid = ix.indexrelid
            JOIN pg_am am ON am.oid = i.relam
            WHERE n.nspname = %s AND t.relname = %s
            ORDER BY i.relname
            """,
            (schema, table),
        )
        idx_rows = cur.fetchall()
    ann, btrees = [], []
    for name, method, nbytes in idx_rows:
        entry = {"name": name, "method": method, "bytes": int(nbytes)}
        (ann if method in _VECTOR_AMS else btrees).append(entry)
    ann_bytes = sum(e["bytes"] for e in ann)
    btree_bytes = sum(e["bytes"] for e in btrees)
    return {
        "present": True,
        "schema": schema,
        "table": table,
        "rows_estimate": int(rows),
        "heap_bytes": int(heap),
        "toast_bytes": int(toast),
        "ann_indexes": ann,
        "ann_index_bytes": ann_bytes,
        "btrees": btrees,
        "btree_total_bytes": btree_bytes,
        "total_bytes": int(total),
        "total_mib": _mib(int(total)),
    }


BACKENDS = [
    ("disk", collect_disk),
    ("pgvector", collect_pg),
]


def ratios(disk: dict, pg: dict) -> dict:
    if not pg.get("present"):
        return {}
    return {
        "pg_total_over_disk_total": _ratio(pg["total_bytes"], disk["total_bytes"]),
        "pg_ann_over_disk_ann": _ratio(pg["ann_index_bytes"], disk["ann_index_bytes"]),
        "pg_heap_over_raw_embeddings": _ratio(pg["heap_bytes"],
                                              disk["raw_embeddings_bytes"]),
    }


def build_report(refs: list[IndexRef], ctx: dict) -> dict:
    datasets: dict[str, dict] = {}
    for ref in tqdm(refs, desc="sizing", unit="idx"):
        entry = {"approach": ref.approach, "index_name": ref.index_name,
                 "dim": ref.dim, "rows": ref.rows}
        for name, collector in BACKENDS:
            entry[name] = collector(ref, ctx)
        entry["ratios"] = ratios(entry["disk"], entry["pgvector"])
        ds = datasets.setdefault(ref.dataset, {"schema": ref.pg_schema, "indexes": []})
        ds["indexes"].append(entry)
    return datasets


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="restrict to these dataset names; warn on any missing")
    ap.add_argument("--datasets-root", default=None,
                    help="override the datasets root (default: from config / src.dataset)")
    ap.add_argument("--config", default=str(_REPO / "config" / "config.ini"),
                    help="config.ini for datasets root + [Database] connection")
    ap.add_argument("--out", default=None,
                    help="output JSON path (default: <runs>/embedding_sizes/<UTC>.json)")
    ap.add_argument("--no-pg", action="store_true", help="skip the pgvector backend")
    args = ap.parse_args()

    config_path = Path(args.config)
    if args.datasets_root:
        root = Path(args.datasets_root).expanduser().resolve()
    else:
        root = load_dataset_config(config_path).root
    if not root.is_dir():
        print(f"ERROR: datasets root {root} does not exist", file=sys.stderr)
        return 2

    refs = discover(root)
    warnings: list[str] = []
    requested = args.datasets
    if requested:
        present = {r.dataset for r in refs}
        for name in requested:
            if name not in present:
                msg = f"requested dataset {name!r} has no embeddings under {root}"
                warnings.append(msg)
                print(f"WARNING: {msg}", file=sys.stderr)
        refs = [r for r in refs if r.dataset in set(requested)]

    ctx: dict = {}
    pg_meta = {"available": False, "params": {}, "error": "skipped (--no-pg)"}
    if not args.no_pg:
        conn, params, error = open_pg(config_path)
        ctx["pg_conn"], ctx["pg_error"] = conn, error
        pg_meta = {"available": conn is not None, "params": params, "error": error}
        if conn is None:
            print(f"WARNING: postgres unavailable ({error}); disk sizes only",
                  file=sys.stderr)

    datasets = build_report(refs, ctx)

    runs_dir = Path(os.environ.get("BLEND_RUNS_DIR", _REPO / "runs"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else runs_dir / "embedding_sizes" / f"{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tool": _THIS.name,
        "datasets_root": str(root),
        "config": str(config_path),
        "backends": [name for name, _ in BACKENDS],
        "pg": pg_meta,
        "requested_datasets": requested,
        "missing_datasets": [w.split("'")[1] for w in warnings],
        "warnings": warnings,
        "datasets": datasets,
    }
    out.write_text(json.dumps(report, indent=2))

    if ctx.get("pg_conn") is not None:
        ctx["pg_conn"].close()

    _print_summary(datasets)
    print(f"\nwrote {out}")
    return 0


def _print_summary(datasets: dict) -> None:
    for ds, body in datasets.items():
        print(f"\n{ds}")
        for e in body["indexes"]:
            disk = e["disk"]["total_mib"]
            pg = e["pgvector"].get("total_mib")
            r = e["ratios"].get("pg_total_over_disk_total")
            pg_str = f"{pg:>8.2f}" if pg is not None else "     n/a"
            r_str = f"{r:.2f}x" if r is not None else " -"
            print(f"  {e['approach']:<7} {e['index_name']:<8} "
                  f"dim={e['dim']:<4} rows={e['rows']:<7} "
                  f"disk={disk:>8.2f} MiB  pg={pg_str} MiB  pg/disk={r_str}")


if __name__ == "__main__":
    raise SystemExit(main())
