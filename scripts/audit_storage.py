"""Audit a dataset's storage across the duckdb, disk-semantic and postgres backends, one JSON per dataset."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))
sys.path.insert(0, str(_THIS.parent.parent))

import duckdb  # noqa: E402

from audit_embedding_sizes import open_pg  # noqa: E402

_INDEX_SUBDIRS = ("index", "ckpt", "aspects")


def collect_duckdb(path: Path) -> dict:
    if not path.exists():
        return {"present": False, "reason": f"not found: {path}"}
    con = duckdb.connect(str(path), read_only=True)
    try:
        db = con.execute("pragma database_size").fetchdf().iloc[0].to_dict()
        bsz = int(db["block_size"])
        total_blocks = int(db["total_blocks"])
        used_blocks = int(db["used_blocks"])
        tabs = [
            r[0]
            for r in con.execute(
                "select table_name from information_schema.tables "
                "where table_schema='main'"
            ).fetchall()
        ]
        tables = {}
        for t in sorted(tabs):
            blk = con.execute(
                f"select count(distinct block_id) from pragma_storage_info('{t}')"
            ).fetchone()[0]
            rows = con.execute(f"select count(*) from {t}").fetchone()[0]
            tables[t] = {"blocks": int(blk), "bytes": int(blk) * bsz, "rows": int(rows)}
        idx = [
            {
                "index_name": r[0],
                "table_name": r[1],
                "is_unique": bool(r[2]),
                "is_primary": bool(r[3]),
            }
            for r in con.execute(
                "select index_name, table_name, is_unique, is_primary "
                "from duckdb_indexes()"
            ).fetchall()
        ]
    finally:
        con.close()
    table_blocks = sum(v["blocks"] for v in tables.values())
    catalog_blocks = used_blocks - table_blocks
    return {
        "present": True,
        "path": str(path),
        "file_st_size_bytes": path.stat().st_size,
        "block_size_bytes": bsz,
        "total_blocks": total_blocks,
        "used_blocks": used_blocks,
        "free_blocks": total_blocks - used_blocks,
        "indexes": idx,
        "tables": tables,
        "accounting": {
            "table_blocks": table_blocks,
            "table_bytes": table_blocks * bsz,
            "catalog_metadata_blocks": catalog_blocks,
            "used_blocks": used_blocks,
            "reconciles": table_blocks + catalog_blocks == used_blocks,
        },
    }


def collect_disk_semantic(sem_dir: Path) -> dict:
    if not sem_dir.exists():
        return {"present": False, "reason": f"not found: {sem_dir}"}
    root = Path(os.path.realpath(sem_dir))
    approaches = {}
    grand = 0
    for approach in sorted(p.name for p in root.iterdir() if p.is_dir()):
        for index in sorted(
            p.name for p in (root / approach).iterdir() if p.is_dir()
        ):
            base = root / approach / index
            files: dict[str, int] = {}
            sub_totals = {k: 0 for k in _INDEX_SUBDIRS}
            other = 0
            for dirpath, _dirs, fnames in os.walk(base):
                for fn in fnames:
                    fp = Path(dirpath) / fn
                    try:
                        sz = fp.stat().st_size
                    except OSError:
                        continue
                    rel = str(fp.relative_to(base))
                    files[rel] = sz
                    top = rel.split(os.sep, 1)[0]
                    if top in sub_totals:
                        sub_totals[top] += sz
                    else:
                        other += sz
            total = sum(files.values())
            grand += total
            approaches[f"{approach}/{index}"] = {
                "files": files,
                "subdir_totals_bytes": sub_totals,
                "other_bytes": other,
                "total_bytes": total,
            }
    return {"present": True, "root": str(root), "approaches": approaches,
            "total_bytes": grand}


def collect_pg_schema(conn, schema: str) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            "select c.relname from pg_class c join pg_namespace n "
            "on n.oid = c.relnamespace where n.nspname = %s and c.relkind='r' "
            "order by c.relname",
            (schema,),
        )
        table_names = [r[0] for r in cur.fetchall()]
        if not table_names:
            return {"present": False, "schema": schema,
                    "reason": "no tables in schema (dataset not loaded)"}
        tables = {}
        for tab in table_names:
            cur.execute(
                "select pg_relation_size(c.oid), coalesce(pg_relation_size(c.reltoastrelid),0), "
                "pg_total_relation_size(c.oid), c.reltuples::bigint "
                "from pg_class c join pg_namespace n on n.oid=c.relnamespace "
                "where n.nspname=%s and c.relname=%s",
                (schema, tab),
            )
            heap, toast, total, rows = cur.fetchone()
            cur.execute(
                "select i.relname, am.amname, pg_relation_size(i.oid) "
                "from pg_class t join pg_namespace n on n.oid=t.relnamespace "
                "join pg_index ix on ix.indrelid=t.oid "
                "join pg_class i on i.oid=ix.indexrelid "
                "join pg_am am on am.oid=i.relam "
                "where n.nspname=%s and t.relname=%s order by i.relname",
                (schema, tab),
            )
            indexes = {
                r[0]: {"method": r[1], "bytes": int(r[2])} for r in cur.fetchall()
            }
            idx_total = sum(v["bytes"] for v in indexes.values())
            other = int(total) - int(heap) - int(toast) - idx_total
            tables[tab] = {
                "heap_bytes": int(heap),
                "toast_bytes": int(toast),
                "indexes": indexes,
                "index_total_bytes": idx_total,
                "fsm_vm_other_bytes": other,
                "total_bytes": int(total),
                "rows_estimate": int(rows),
                "accounts_for_total": int(heap) + int(toast) + idx_total + other == int(total),
            }
    schema_total = sum(v["total_bytes"] for v in tables.values())
    return {"present": True, "schema": schema, "tables": tables,
            "schema_total_bytes": schema_total}


def rollup(duckdb_r: dict, disk_r: dict, pg_r: dict) -> dict:
    """Per-approach semantic backend totals plus syntactic; ckpt+aspects count toward both backends."""
    r: dict = {"note": "derived; semantic backend totals include the shared on-disk "
                       "ckpt+aspects that both vector backends require"}
    if duckdb_r.get("present"):
        r["syntactic_duckdb_bytes"] = duckdb_r["accounting"]["table_bytes"]
    if pg_r.get("present") and "blend_index" in pg_r["tables"]:
        r["syntactic_pg_bytes"] = pg_r["tables"]["blend_index"]["total_bytes"]

    approaches = disk_r.get("approaches") or {}
    pg_tables = (pg_r.get("tables") or {}) if pg_r.get("present") else {}
    keys = set(approaches)
    for t in pg_tables:
        if t.startswith("semantic_columns__"):
            app, _, idx = t[len("semantic_columns__"):].partition("__")
            keys.add(f"{app}/{idx}")
    sem: dict = {}
    for key in sorted(keys):
        app, idx = key.split("/", 1)
        a = approaches.get(key)
        shared = faiss_store = None
        if a:
            faiss_store = a["subdir_totals_bytes"].get("index", 0)
            shared = a["total_bytes"] - faiss_store
        pg_tab = pg_tables.get(f"semantic_columns__{app}__{idx}")
        pg_store = pg_tab["total_bytes"] if pg_tab else None
        entry = {
            "shared_disk_bytes": shared,
            "faiss_store_bytes": faiss_store,
            "pg_store_bytes": pg_store,
        }
        if shared is not None and faiss_store is not None:
            entry["faiss_backend_total_bytes"] = shared + faiss_store
        if shared is not None and pg_store is not None:
            entry["pg_backend_total_bytes"] = shared + pg_store
        sem[key] = entry
    if sem:
        r["semantic"] = sem
    return r


def audit_one(name: str, duckdb_path: Path | None, sem_dir: Path | None,
              pg_schema: str | None, pg_conn) -> dict:
    duckdb_r = collect_duckdb(duckdb_path) if duckdb_path else {"present": False}
    disk_r = collect_disk_semantic(sem_dir) if sem_dir else {"present": False}
    if pg_schema and pg_conn is not None:
        pg_r = collect_pg_schema(pg_conn, pg_schema)
    elif pg_schema:
        pg_r = {"present": False, "schema": pg_schema, "reason": "postgres unavailable"}
    else:
        pg_r = {"present": False}
    return {
        "dataset": name,
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "duckdb": duckdb_r,
        "disk_semantic": disk_r,
        "postgres": pg_r,
        "rollup_bytes": rollup(duckdb_r, disk_r, pg_r),
    }


def discover(root: Path, db_filename: str) -> list[str]:
    return sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and ((p / db_filename).exists() or (p / "semantic").exists())
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    ap.add_argument("--datasets", nargs="*", help="restrict auto mode to these names")
    ap.add_argument("--db-filename", default="blend.duckdb")
    ap.add_argument("--name", help="explicit single-dataset mode: dataset name")
    ap.add_argument("--duckdb", type=Path, help="explicit: duckdb file path")
    ap.add_argument("--semantic-dir", type=Path, help="explicit: semantic/ dir path")
    ap.add_argument("--pg-schema", help="explicit: Postgres schema name")
    ap.add_argument("--config", type=Path, default=Path("config/config.ini"),
                    help="config.ini for the [Database] Postgres connection")
    ap.add_argument("--no-pg", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=Path("runs/storage"))
    args = ap.parse_args()

    pg_conn = None
    if not args.no_pg:
        pg_conn, params, err = open_pg(args.config)
        if pg_conn is None:
            print(f"WARNING: postgres unavailable ({err}); pg sizes skipped")

    if args.name or args.duckdb or args.semantic_dir or args.pg_schema:
        specs = [(
            args.name or (args.duckdb.parent.name if args.duckdb else "dataset"),
            args.duckdb, args.semantic_dir, args.pg_schema,
        )]
    else:
        names = args.datasets or discover(args.datasets_root, args.db_filename)
        specs = []
        for n in names:
            base = args.datasets_root / n
            db = base / args.db_filename
            sem = base / "semantic"
            specs.append((
                n,
                db if db.exists() else None,
                sem if sem.exists() else None,
                n if not args.no_pg else None,
            ))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, db, sem, sch in tqdm(specs, unit="dataset", desc="audit"):
        rec = audit_one(name, db, sem, sch, pg_conn)
        out = args.out_dir / f"{name}.json"
        out.write_text(json.dumps(rec, indent=2))
        rb = rec["rollup_bytes"]

        def mib(v):
            return "-" if v is None else f"{v / 1048576:.1f}"

        syn = " ".join(f"{k.replace('_bytes', '')}={mib(v)}MiB"
                       for k, v in rb.items() if k.endswith("_bytes"))
        tqdm.write(f"{name}: {syn}")
        for ap, e in rb.get("semantic", {}).items():
            tqdm.write(
                f"    sem {ap}: shared={mib(e['shared_disk_bytes'])} "
                f"faiss_store={mib(e['faiss_store_bytes'])} "
                f"pg_store={mib(e['pg_store_bytes'])} | "
                f"faiss_total={mib(e.get('faiss_backend_total_bytes'))} "
                f"pg_total={mib(e.get('pg_backend_total_bytes'))} MiB"
            )

    if pg_conn is not None:
        pg_conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
