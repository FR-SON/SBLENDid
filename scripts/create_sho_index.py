"""Build the SHO simhash-code index for a dataset (stages sample, hash, materialize; default all)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import duckdb
import numpy as np
import pandas as pd
from tqdm import tqdm

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

from src.Semantic.emb_store import PersistentEmbeddingStore, resolve_embeddings  # noqa: E402
from src.Semantic.simhash import (  # noqa: E402
    SHO_ENCODER_MODEL, codes_from_embeddings, get_hyperplanes,
    is_numeric_column, sample_row_indices, table_seed,
)


def run_sample(csv_dir: Path, out_dir: Path, sidecar_path: Path, *,
               row_cap: int, seed: int, skip_numeric: bool) -> dict:
    if not sidecar_path.is_file():
        raise FileNotFoundError(
            f"basenames sidecar not found: {sidecar_path} — run "
            "scripts/create_blend_csv_index.py for this dataset first")
    sc = pd.read_parquet(sidecar_path)
    b2i = dict(zip(sc["basename"].astype(str), sc["table_int_id"].astype(int)))
    csv_paths = sorted(Path(csv_dir).glob("*.csv"), key=lambda p: p.name)
    if not csv_paths:
        raise FileNotFoundError(f"no CSVs under {csv_dir}")

    value_ids: dict[str, int] = {}
    tid_parts, colid_parts, rowid_parts, vid_parts = [], [], [], []
    stats = {"n_tables": 0, "n_cols_skipped_numeric": 0, "n_missing_sidecar": 0}
    for path in tqdm(csv_paths, unit="tbl", desc="sho sample"):
        tid = b2i.get(path.name)
        if tid is None:
            stats["n_missing_sidecar"] += 1
            continue
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        if df.empty:
            continue
        idx = sample_row_indices(len(df), row_cap, table_seed(seed, path.name))
        rows = np.asarray(idx)
        sub = df if rows.shape[0] == len(df) else df.iloc[idx]
        stats["n_tables"] += 1
        for colid, col in enumerate(df.columns):
            s = sub[col]
            keep = (s.to_numpy() != "") & (s.str.lower().to_numpy() != "nan")
            if not keep.any():
                continue
            kv = s.to_numpy()[keep]
            if skip_numeric and is_numeric_column(kv.tolist()):
                stats["n_cols_skipped_numeric"] += 1
                continue
            # factorize keeps first-appearance order, matching per-cell setdefault id numbering
            codes, uniq = pd.factorize(kv)
            new_ids = np.fromiter(
                (value_ids.setdefault(v, len(value_ids)) for v in uniq),
                dtype=np.int64, count=len(uniq))
            tid_parts.append(np.full(kv.shape[0], tid, dtype=np.int32))
            colid_parts.append(np.full(kv.shape[0], colid, dtype=np.int32))
            rowid_parts.append(rows[keep].astype(np.int32))
            vid_parts.append(new_ids[codes])

    def _cat(parts, dtype):
        return np.concatenate(parts) if parts else np.empty(0, dtype=dtype)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"value_id": np.arange(len(value_ids), dtype=np.int64),
                  "value": list(value_ids.keys())}
                 ).to_parquet(out_dir / "values.parquet")
    cells = {"tableid": _cat(tid_parts, np.int32), "colid": _cat(colid_parts, np.int32),
             "rowid": _cat(rowid_parts, np.int32), "value_id": _cat(vid_parts, np.int64)}
    pd.DataFrame({k: pd.array(v, dtype="int32" if k != "value_id" else "int64")
                  for k, v in cells.items()}).to_parquet(out_dir / "cells.parquet")
    (out_dir / "sample_meta.json").write_text(json.dumps(
        {"row_cap": row_cap, "seed": seed, "skip_numeric": skip_numeric}))
    stats["n_values"] = len(value_ids)
    stats["n_cells"] = int(cells["tableid"].shape[0])
    return stats


def run_hash(out_dir: Path, *, bits: int, seed: int, encode_fn=None,
            store_model: str = SHO_ENCODER_MODEL, batch: int = 100_000) -> dict:
    if bits > 30:
        raise ValueError("bits must be <= 30 (codes stored as int32)")
    values = pd.read_parquet(out_dir / "values.parquet")
    sample_meta = json.loads((out_dir / "sample_meta.json").read_text())
    store = PersistentEmbeddingStore(out_dir / "emb_store", model=store_model)
    vals = values["value"].tolist()
    n = len(vals)
    print(f"resolving {n} value embeddings (store has {store.count})...", flush=True)
    sho_codes = np.empty(n, dtype=np.int32)
    hp = None
    dim = 0
    for i in tqdm(range(0, n, batch), unit="batch", desc="sho hash"):
        chunk = vals[i:i + batch]
        emb = resolve_embeddings(chunk, store, encode_fn=encode_fn)
        if hp is None:
            dim = int(emb.shape[1])
            hp = get_hyperplanes(bits, dim, seed)
        sho_codes[i:i + len(chunk)] = codes_from_embeddings(emb, hp).astype(np.int32)
    values = values.assign(simhash_code=sho_codes)

    t = perf_counter()
    print("reading cells.parquet + merging value codes...", flush=True)
    cells = pd.read_parquet(out_dir / "cells.parquet")
    codes = cells.merge(values[["value_id", "simhash_code"]], on="value_id")
    codes = codes[["simhash_code", "tableid", "colid", "rowid"]].astype("int32")
    print(f"writing codes.parquet ({len(codes)} rows)...", flush=True)
    codes.to_parquet(out_dir / "codes.parquet")
    print(f"cells merge + write: {perf_counter() - t:.0f}s", flush=True)
    manifest = {
        "bits": bits, "seed": seed, "sample_seed": sample_meta["seed"],
        "row_cap": sample_meta["row_cap"], "skip_numeric": sample_meta["skip_numeric"],
        "encoder_model": store_model, "dim": dim,
        "n_tables": int(cells["tableid"].nunique()), "n_codes": int(len(codes)),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def materialize_duckdb(duckdb_path: Path, codes_path: Path) -> None:
    con = duckdb.connect(str(duckdb_path), read_only=False)
    try:
        con.execute("DROP TABLE IF EXISTS blend_index_code")
        con.execute(
            "CREATE TABLE blend_index_code AS "
            "SELECT simhash_code, tableid, colid, rowid "
            "FROM read_parquet(?) "
            "ORDER BY simhash_code, tableid, colid", [str(codes_path)])
    finally:
        con.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-dir", type=Path, default=None,
                    help="required for --stage sample/all")
    ap.add_argument("--dataset", default=None, help="override [Dataset].name")
    ap.add_argument("--index-name", default=None,
                    help="default: [Semantic.SHO] index_name")
    ap.add_argument("--stage", choices=["sample", "hash", "materialize", "all"],
                    default="all", help="materialize = load an existing "
                    "codes.parquet into the backend (no sample/hash, no emb store)")
    ap.add_argument("--row-cap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bits", type=int, default=18)
    ap.add_argument("--include-numeric", action="store_true",
                    help="default excludes numeric columns (SemDisc rule)")
    ap.add_argument("--materialize", choices=["duckdb", "postgres", "both", "none"],
                    default=None, help="default: [Database].dbms")
    ap.add_argument("--device", default="cpu", help="local-encode device (--stage all)")
    args = ap.parse_args()

    from src.Semantic.config import SemanticConfig, SemanticOp
    overrides = {"dataset": args.dataset} if args.dataset else None
    cfg = SemanticConfig.load(overrides=overrides)
    op = cfg.operator(SemanticOp.SHO)
    index_name = args.index_name or op.index_name
    out_dir = cfg.approach_dir(op.approach, index_name)

    t0 = perf_counter()
    if args.stage in ("sample", "all"):
        if args.csv_dir is None:
            ap.error("--csv-dir is required for --stage sample/all")
        stats = run_sample(args.csv_dir, out_dir, cfg.blend_basenames_path,
                           row_cap=args.row_cap, seed=args.seed,
                           skip_numeric=not args.include_numeric)
        print(f"sample: {stats}")
    if args.stage in ("hash", "all"):
        encode_fn = None
        if args.stage == "all":
            def encode_fn(chunk, _cache={}):
                if "fn" not in _cache:
                    from src.Semantic.encoders.sbert import build_encode_fn
                    _cache["fn"] = build_encode_fn(SHO_ENCODER_MODEL, device=args.device)
                return _cache["fn"](chunk)
        manifest = run_hash(out_dir, bits=args.bits, seed=args.seed,
                            encode_fn=encode_fn)
        print(f"hash: {manifest}")
    if args.stage in ("hash", "all", "materialize"):
        codes_path = out_dir / "codes.parquet"
        if not codes_path.is_file():
            ap.error(f"{codes_path} not found — run --stage hash (or drop in a "
                     "prebuilt codes.parquet) before --stage materialize")
        target = args.materialize
        if target is None:
            import configparser
            from src import paths
            cp = configparser.ConfigParser()
            cp.read(str(paths.config_path()))
            target = cp["Database"].get("dbms", "duckdb") if cp.has_section("Database") else "duckdb"
        if target in ("duckdb", "both"):
            materialize_duckdb(cfg.duckdb_path, codes_path)
            print(f"materialized duckdb: {cfg.duckdb_path} (blend_index_code)")
        if target in ("postgres", "both"):
            from scripts.load_sho_index_pg import load_codes_into_pg
            load_codes_into_pg(codes_path, cfg.dataset.name)
            print("materialized postgres: blend_index_code")
    print(f"done in {perf_counter() - t0:.1f}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
