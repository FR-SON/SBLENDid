import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# Set before torch/faiss import: duplicate libomp on macOS segfaults FAISS HNSW.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")


def _default_blend_config() -> None:
    """Point BLEND_CONFIG at an empty DuckDB unless set: operators connect at import."""
    if os.environ.get("BLEND_CONFIG"):
        return
    import duckdb
    root = Path(tempfile.mkdtemp(prefix="blend-tests-"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    (root / "_unit").mkdir()
    duckdb.connect(str(root / "_unit" / "blend.duckdb")).close()
    cfg = root / "config.ini"
    cfg.write_text(
        f"[Dataset]\nname = _unit\nroot = {root}\n\n"
        "[Database]\ndbms = duckdb\ndb_filename = blend.duckdb\nindex_table = blend_index\n\n"
        "[Semantic]\nsidecar_filename = blend_index_basenames.parquet\ndevice = cpu\n"
        "vector_backend = faiss\n\n"
        "[Semantic.SU]\napproach = liftus\nindex_name = default\n\n"
        "[Semantic.SJ]\napproach = snoopy\nindex_name = default\n\n"
        "[Semantic.SHO]\napproach = simhash\nindex_name = default\n\n"
        "[Optimizer]\ncost_basis = logical\n"
    )
    os.environ["BLEND_CONFIG"] = str(cfg)


_default_blend_config()

pytest_plugins = ["tests.test_semantic_join_seeker", "tests.test_bench_engine"]


def drop_blend_modules() -> None:
    """Force re-import of src modules that bind DBHandler at class level."""
    for name in list(sys.modules):
        if name == "src" or name.startswith("src."):
            if name.startswith("src.Semantic") and not name.startswith("src.Semantic.seekers"):
                continue
            # evicting source_ingest breaks multiprocessing pool pickling
            if name == "src.Index.source_ingest" or name.startswith("src.Index.source_ingest."):
                continue
            del sys.modules[name]


@pytest.fixture
def duckdb_unit_env(monkeypatch, tmp_path):
    """Point BLEND_CONFIG at a throwaway duckdb config so operator imports never touch Postgres."""
    import duckdb
    ds_dir = tmp_path / "_unit_test"
    ds_dir.mkdir()
    db_path = ds_dir / "empty.duckdb"
    duckdb.connect(str(db_path)).close()

    cfg = tmp_path / "config.ini"
    cfg.write_text(
        f"[Dataset]\nname = _unit_test\nroot = {tmp_path}\n\n"
        f"[Database]\ndbms = duckdb\ndb_filename = {db_path.name}\nindex_table = blend_index\n\n"
        "[Semantic]\nsidecar_filename = blend_index_basenames.parquet\ndevice = cpu\n\n"
        "[Semantic.SHO]\napproach = simhash\nindex_name = default\n"
    )
    monkeypatch.setenv("BLEND_CONFIG", str(cfg))
    drop_blend_modules()
    sho_mod = sys.modules.get("src.Semantic.sho_artifacts")
    if sho_mod is not None:
        sho_mod.ShoHandle._CACHE.clear()
    yield
    drop_blend_modules()


@pytest.fixture
def pg_conn():
    """Live Postgres connection in a throwaway 'pytest_pg' schema; skips if unreachable."""
    import src.Semantic.pgvector_store as store
    try:
        conn = store.connect("public", autocommit=True)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no Postgres: {e}")
    try:
        store.ensure_schema(conn, "pytest_pg")
        with conn.cursor() as cur:
            cur.execute('SET search_path TO "pytest_pg", public')
        yield conn
    finally:
        with conn.cursor() as cur:
            cur.execute('DROP SCHEMA IF EXISTS "pytest_pg" CASCADE')
        conn.close()


@pytest.fixture
def pg_semantic_index(pg_conn, tmp_path, monkeypatch):
    """Build the liftus/demo fixture index under a tmp dataset root and load it into pytest_pg."""
    import shutil
    from pathlib import Path
    import pandas as pd

    monkeypatch.setenv("BLEND_DATASETS_DIR", str(tmp_path))
    from src.Semantic.config import SemanticConfig
    from src.Semantic.index_build import build_semantic_index
    from src.Semantic.retrieve import reset_semantic_cache
    from scripts.load_semantic_index_pg import load_semantic_index_pg

    fix = Path(__file__).parent / "fixtures" / "semantic"
    cfg = SemanticConfig.load(overrides={"dataset": "pytest_pg",
                                         "vector_backend": "faiss",
                                         "faiss_quant": "flat"})
    approach_dir = cfg.approach_dir("liftus", "demo")
    approach_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(fix / "ckpt", approach_dir / "ckpt", dirs_exist_ok=True)
    shutil.copytree(fix / "aspects", approach_dir / "aspects")

    basenames = sorted(p.name for p in (fix / "csvs").glob("*.csv"))
    cfg.blend_basenames_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [(i, b) for i, b in enumerate(basenames)],
        columns=["table_int_id", "basename"],
    ).astype({"table_int_id": "int64", "basename": "string"}).to_parquet(
        cfg.blend_basenames_path, index=False)

    reset_semantic_cache()
    build_semantic_index(fix / "csvs", "liftus", "demo", cfg)
    load_semantic_index_pg("pytest_pg", "liftus", "demo")
    reset_semantic_cache()
    yield "pytest_pg", "liftus", "demo"
