import shutil
import subprocess
import sys
from pathlib import Path


FIXTURE = Path(__file__).parent / "fixtures" / "semantic"
SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "create_semantic_index.py"


def test_cli_builds_index(tmp_path):
    root = tmp_path / "data"
    approach_dir = root / "test_ds" / "semantic" / "liftus" / "cli-demo"
    (approach_dir / "ckpt").mkdir(parents=True)
    shutil.copytree(FIXTURE / "ckpt", approach_dir / "ckpt", dirs_exist_ok=True)
    shutil.copytree(FIXTURE / "aspects", approach_dir / "aspects")

    res = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--csv-dir", str(FIXTURE / "csvs"),
         "--approach", "liftus",
         "--index-name", "cli-demo",
         "--dataset", "test_ds",
         "--dataset-root", str(root),
         "--quant", "flat"],
        capture_output=True, text=True, timeout=180,
        env={
            **__import__("os").environ,
            "KMP_DUPLICATE_LIB_OK": "TRUE",
            "OMP_NUM_THREADS": "1",
        },
    )
    assert res.returncode == 0, res.stderr
    index_dir = approach_dir / "index"
    assert (index_dir / "manifest.json").is_file()
    assert (index_dir / "hnsw.faiss").is_file()
