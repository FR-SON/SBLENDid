import configparser
import tempfile
from pathlib import Path

import pandas as pd

from src.dataset import load_dataset_config
from src.DBHandler import DBHandler
from src import paths

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = paths.config_path()


def resolve_scan_table(table_names, index_table, prefer="token") -> str:
    """Return the physical index table a raw full-scan should read ('token' or 'tableid' copy)."""
    names = set(table_names)
    if index_table in names:
        return index_table
    tok, tid = f"{index_table}_token", f"{index_table}_tableid"
    if tok in names and tid in names:
        return tid if prefer == "tableid" else tok
    raise RuntimeError(
        f"{index_table} absent and no two_table copies ({tok}/{tid}) in {sorted(names)}"
    )


def _overrides(lake, root):
    ov = {"dataset": lake}
    if root is not None:
        ov["dataset_root"] = str(root)
    return ov


def lake_dir(lake, *, config_path=CONFIG_PATH, root=None) -> Path:
    return load_dataset_config(config_path, overrides=_overrides(lake, root)).dir()


def lake_dbms(*, config_path=CONFIG_PATH) -> str:
    """Configured backend, without opening a connection."""
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    return cfg["Database"]["dbms"].lower()


def _db_filename(config_path):
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    return cfg["Database"].get("db_filename", "blend.duckdb")


def lake_duckdb_path(lake, *, config_path=CONFIG_PATH, root=None) -> Path:
    return lake_dir(lake, config_path=config_path, root=root) / _db_filename(config_path)


def lake_csv_dir(lake, *, config_path=CONFIG_PATH, root=None) -> Path:
    return lake_dir(lake, config_path=config_path, root=root) / "csvs"


ROWID_SLICE_FILENAME = "blend_index_rowid256.duckdb"


def lake_rowid_slice_path(lake, *, config_path=CONFIG_PATH, root=None) -> Path:
    return lake_dir(lake, config_path=config_path, root=root) / ROWID_SLICE_FILENAME


def open_lake_db(lake, *, config_path=CONFIG_PATH, db_filename=None) -> DBHandler:
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    cfg["Dataset"]["name"] = lake
    if db_filename is not None:
        cfg["Database"]["db_filename"] = db_filename
    tmp = Path(tempfile.mkdtemp()) / "config.ini"
    with open(tmp, "w") as f:
        cfg.write(f)
    handler = DBHandler()
    handler.load_config(tmp)
    handler.index_table = cfg["Database"]["index_table"]
    handler.layout = handler._detect_layout()
    return handler


def use_lake(lake, *, config_path=CONFIG_PATH) -> DBHandler:
    from src.Operators.OperatorBase import Operator  # connects at import

    handler = open_lake_db(lake, config_path=config_path)
    Operator.DB = handler
    return handler


def load_freqs(path) -> None:
    df = pd.read_csv(path)
    DBHandler.frequency_dict = dict(zip(df["tokenized"].astype(str), df["frequency"]))
