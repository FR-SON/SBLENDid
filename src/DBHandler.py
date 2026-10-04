import os
import random
import configparser
from pathlib import Path
import pandas as pd
import time

# Typing imports
from typing import List, Union, Tuple, Iterable
from numbers import Number

from src.Index.tokenize import tokenize_cell
from src.index_routing import physical_table
from src import paths


class DBHandler(object):
    USE_ML_OPTIMIZER = os.environ.get("BLEND_USE_ML_OPTIMIZER", "").strip().lower() in ("1", "true", "yes")
    frequency_dict = None

    def __init__(self) -> None:
        self.connection = None
        self.cursor = None
        self.index_table = None
        self.layout = 'single'
        self.dbms = None
        self._dataset_dir = None
        self.vector_backend = 'faiss'

        config_path = paths.config_path()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")
        self.load_config(config_path)

    def load_config(self, config_path: Path) -> None:
        self.close()

        config = configparser.ConfigParser()
        config.read(config_path)

        dbms = config['Database']['dbms'].lower()
        self.dbms = dbms
        if dbms == 'vertica':
            import vertica_python
            self.connection = vertica_python.connect(
                host=config['Database']['host'],
                port=config['Database']['port'],
                user=config['Database']['user'],
                password=config['Database']['password'],
                database=config['Database']['dbname'],
                session_label='some_label',
                read_timeout=60000,
                unicode_error='strict',
                ssl=False,
                use_prepared_statements=False
            )
        elif dbms == 'postgres':
            import psycopg as pg
            self.connection = pg.connect(
                host=config['Database']['host'],
                port=config['Database']['port'],
                user=config['Database']['user'],
                password=config['Database']['password'],
                dbname=config['Database']['dbname'],
            )
            from src.dataset import load_dataset_config
            from src.Semantic.pgvector_store import _check_ident
            ds = load_dataset_config(config_path)
            _check_ident(ds.name, "dataset", allow_dash=True)
            with self.connection.cursor() as cur:
                cur.execute(f'SET search_path TO "{ds.name}", public')
            self.connection.commit()
        elif dbms == 'duckdb':
            import duckdb
            from src.dataset import load_dataset_config
            ds = load_dataset_config(config_path)
            db_filename = config['Database'].get('db_filename', 'blend.duckdb')
            duckdb_path = ds.dir() / db_filename
            self.connection = duckdb.connect(database=str(duckdb_path), read_only=True)
            self._apply_duckdb_settings(self.connection, config)

        self.cursor = self.connection.cursor()
        self.index_table = config['Database']['index_table']
        self.layout = self._detect_layout()
        from src.dataset import load_dataset_config
        self._dataset_dir = load_dataset_config(config_path).dir()
        self.vector_backend = config.get('Semantic', 'vector_backend', fallback='faiss')
        self._warm_frequency_dict()

    @staticmethod
    def _apply_duckdb_settings(connection, config: configparser.ConfigParser) -> None:
        """Cap DuckDB threads/memory: env BLEND_DUCKDB_* > [Database] duckdb_* > DuckDB defaults."""
        threads = (os.environ.get("BLEND_DUCKDB_THREADS")
                   or config['Database'].get('duckdb_threads', fallback='') or '').strip()
        if threads:
            connection.execute(f"SET threads={int(threads)}")
        mem = (os.environ.get("BLEND_DUCKDB_MEMORY_LIMIT")
               or config['Database'].get('duckdb_memory_limit', fallback='') or '').strip()
        if mem:
            connection.execute("SET memory_limit=?", [mem])

    def _warm_frequency_dict(self) -> None:
        if DBHandler.frequency_dict is not None:
            return
        print("-------- Database Configuration --------")
        print(f"Using {self.dbms.capitalize()} database, with index table {self.index_table}")
        if DBHandler.USE_ML_OPTIMIZER:
            from src.optimizer_paths import freqs_path as _freqs_path
            fpath = _freqs_path(self.dataset_dir())
            try:
                print("Loading frequency dict...", end="", flush=True)
                start = time.time()
                df = pd.read_csv(fpath)
                DBHandler.frequency_dict = dict(zip(df['tokenized'], df['frequency']))
                print(f"\rFrequency dict loaded in {time.time() - start:.2f} seconds")
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"USE_ML_OPTIMIZER=True but {fpath} is missing. "
                    f"Generate it with `python -m src.Optimizer.cli build-freqs --lake <name>`, "
                    f"or unset BLEND_USE_ML_OPTIMIZER (DBHandler.USE_ML_OPTIMIZER = False)."
                ) from e
        else:
            DBHandler.frequency_dict = {}
            print("ML optimizer disabled (USE_ML_OPTIMIZER=False); frequency dict not loaded.")
        print("----------------------------------------")

    @classmethod
    def for_dataset(cls, dataset: str, index_table: str = "blend_index", *,
                    config_path: Path | None = None) -> "DBHandler":
        """Handler for an explicit dataset (DuckDB file or Postgres schema), ignoring [Dataset].name."""
        config_path = config_path or paths.config_path()
        config = configparser.ConfigParser()
        config.read(config_path)
        dbms = config['Database']['dbms'].lower()
        self = cls.__new__(cls)
        self.connection = None
        self.cursor = None
        self.index_table = None
        self.dbms = dbms
        if dbms == 'postgres':
            from src.Semantic import pgvector_store as store
            self.connection = store.connect(dataset, autocommit=True)
        elif dbms == 'duckdb':
            import duckdb
            from src.dataset import load_dataset_config
            ds = load_dataset_config(config_path, overrides={"dataset": dataset})
            db_filename = config['Database'].get('db_filename', 'blend.duckdb')
            path = ds.dir() / db_filename
            if not path.exists():
                raise FileNotFoundError(f"no {db_filename} for dataset {dataset!r} at {path}")
            self.connection = duckdb.connect(database=str(path), read_only=True)
            self._apply_duckdb_settings(self.connection, config)
        else:
            raise ValueError(f"DBHandler.for_dataset unsupported for dbms={dbms!r}")
        self.cursor = self.connection.cursor()
        self.index_table = index_table
        self.layout = self._detect_layout()
        from src.dataset import load_dataset_config
        self._dataset_dir = load_dataset_config(config_path, overrides={"dataset": dataset}).dir()
        self.vector_backend = config.get('Semantic', 'vector_backend', fallback='faiss')
        self._warm_frequency_dict()
        return self

    def _detect_layout(self) -> str:
        if self.dbms != 'duckdb':
            return 'single'
        try:
            names = {r[0] for r in self.cursor.execute(
                "SELECT table_name FROM information_schema.tables").fetchall()}
        except Exception:
            return 'single'
        base = self.index_table
        if f"{base}_token" in names and f"{base}_tableid" in names:
            return 'two_table'
        return 'single'

    def dataset_dir(self) -> Path:
        """Resolved datasets/<name>/ dir for this handler (optimizer artifact root)."""
        return self._dataset_dir

    def optimizer_profile_str(self) -> str:
        """Profile key for runtime-dependent optimizer artifacts: dbms-backend-layout."""
        from src.optimizer_paths import optimizer_profile
        return optimizer_profile(self.dbms, self.vector_backend, self.layout)

    def close(self) -> None:
        if self.cursor is not None:
            self.cursor.close()
        if self.connection is not None:
            self.connection.close()

        self.cursor = None
        self.connection = None

    def clean_query(self, query: str) -> str:
        return query.replace('AllTables', physical_table(self.index_table, self.layout, query))

    def execute_and_fetchall(self, query: str) -> List[Union[Tuple, List]]:
        """Returns results"""
        query = self.clean_query(query)
        # TO_BITSTRING is Vertica-only; Postgres/DuckDB already store superkey as hex VARCHAR.
        if self.dbms in ('postgres', 'duckdb'):
            query = query.replace('TO_BITSTRING(superkey)', 'superkey')
        query = query.replace('CellValue', 'tokenized').replace("superkey", "super_key").replace("ColumnId", "colid")

        self.cursor.execute(query)
        results = self.cursor.fetchall()

        return results
    
    def get_table_from_index(self, table_id: int) -> pd.DataFrame:
        sql = f"""
        SELECT CellValue, ColumnId, RowId
        FROM AllTables
        WHERE TableId = {table_id}
        """

        results = self.execute_and_fetchall(sql)

        df = pd.DataFrame(results, columns=['CellValue', 'ColumnId', 'RowId'], dtype=str)
        df = df.drop_duplicates()
        df = df.pivot(index='RowId', columns='ColumnId', values='CellValue')
        df.index.name = None
        df.columns.name = None

        return df
    
    def table_ids_to_sql(self, table_ids: Iterable[int]) -> str:
        if len(table_ids) == 0:
            return "SELECT 0 AS TableId WHERE 1 = 0"

        if self.dbms == 'postgres':
            return f"""
            SELECT * FROM (
                VALUES {' ,'.join([f"({table_id})" for table_id in table_ids])}
            ) AS {DBHandler.random_subquery_name()}(TableId)
            """
        elif self.dbms == 'vertica':
            return f"""
            SELECT TableId
            FROM (
                SELECT Explode(Array[{', '.join(f"{table_id}" for table_id in table_ids)}])
                OVER (Partition Best) AS (Index_In_Array, TableId)
            ) {DBHandler.random_subquery_name()}
            """
        
        return f"""
            SELECT TableId FROM (
            {' UNION ALL '.join([f'SELECT {table_id} AS TableId' for table_id in table_ids])}
            ) AS {DBHandler.random_subquery_name()}
        """
    
    def get_token_frequencies(self, tokens: Iterable[str]) -> dict[str, int]:
        tokens = DBHandler.clean_value_collection(set(tokens))
        
        return {token: DBHandler.frequency_dict.get(token, 1) for token in tokens}

    
    @staticmethod
    def clean_value_collection(values: Iterable[any]) -> List[str]:
        # Must match ingest tokenization (src/Index/tokenize.py).
        out = []
        for v in values:
            t = tokenize_cell(v)
            if t:
                out.append(t)
        return out

    @staticmethod
    def create_sql_list_str(values: Iterable[any]) -> str:
        values = [str(x).replace('\'', '') for x in values]
        return "'{}'".format("' , '".join(set(values)))

    @staticmethod
    def create_sql_list_numeric(values: Iterable[Number]) -> str:
        values = [str(x) for x in values]
        return "{}".format(" , ".join(values))

    @staticmethod
    def random_subquery_name() -> str:
        return f"subquery{random.random()*1000000:.0f}"
