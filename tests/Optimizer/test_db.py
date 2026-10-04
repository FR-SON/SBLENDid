from src.Optimizer import db as odb


def test_lake_paths_use_overrides(tmp_path):
    root = tmp_path / "datasets"
    (root / "mylake" / "csvs").mkdir(parents=True)
    assert odb.lake_dir("mylake", root=root) == root / "mylake"
    assert odb.lake_duckdb_path("mylake", root=root) == root / "mylake" / "blend.duckdb"
    assert odb.lake_csv_dir("mylake", root=root) == root / "mylake" / "csvs"
    assert odb.lake_rowid_slice_path("mylake", root=root) == root / "mylake" / odb.ROWID_SLICE_FILENAME


def test_load_freqs_sets_class_dict(tmp_path):
    p = tmp_path / "freqs_dict.csv"
    p.write_text("tokenized,frequency\nfoo,7\nbar,3\n")
    from src.DBHandler import DBHandler
    odb.load_freqs(p)
    assert DBHandler.frequency_dict["foo"] == 7
    assert DBHandler.frequency_dict["bar"] == 3


def test_run_dir_is_profile_keyed_and_samples_are_not(tmp_path):
    from src.optimizer_paths import optimizer_lake_run_dir, optimizer_run_dir
    assert optimizer_run_dir(tmp_path, "lake", "duckdb-faiss-single") == \
        tmp_path / "optimizer" / "lake" / "duckdb-faiss-single"
    assert optimizer_lake_run_dir(tmp_path, "lake") == tmp_path / "optimizer" / "lake"


def test_reader_prefers_the_profile_dir_but_falls_back(tmp_path):
    from src.optimizer_paths import resolve_optimizer_run_dir as resolve
    lake_dir = tmp_path / "optimizer" / "lake"
    (lake_dir / "semantic").mkdir(parents=True)
    assert resolve(tmp_path, "lake", "duckdb-faiss-single", subdir="semantic") == lake_dir
    (lake_dir / "duckdb-faiss-single" / "semantic").mkdir(parents=True)
    assert resolve(tmp_path, "lake", "duckdb-faiss-single", subdir="semantic") == \
        lake_dir / "duckdb-faiss-single"
    assert resolve(tmp_path, "lake", "postgres-pgvector-single", subdir="semantic") == lake_dir


def test_reader_names_the_profile_path_when_nothing_exists(tmp_path):
    from src.optimizer_paths import resolve_optimizer_run_dir as resolve
    got = resolve(tmp_path, "lake", "duckdb-faiss-single", subdir="measure")
    assert got.name == "duckdb-faiss-single"
