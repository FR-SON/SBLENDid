import pytest


@pytest.fixture(autouse=True)
def _duckdb_env(duckdb_unit_env):
    yield


class FakeHandle:
    manifest = {"bits": 6, "seed": 0, "sample_seed": 0, "row_cap": 3,
                "skip_numeric": True, "encoder_model": "m", "dim": 8}

    def codes_for_values(self, values):
        return [len(v) for v in values]


class FakeDB:
    dbms = "duckdb"

    @staticmethod
    def create_sql_list_numeric(values):
        return ", ".join(str(v) for v in values)


@pytest.fixture
def seeker_cls(monkeypatch):
    from src.Semantic.seekers import sho as sho_mod
    monkeypatch.setattr(sho_mod.ShoHandle, "open", classmethod(lambda cls, cfg: FakeHandle()))
    return sho_mod.SimHashOverlapSeekerBase


def test_sql_shape_and_pushdown(seeker_cls):
    s = seeker_cls(["ab", "xyz", "ab"], k=7)
    sql = s.create_sql_query(FakeDB(), additionals=" AND TableId IN (1,2) ")
    assert "FROM AllTables" in sql
    assert "simhash_code IN (2, 3)" in sql
    assert "AND TableId IN (1,2)" in sql
    assert "GROUP BY TableId, ColumnId" in sql
    assert "COUNT(DISTINCT simhash_code) DESC" in sql
    assert "LIMIT 7" in sql


def test_codes_cached_across_additionals(seeker_cls, monkeypatch):
    s = seeker_cls(["ab"], k=5)
    s.create_sql_query(FakeDB())
    calls = []
    monkeypatch.setattr(FakeHandle, "codes_for_values",
                        lambda self, v: calls.append(v) or [1])
    s.create_sql_query(FakeDB(), additionals=" AND TableId IN (9) ")
    assert calls == []


def test_numeric_column_short_circuits(seeker_cls):
    s = seeker_cls(["123", "456", "789"], k=5)
    sql = s.create_sql_query(FakeDB())
    assert "WHERE 1=0" in sql


def test_empty_input_short_circuits(seeker_cls):
    s = seeker_cls(["", "nan"], k=5)
    assert "WHERE 1=0" in s.create_sql_query(FakeDB())


def test_row_cap_sampling_applied(seeker_cls):
    vals = [f"v{i}" for i in range(10)]
    s = seeker_cls(vals, k=5)
    sql = s.create_sql_query(FakeDB())
    n_codes = sql.split("simhash_code IN (")[1].split(")")[0].count(",") + 1
    assert n_codes <= 3


def test_flags_and_cost(seeker_cls):
    s = seeker_cls(["ab"], k=5)
    assert s.is_approximate is True
    assert s.HAS_ML_COST_MODEL is False
    assert isinstance(s.cost(), int)


def test_alias_importable():
    from src.Operators import Seekers
    assert Seekers.SHO.__name__ == "SimHashOverlap"


def test_lookup_path_db_only_no_encoder(monkeypatch):
    from src.Semantic.seekers import sho as sho_mod
    handle = FakeHandle()
    handle.basename_to_int = {"t.csv": 5}
    monkeypatch.setattr(sho_mod.ShoHandle, "open",
                        classmethod(lambda cls, cfg: handle))
    calls = []
    monkeypatch.setattr(FakeHandle, "codes_for_values",
                        lambda self, v: calls.append(v) or [999])

    class LookupDB(FakeDB):
        def __init__(self):
            self.issued = []

        def execute_and_fetchall(self, sql):
            self.issued.append(sql)
            return [(3,), (2,)]

    db = LookupDB()
    s = sho_mod.SimHashOverlapSeekerBase(
        [], k=5, query_table_id="t.csv", query_col_id=0)
    sql = s.create_sql_query(db)
    assert "simhash_code IN (2, 3)" in sql
    assert calls == []
    assert any("TableId = 5" in q and "ColumnId = 0" in q for q in db.issued)


def test_lookup_accepts_int_table_id(monkeypatch):
    from src.Semantic.seekers import sho as sho_mod
    handle = FakeHandle()
    handle.basename_to_int = {}
    monkeypatch.setattr(sho_mod.ShoHandle, "open",
                        classmethod(lambda cls, cfg: handle))

    class LookupDB(FakeDB):
        def execute_and_fetchall(self, sql):
            assert "TableId = 7" in sql and "ColumnId = 2" in sql
            return [(1,)]

    s = sho_mod.SimHashOverlapSeekerBase(
        [], k=5, query_table_id=7, query_col_id=2)
    assert "simhash_code IN (1)" in s.create_sql_query(LookupDB())


def test_lookup_unknown_basename_short_circuits(monkeypatch):
    from src.Semantic.seekers import sho as sho_mod
    handle = FakeHandle()
    handle.basename_to_int = {}
    monkeypatch.setattr(sho_mod.ShoHandle, "open",
                        classmethod(lambda cls, cfg: handle))
    s = sho_mod.SimHashOverlapSeekerBase(
        ["ab"], k=5, query_table_id="missing.csv", query_col_id=0)
    assert "WHERE 1=0" in s.create_sql_query(FakeDB())


def test_embed_fallback_emits_banner(seeker_cls, capsys):
    s = seeker_cls(["ab", "xyz"], k=5)
    s.create_sql_query(FakeDB())
    assert "FALLBACK-ENCODE" in capsys.readouterr().err
