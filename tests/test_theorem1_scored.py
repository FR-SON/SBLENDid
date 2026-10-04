import pytest

from src.Benchmark.correctness.theorem1.scored import scored_ranking, scored_sql_template


def _seekers():
    from src.Operators import Seekers
    return Seekers


def test_sc_template_drops_limit_promotes_score(duckdb_unit_env):
    S = _seekers()
    sql = scored_sql_template(S.SC(["a"], k=7).base_sql)
    assert "LIMIT" not in sql
    assert "COUNT(DISTINCT CellValue) AS score" in sql
    assert "AllTables" in sql
    assert "ORDER BY COUNT(DISTINCT CellValue) DESC" in " ".join(sql.split())


def test_kw_template_scoreable(duckdb_unit_env):
    S = _seekers()
    sql = scored_sql_template(S.Keyword(["a"], k=7).base_sql)
    assert "LIMIT" not in sql and "AS score" in sql


def test_non_scoreable_raises():
    with pytest.raises(ValueError):
        scored_sql_template("SELECT TableId FROM AllTables")


def test_scored_ranking_restores_base_sql_and_parses(duckdb_unit_env):
    S = _seekers()
    leg = S.SC(["a"], k=7)
    saved = leg.base_sql

    class _Db:
        def create_sql_list_str(self, vals): return ",".join(f"'{v}'" for v in vals)
        def clean_value_collection(self, vals): return list(vals)
        def execute_and_fetchall(self, sql):
            assert "LIMIT" not in sql
            return [(3, 5), (1, 2)]

    assert scored_ranking(leg, _Db()) == [(3, 5.0), (1, 2.0)]
    assert leg.base_sql == saved
