import io

import pandas as pd
import pytest

from src.Benchmark.correctness.theorem1 import shapes as SH
from src.Benchmark.datasource import load_query_table


@pytest.fixture
def tiny_lake(tmp_path, monkeypatch):
    csvs = tmp_path / "lakeA" / "csvs"
    csvs.mkdir(parents=True)
    (csvs / "q1.csv").write_text("x,y\n1,a\n2,b\n3,c\n")
    monkeypatch.setenv("BLEND_DATASETS_DIR", str(tmp_path))
    return "lakeA"


def test_load_query_table_nrows_truncates(tiny_lake):
    assert len(load_query_table(tiny_lake, "q1.csv", nrows=2)) == 2


def test_load_query_table_default_unchanged(tiny_lake):
    df = load_query_table(tiny_lake, "q1.csv")
    assert len(df) == 3
    assert df.dtypes.map(str).eq("object").all()


def test_present_queries_filters_absent_before_limit(tiny_lake):
    from src.Benchmark.correctness.theorem1.run import present_queries
    present, n_absent = present_queries(tiny_lake, ["nope.csv", "q1.csv", "gone.csv"])
    assert present == ["q1.csv"]
    assert n_absent == 2


def test_registry_has_all_seven_shapes_with_provenance():
    assert set(SH.SHAPES) == {"sc_sc_flat", "sc_sc_prov", "sc_kw",
                              "mc_sc_paper", "mc_sc_repo", "su_sc", "sj_sc"}
    for s in SH.SHAPES.values():
        assert s.shape_source.strip()
        assert s.arm3_only == SH.has_mc(s)
        assert s.prefix_stable == (not SH.has_mc(s))


def test_leg_multipliers_match_design():
    m = {n: tuple(l.k_mult for l in s.legs) for n, s in SH.SHAPES.items()}
    assert m["sc_sc_flat"] == (1, 1)
    assert m["sc_sc_prov"] == (30, 30)
    assert m["sc_kw"] == (1, 1)
    assert m["mc_sc_paper"] == (1, 1)
    assert m["mc_sc_repo"] == (10, 30)


def test_sc_kw_not_optimizer_reachable():
    assert not SH.SHAPES["sc_kw"].optimizer_reachable
    assert all(SH.SHAPES[n].optimizer_reachable for n in
               ("sc_sc_flat", "sc_sc_prov", "mc_sc_paper", "mc_sc_repo", "su_sc", "sj_sc"))


def test_usable_columns_convention_independent():
    text = "a,b,c\nnan,x,1\nNONE,y,2\n'',z,3\n"
    default_na = pd.read_csv(io.StringIO(text))
    no_na = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    assert SH.select_columns(default_na) == SH.select_columns(no_na) == ("b", "c")


def test_too_few_usable_columns():
    df = pd.DataFrame({"a": ["nan", "none"], "b": ["x", "y"]})
    assert SH.select_columns(df) is None


def test_terminal_leg():
    assert SH.terminal_leg(SH.SHAPES["sc_kw"]).kind == "SC"
    assert SH.terminal_leg(SH.SHAPES["mc_sc_repo"]).kind == "MC"
    assert SH.terminal_leg(SH.SHAPES["su_sc"]).kind == "SC"
    assert SH.terminal_leg(SH.SHAPES["su_sc"], sem_position="last").kind == "SU"
    a, b = SH.SHAPES["sc_sc_flat"].legs
    assert SH.terminal_leg(SH.SHAPES["sc_sc_flat"]) is b
