from src.Benchmark.correctness.theorem1.audit import audit_witness

R_A = [(10, 5.0), (11, 4.0), (12, 3.0), (13, 2.0)]


def test_genuine_strict_exclusion():
    out = audit_witness({13}, {"a": R_A}, {"a": 2})
    assert out["verdict"] == "genuine"
    rec = out["records"][0]
    assert rec["best_score"] == 2.0 and rec["boundary"] == 4.0 and rec["strict"]


def test_absent_table_is_strictly_excluded():
    out = audit_witness({99}, {"a": R_A}, {"a": 2})
    assert out["verdict"] == "genuine"
    assert out["records"][0]["best_score"] is None


def test_tie_fragile_on_boundary():
    ranking = [(10, 5.0), (11, 4.0), (13, 4.0), (12, 3.0)]
    out = audit_witness({13}, {"a": ranking}, {"a": 2})
    assert out["verdict"] == "tie_fragile"


def test_not_truncation_no_leg_truncates():
    out = audit_witness({13}, {"a": R_A}, {"a": 10})
    assert out["verdict"] == "not_truncation"


def test_not_truncation_all_above_boundary():
    out = audit_witness({10}, {"a": R_A}, {"a": 3})
    assert out["verdict"] == "not_truncation"


def test_mixed_case_is_tie_fragile():
    ranking = [(10, 5.0), (11, 4.0), (13, 4.0), (12, 3.0)]
    out = audit_witness({10, 13}, {"a": ranking}, {"a": 2})
    assert out["verdict"] == "tie_fragile"


def test_best_score_is_max_over_duplicate_rows():
    ranking = [(10, 5.0), (13, 4.5), (11, 4.0), (13, 1.0)]
    out = audit_witness({13}, {"a": ranking}, {"a": 3})
    assert out["verdict"] == "not_truncation"
    assert out["records"][0]["best_score"] == 4.5


def test_no_added_tables_not_auditable():
    out = audit_witness(set(), {"a": R_A}, {"a": 2})
    assert out["verdict"] == "not_auditable" and out["reason"] == "no_added_tables"
