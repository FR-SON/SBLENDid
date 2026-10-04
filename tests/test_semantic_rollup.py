from src.Semantic.rollup import vote_solve, rollup_columns_to_tables


def test_vote_solve_single_col_strict_decay():
    M = {}
    vote_solve(M, ["t1", "t2", "t3"], [0.1, 0.2, 0.3])
    assert M == {"t1": 3, "t2": 2, "t3": 1}


def test_vote_solve_tie_blocks_decay():
    M = {}
    vote_solve(M, ["t1", "t2", "t3"], [0.1, 0.1, 0.2])
    assert M == {"t1": 3, "t2": 3, "t3": 2}


def test_rollup_combines_columns():
    per_col = [
        (["t1", "t2"], [0.1, 0.3]),
        (["t2", "t3"], [0.1, 0.3]),
    ]
    out = rollup_columns_to_tables(per_col, k=2)
    assert dict(out) == {"t2": 3, "t1": 2, "t3": 1}
    assert out[0][0] == "t2"


def test_rollup_empty_inputs():
    assert rollup_columns_to_tables([], k=10) == []
    assert rollup_columns_to_tables([([], [])], k=10) == []
