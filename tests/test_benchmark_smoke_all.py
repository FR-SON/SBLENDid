def test_all_modules_import():
    from src.Benchmark.adapters import union, join, imputation, correlation, negative_example  # noqa: F401
    from src.Benchmark.recipes import imputation as ri, correlation as rc, negative_example as rn  # noqa: F401
    from src.Benchmark import cli, datasource, db, metrics, runspec, prepare  # noqa: F401
    assert cli.build_parser() is not None
    from src.Benchmark import impute_resolve, impute_compare  # noqa: F401
    assert set(impute_compare.ALL_LEGS) == {
        "syntactic_join", "semantic_join", "syntactic_paper", "semantic_paper",
        "mc_only"}
    from src.Benchmark import plans
    assert "semantic_oracle_repair" in plans.PLANS
