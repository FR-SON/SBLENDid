from pathlib import Path
from src.Semantic.config import SemanticConfig


def test_defaults_are_faiss():
    cfg = SemanticConfig()
    assert cfg.vector_backend == "faiss"


def test_loads_pgvector_from_ini(tmp_path: Path):
    ini = tmp_path / "config.ini"
    ini.write_text(
        "[Dataset]\nname = santos\n"
        "[Semantic]\nvector_backend = pgvector\n"
    )
    cfg = SemanticConfig.load(path=ini)
    assert cfg.vector_backend == "pgvector"


def test_signature_differs_by_vector_backend():
    cfg1 = SemanticConfig(vector_backend="faiss")
    cfg2 = SemanticConfig(vector_backend="pgvector")
    assert cfg1.signature() != cfg2.signature()


def test_default_pushdown_mode_is_prefilter():
    assert SemanticConfig().pg_pushdown_mode == "prefilter"


def test_loads_pushdown_mode_from_ini(tmp_path: Path):
    ini = tmp_path / "config.ini"
    ini.write_text(
        "[Dataset]\nname = santos\n"
        "[Semantic]\npg_pushdown_mode = postfilter\n"
    )
    assert SemanticConfig.load(path=ini).pg_pushdown_mode == "postfilter"


def test_signature_differs_by_pushdown_mode():
    a = SemanticConfig(pg_pushdown_mode="prefilter")
    b = SemanticConfig(pg_pushdown_mode="postfilter")
    assert a.signature() != b.signature()


def test_pgvector_nulls_exact_threshold(tmp_path: Path):
    ini = tmp_path / "config.ini"
    ini.write_text(
        "[Dataset]\nname = santos\n"
        "[Semantic]\nvector_backend = pgvector\nexact_threshold = 1035\n"
    )
    assert SemanticConfig.load(path=ini).exact_threshold is None


def test_faiss_keeps_exact_threshold(tmp_path: Path):
    ini = tmp_path / "config.ini"
    ini.write_text(
        "[Dataset]\nname = santos\n"
        "[Semantic]\nvector_backend = faiss\nexact_threshold = 1035\n"
    )
    assert SemanticConfig.load(path=ini).exact_threshold == 1035


def test_pgvector_explicit_exact_override_survives():
    import pandas as pd
    from src.Operators.Seekers.SemanticUnion import SemanticUnion
    df = pd.DataFrame({"c": ["x"]}); df.attrs["table_id"] = "t.csv"
    forced = SemanticUnion(df, exact_threshold=500,
                           config_overrides={"vector_backend": "pgvector"})
    assert forced._exact_threshold == 500
    defaulted = SemanticUnion(df, config_overrides={"vector_backend": "pgvector"})
    assert defaulted._exact_threshold is None
