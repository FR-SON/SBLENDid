import importlib


def test_module_imports_and_exposes_main():
    mod = importlib.import_module("scripts.build_oracle_keywords")
    assert callable(mod.main)
    assert callable(mod.propose_tokens)


def test_propose_tokens_normalizes_and_ranks(tmp_path, monkeypatch):
    import pandas as pd
    monkeypatch.setenv("BLEND_DATASETS_DIR", str(tmp_path))
    csvs = tmp_path / "ds" / "csvs"
    csvs.mkdir(parents=True)
    pd.DataFrame({"c": ["Quebec", "Quebec", "MONTREAL"]}).to_csv(
        csvs / "a.csv", index=False)
    from scripts.build_oracle_keywords import propose_tokens
    toks = propose_tokens("ds", ["a.csv"], top_n=5)
    assert "quebec" in toks
    assert "montreal" in toks
