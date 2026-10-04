import pandas as pd
from src.Benchmark.impute_resolve import _norm, align_columns, build_mapping


def test_norm():
    assert _norm("  Paris ") == "paris"


def test_norm_is_tokenize_cell():
    assert _norm("  Paris ") == "paris"
    assert _norm('O"Hara') == "ohara"
    assert _norm("NaN") == ""


def test_align_columns_skips_renorm_when_cand_normalized():
    examples = pd.DataFrame({"city": ["Paris", "Rome"], "country": ["France", "Italy"]})
    cand = pd.DataFrame({0: ["x", "y", "z"], 1: ["paris", "rome", "bonn"],
                         2: ["france", "italy", "germany"]})
    assert align_columns(cand, examples, cand_normalized=True) == (1, 2)


def test_build_mapping_normalized_passthrough():
    cand = pd.DataFrame({0: ["paris", "paris", "rome"], 1: ["france", "francia", "italy"]})
    assert build_mapping(cand, 0, 1, normalized=True) == {"paris": "france", "rome": "italy"}


def test_align_columns_uses_provided_col_sets():
    examples = pd.DataFrame({"city": ["Paris", "Rome"], "country": ["France", "Italy"]})
    cand = pd.DataFrame({0: ["x", "y", "z"], 1: ["paris", "rome", "bonn"],
                         2: ["france", "italy", "germany"]})
    cs = {0: {"x", "y", "z"}, 1: {"paris", "rome", "bonn"}, 2: {"france", "italy", "germany"}}
    assert align_columns(cand, examples, cand_normalized=True, col_sets=cs) == (1, 2)
    assert align_columns(cand, examples, cand_normalized=True) == (1, 2)


def test_predict_col_sets_by_key_matches_plain():
    examples = pd.DataFrame({"city": ["Paris"], "country": ["France"]})
    c1 = pd.DataFrame({0: ["paris", "rome"], 1: ["france", "italy"]})
    cs = {0: {"paris", "rome"}, 1: {"france", "italy"}}
    plain = predict([("c1", c1)], examples, {"paris", "rome"}, cand_normalized=True)
    cached = predict([("c1", c1)], examples, {"paris", "rome"}, cand_normalized=True,
                     col_sets_by_key={"c1": cs})
    assert plain == cached == {"paris": "france", "rome": "italy"}


def test_align_columns_finds_key_value_pair():
    examples = pd.DataFrame({"city": ["Paris", "Rome"], "country": ["France", "Italy"]})
    cand = pd.DataFrame({
        "id":      ["7", "8", "9"],
        "the_city": ["paris", "rome", "bonn"],
        "nation":   ["france", "italy", "germany"],
    })
    assert align_columns(cand, examples) == ("the_city", "nation")


def test_align_columns_none_when_no_example_matches():
    examples = pd.DataFrame({"k": ["zzz"], "v": ["qqq"]})
    cand = pd.DataFrame({"a": ["x"], "b": ["y"]})
    assert align_columns(cand, examples) is None


def test_build_mapping_first_occurrence():
    cand = pd.DataFrame({"k": ["A", "A", "B"], "v": ["1", "2", "3"]})
    assert build_mapping(cand, "k", "v") == {"a": "1", "b": "3"}


from src.Benchmark.impute_resolve import predict, score


def test_predict_majority_vote_across_candidates():
    examples = pd.DataFrame({"city": ["Paris"], "country": ["France"]})
    c1 = pd.DataFrame({"city": ["paris", "rome"], "country": ["france", "italy"]})
    c2 = pd.DataFrame({"city": ["paris"], "country": ["francia"]})
    c3 = pd.DataFrame({"city": ["paris"], "country": ["france"]})
    preds = predict([("c1", c1), ("c2", c2), ("c3", c3)], examples, {"paris", "rome"})
    assert preds["paris"] == "france"
    assert preds["rome"] == "italy"


def test_predict_omits_uncovered_key():
    examples = pd.DataFrame({"city": ["Paris"], "country": ["France"]})
    c1 = pd.DataFrame({"city": ["paris"], "country": ["france"]})
    preds = predict([("c1", c1)], examples, {"paris", "atlantis"})
    assert "paris" in preds and "atlantis" not in preds


def test_score_counts():
    truth = {"paris": "france", "rome": "italy", "berlin": "germany"}
    preds = {"paris": "france", "rome": "spain"}
    assert score(preds, truth) == {"n_keys": 3, "n_covered": 2, "n_correct": 1}
