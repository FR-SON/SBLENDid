import pandas as pd

from src.Optimizer import sampling as smp


def _lake(tmp_path):
    for i in range(4):
        pd.DataFrame({"cat": list("abcab"), "num": [str(x) for x in range(5)],
                      "id": [f"k{i}{x}" for x in range(5)]}).to_csv(
            tmp_path / f"t{i}.csv", index=False)
    return tmp_path


def test_sample_many_matches_per_type_sampling(tmp_path):
    d = _lake(tmp_path)
    types, nbt = list(smp.SEEKER_TYPES), {"C": 3, "MC": 2}
    old = {t: smp.sample_queries(d, t, n=nbt.get(t, 5), seed=0) for t in types}
    new = smp.sample_many(d, types, n_by_type=nbt, n_default=5, seed=0)
    assert list(new) == types
    assert new == old


def test_aliased_types_scan_the_lake_once(tmp_path, monkeypatch):
    d = _lake(tmp_path)
    calls = []
    real = smp.candidates
    monkeypatch.setattr(smp, "candidates",
                        lambda cd, t, *a, **k: (calls.append(t), real(cd, t, *a, **k))[1])
    smp.sample_many(d, ["SC", "Keyword"], n_by_type={}, n_default=5, seed=0)
    assert calls == ["SC"]


def test_alias_relabels_without_changing_the_draw(tmp_path):
    d = _lake(tmp_path)
    out = smp.sample_many(d, ["SC", "Keyword"], n_by_type={}, n_default=5, seed=0)
    assert {s.seeker_type for s in out["Keyword"]} == {"Keyword"}
    strip = lambda specs: [(s.csv, s.cols, s.cardinality) for s in specs]
    assert strip(out["SC"]) == strip(out["Keyword"])
