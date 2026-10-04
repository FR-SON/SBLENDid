import random

from src.Benchmark.pushdown.bench import build_allowed


def _rng(seed=0):
    return random.Random(seed)


def _setup(n=100, k=10):
    exact_rank = list(range(n))
    U = list(range(n))
    return exact_rank, U, k


def test_achievable_size_equals_c():
    exact_rank, U, k = _setup()
    for c in (0, 5, 10):
        A, achievable = build_allowed(exact_rank, U, k=k, c=c, target_size=40, rng=_rng())
        assert len(achievable) == c


def test_achievable_subset_of_A():
    exact_rank, U, k = _setup()
    A, achievable = build_allowed(exact_rank, U, k=k, c=5, target_size=40, rng=_rng())
    assert achievable <= A


def test_A_size_is_max_c_target_when_pool_large():
    exact_rank, U, k = _setup(n=500, k=10)
    A, achievable = build_allowed(exact_rank, U, k=k, c=5, target_size=40, rng=_rng())
    assert len(A) == max(5, 40)
    A2, _ = build_allowed(exact_rank, U, k=k, c=10, target_size=3, rng=_rng())
    assert len(A2) == 10


def test_decoys_excluded_from_top_2k():
    exact_rank, U, k = _setup(n=500, k=10)
    top2k = set(exact_rank[: 2 * k])
    A, achievable = build_allowed(exact_rank, U, k=k, c=5, target_size=60, rng=_rng())
    decoys = A - achievable
    assert decoys.isdisjoint(top2k)


def test_seed_determinism():
    exact_rank, U, k = _setup()
    a1, ach1 = build_allowed(exact_rank, U, k=k, c=5, target_size=40, rng=_rng(7))
    a2, ach2 = build_allowed(exact_rank, U, k=k, c=5, target_size=40, rng=_rng(7))
    assert a1 == a2 and ach1 == ach2


def test_c_zero_achievable_empty():
    exact_rank, U, k = _setup()
    A, achievable = build_allowed(exact_rank, U, k=k, c=0, target_size=40, rng=_rng())
    assert achievable == set()
    assert len(A) == 40


def test_c_equals_k_all_topk_in_A():
    exact_rank, U, k = _setup()
    topk = set(exact_rank[:k])
    A, achievable = build_allowed(exact_rank, U, k=k, c=k, target_size=40, rng=_rng())
    assert topk <= A
    assert achievable == topk


def test_no_band_is_the_pre_band_behaviour():
    exact_rank, U, k = _setup(n=500, k=10)
    a1, ach1 = build_allowed(exact_rank, U, k=k, c=5, target_size=60, rng=_rng(3))
    a2, ach2 = build_allowed(exact_rank, U, k=k, c=5, target_size=60, rng=_rng(3),
                             decoy_band=None)
    assert a1 == a2 and ach1 == ach2


def test_band_confines_decoys_to_the_rank_window():
    exact_rank, U, k = _setup(n=500, k=10)
    A, achievable = build_allowed(exact_rank, U, k=k, c=5, target_size=60, rng=_rng(),
                                  decoy_band=(100, 200))
    decoys = A - achievable
    assert decoys <= set(exact_rank[100:200])


def test_band_overlapping_the_top_2k_still_excludes_it():
    exact_rank, U, k = _setup(n=500, k=10)
    A, achievable = build_allowed(exact_rank, U, k=k, c=5, target_size=40, rng=_rng(),
                                  decoy_band=(0, 100))
    assert (A - achievable).isdisjoint(set(exact_rank[: 2 * k]))


def test_narrow_band_underfills_rather_than_widening():
    exact_rank, U, k = _setup(n=500, k=10)
    A, _ = build_allowed(exact_rank, U, k=k, c=5, target_size=200, rng=_rng(),
                         decoy_band=(100, 130))
    assert len(A) == 5 + 30


def test_warmup_runs_one_untimed_call_and_no_warmup_skips_it():
    from src.Benchmark.pushdown.bench import _timed_arm

    calls: list[str] = []

    class _Stub:
        def run(self, additionals):
            calls.append(additionals)
            return [1, 2]

    res, ms = _timed_arm(lambda: _Stub(), "ADD", repeats=3, warmup=True)
    assert len(calls) == 4 and res == [1, 2] and ms >= 0.0

    calls.clear()
    _timed_arm(lambda: _Stub(), "ADD", repeats=3, warmup=False)
    assert len(calls) == 3
