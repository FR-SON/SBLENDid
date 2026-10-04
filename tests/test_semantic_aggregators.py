from src.Semantic.aggregators.decay_vote import DecayVoteAggregator


def test_decay_vote_aggregator_basic():
    agg = DecayVoteAggregator()
    per_col = [
        (["t1", "t2"], [0.1, 0.3]),
        (["t2", "t3"], [0.1, 0.3]),
    ]
    out = agg.aggregate(per_col, k=2)
    assert out[0] == ("t2", 3.0)
    assert {tid for tid, _ in out} == {"t1", "t2", "t3"}
    assert all(isinstance(s, float) for _, s in out)


def test_decay_vote_aggregator_empty():
    agg = DecayVoteAggregator()
    assert agg.aggregate([], k=10) == []
