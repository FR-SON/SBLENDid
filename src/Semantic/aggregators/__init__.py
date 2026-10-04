from .base import AggregationContext, Aggregator
from .decay_vote import DecayVoteAggregator
from .max_pool import MaxPoolAggregator


def build_op_aggregator(op_cfg, default):
    """Resolve [Semantic.<OP>] aggregator= to an instance; 'default' = plugin's."""
    choice = getattr(op_cfg, "aggregator", "default") or "default"
    if choice == "default":
        return default
    if choice == "decay_vote":
        return DecayVoteAggregator()
    if choice == "max_pool":
        return MaxPoolAggregator()
    if choice == "munkres":
        if op_cfg.munkres_threshold is None:
            raise ValueError(
                "aggregator = munkres requires munkres_threshold in the "
                "[Semantic.<OP>] section (e.g. munkres_threshold = 0.3)"
            )
        from .munkres import MunkresAggregator
        return MunkresAggregator(threshold=op_cfg.munkres_threshold)
    raise ValueError(f"unknown aggregator {choice!r}; "
                     "expected default | decay_vote | max_pool | munkres")


__all__ = [
    "AggregationContext",
    "Aggregator",
    "DecayVoteAggregator",
    "MaxPoolAggregator",
    "build_op_aggregator",
]
