from src.Operators.Seekers.Correlation import Correlation
from src.Operators.Seekers.Keyword import Keyword
from src.Operators.Seekers.MultiColumnOverlap import MultiColumnOverlap
from src.Operators.Seekers.SemanticJoin import SemanticJoin
from src.Operators.Seekers.SemanticUnion import SemanticUnion
from src.Operators.Seekers.SingleColumnOverlap import SingleColumnOverlap

MC = MultiColumnOverlap
SC = SingleColumnOverlap
C = Correlation
SU = SemanticUnion
SJ = SemanticJoin


def __getattr__(name):
    # Lazy: an eager import is circular (src.Semantic.seekers.sho imports this package).
    if name in ("SimHashOverlap", "SHO"):
        from src.Operators.Seekers.SimHashOverlap import SimHashOverlap
        globals()["SimHashOverlap"] = SimHashOverlap
        globals()["SHO"] = SimHashOverlap
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
