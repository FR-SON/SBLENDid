from src.Operators.OperatorBase import Operator


def reference(legs: list[Operator], combine: str) -> list[int]:
    runs = [leg.run("") for leg in legs]
    first = runs[0]
    if combine == "intersection":
        keep = set.intersection(*[set(r) for r in runs])
        return [i for i in first if i in keep]
    if combine == "difference":
        excluded: set[int] = set()
        for r in runs[1:]:
            excluded |= set(r)
        return [i for i in first if i not in excluded]
    raise ValueError(combine)


def exact_semantic_ids(seeker, all_int_ids: list[int]) -> list[int]:
    # full-lake filter + exact_threshold=inf (set at construction) forces the exact path
    add = f" AND TableId IN ({','.join(str(i) for i in all_int_ids)}) "
    return seeker.run(add)
