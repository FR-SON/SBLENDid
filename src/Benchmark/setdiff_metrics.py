def retention(arm: list[int], ref: list[int]) -> float:
    rs = set(ref)
    if not rs:
        return 1.0
    return len(set(arm) & rs) / len(rs)


def additions(arm: list[int], ref: list[int]) -> int:
    return len(set(arm) - set(ref))


def changed(arm: list[int], ref: list[int]) -> bool:
    return set(arm) ^ set(ref) != set()


def rbo(arm: list[int], ref: list[int], p: float = 0.9) -> float:
    if arm == ref:
        return 1.0
    sa: set[int] = set()
    sr: set[int] = set()
    depth = max(len(arm), len(ref))
    score = 0.0
    for d in range(depth):
        if d < len(arm):
            sa.add(arm[d])
        if d < len(ref):
            sr.add(ref[d])
        overlap = len(sa & sr)
        score += (overlap / (d + 1)) * (p ** d)
    return score * (1 - p) / (1 - p ** depth)
