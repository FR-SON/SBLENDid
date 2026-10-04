import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")


def make_backend(method, cfg):
    """method -> SeekerBackend: `sc`, `sho`, or a semantic `approach[:index_name]`."""
    from .token import TokenBackend
    from .sho import ShoBackend
    from .semantic import SemanticBackend
    if method == "sc":
        return TokenBackend(cfg)
    if method == "sho":
        return ShoBackend(cfg)
    approach, _, idx = method.partition(":")
    return SemanticBackend(cfg, approach, idx or "default")
