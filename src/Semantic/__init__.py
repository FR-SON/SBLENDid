import os

# Must precede any faiss/torch import: duplicate libomp segfaults FAISS HNSW on macOS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from .approaches import DEEPJOIN_PLUGIN, LIFTUS_PLUGIN, SNOOPY_PLUGIN, register  # noqa: E402

register(LIFTUS_PLUGIN)
register(SNOOPY_PLUGIN)
register(DEEPJOIN_PLUGIN)
