"""Build a Semantic index over a CSV directory using a configured approach."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from time import perf_counter

# torch + faiss-cpu libomp conflict on macOS; must precede any torch/faiss import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

from src.Semantic.config import SemanticConfig
from src.Semantic.index_build import build_semantic_index


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv-dir", required=True, type=Path)
    ap.add_argument("--dataset", default=None,
                    help="override [Dataset].name from config")
    ap.add_argument("--dataset-root", default=None, type=Path,
                    help="override [Dataset].root from config")
    ap.add_argument("--approach", required=True)
    ap.add_argument("--index-name", required=True)
    ap.add_argument("--config", default=None, type=Path,
                    help="path to a Blend config.ini (default: config/config.ini)")
    ap.add_argument("--quant", choices=["flat", "pq"], default=None)
    ap.add_argument("--device", default=None,
                    help="cpu | cuda | auto")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    overrides = {}
    if args.dataset is not None:
        overrides["dataset"] = args.dataset
    if args.dataset_root is not None:
        overrides["dataset_root"] = str(args.dataset_root)
    if args.quant is not None:
        overrides["faiss_quant"] = args.quant
    if args.device is not None:
        overrides["device"] = args.device
    cfg = SemanticConfig.load(path=args.config, overrides=overrides or None)
    t0 = perf_counter()
    index_dir = build_semantic_index(args.csv_dir, args.approach, args.index_name, cfg)
    secs = perf_counter() - t0
    print(f"built {args.approach}/{args.index_name} index at {index_dir} in {secs:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
