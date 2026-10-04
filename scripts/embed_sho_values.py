"""Stage-2 SHO embedder: fill the value-keyed embedding store from values.parquet (resumable)."""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

from src.Semantic.emb_store import PersistentEmbeddingStore  # noqa: E402
from src.Semantic.simhash import SHO_ENCODER_MODEL  # noqa: E402


def embed_values(values_path: Path, store_dir: Path, *, model: str,
                 device: str, batch: int, gpu_batch: int = 512,
                 encode_fn=None) -> dict:
    print(f"reading {values_path} ...", flush=True)
    values = pd.read_parquet(values_path)["value"].tolist()
    store = PersistentEmbeddingStore(store_dir, model=model)
    print(f"{len(values)} values; {store.count} already in store — scanning for new ...",
          flush=True)
    todo = [v for v in values if v not in store]
    print(f"{len(todo)} values to encode", flush=True)
    if todo and encode_fn is None:
        print(f"loading encoder {model} on {device} "
              f"(first run downloads the model, ~300MB) ...", flush=True)
        from src.Semantic.encoders.sbert import build_encode_fn
        encode_fn = build_encode_fn(model, device=device, batch_size=gpu_batch)
        print("encoder ready — encoding", flush=True)
    writes: queue.Queue = queue.Queue(maxsize=4)
    store_s = [0.0]

    def _writer():
        while True:
            item = writes.get()
            if item is None:
                writes.task_done()
                return
            t = time.perf_counter()
            store.append(item)
            store_s[0] += time.perf_counter() - t
            writes.task_done()

    wt = threading.Thread(target=_writer, daemon=True)
    wt.start()
    n_encoded = 0
    enc_s = 0.0
    bar = tqdm(range(0, len(todo), batch), unit="batch", desc="sho embed")
    for i in bar:
        chunk = todo[i:i + batch]
        t0 = time.perf_counter()
        vecs = encode_fn(chunk)
        enc_s += time.perf_counter() - t0
        writes.put(dict(zip(chunk, vecs)))
        n_encoded += len(chunk)
        bar.set_postfix(enc=f"{enc_s:.0f}s", store=f"{store_s[0]:.0f}s", q=writes.qsize())
    writes.put(None)
    wt.join()
    return {"n_total": len(values), "n_encoded": n_encoded,
            "n_in_store": store.count, "store_bytes": store.nbytes()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--values", type=Path, required=True)
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--model", default=SHO_ENCODER_MODEL)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=16384,
                    help="values per store flush; larger = far fewer network writes")
    ap.add_argument("--gpu-batch", type=int, default=512,
                    help="encoder sub-batch on the GPU (kept small: low padding waste)")
    args = ap.parse_args()
    stats = embed_values(args.values, args.store, model=args.model,
                         device=args.device, batch=args.batch, gpu_batch=args.gpu_batch)
    print(stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
