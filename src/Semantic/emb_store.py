"""Value-keyed embedding store and resolver for the SHO seeker (SemDisc port)."""
from __future__ import annotations

import json
import os

import numpy as np


class PersistentEmbeddingStore:
    """Append-only on-disk value->embedding store (`emb.f32`, `keys.jsonl`, `meta.json`)."""
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.emb_path = str(root / "emb.f32")
        self.keys_path = str(root / "keys.jsonl")
        self.meta_path = str(root / "meta.json")
        self.dim = None
        self.count = 0
        self._index = {}
        self._keys = []
        self._mm = None
        self._load()

    def _load(self):
        if not (os.path.exists(self.meta_path) and os.path.exists(self.emb_path)
                and os.path.exists(self.keys_path)):
            return
        with open(self.meta_path) as f:
            meta = json.load(f)
        stored_model = meta.get("model")
        if stored_model and stored_model != self.model:
            raise ValueError(f"emb store model mismatch: store={stored_model} "
                             f"run={self.model} ({self.root})")
        self.dim = int(meta["dim"])
        n = int(meta["count"])
        keys = []
        with open(self.keys_path) as f:
            for line in f:
                line = line.rstrip("\n")
                if line:
                    keys.append(json.loads(line))
        n = min(n, len(keys))
        exp = n * self.dim * 4
        if os.path.getsize(self.emb_path) > exp:
            with open(self.emb_path, "r+b") as f:
                f.truncate(exp)
        if len(keys) > n:
            with open(self.keys_path, "w") as f:
                for k in keys[:n]:
                    f.write(json.dumps(k) + "\n")
        self.count = n
        self._keys = keys[:n]
        self._index = {k: i for i, k in enumerate(self._keys)}
        if n:
            self._mm = np.memmap(self.emb_path, dtype=np.float32, mode="r", shape=(n, self.dim))

    def __contains__(self, v):
        return v in self._index

    def __len__(self):
        return self.count

    def get(self, v):
        i = self._index.get(v)
        if i is None:
            return None
        return np.array(self._mm[i], dtype=np.float32)

    def append(self, items):
        """Append a value->vector dict, skipping values already present; returns rows added."""
        pairs = [(k, v) for k, v in items.items() if k not in self._index]
        if not pairs:
            return 0
        mat = np.ascontiguousarray([p[1] for p in pairs], dtype=np.float32)
        if mat.ndim != 2:
            raise ValueError("embeddings must be 2-D")
        if self.dim is None:
            self.dim = int(mat.shape[1])
        elif mat.shape[1] != self.dim:
            raise ValueError(f"dim mismatch {mat.shape[1]} vs {self.dim}")
        os.makedirs(self.root, exist_ok=True)
        with open(self.emb_path, "ab") as f:
            f.write(mat.tobytes())
        with open(self.keys_path, "a") as f:
            for k, _ in pairs:
                f.write(json.dumps(k) + "\n")
        base = self.count
        for j, (k, _) in enumerate(pairs):
            self._index[k] = base + j
            self._keys.append(k)
        self.count += len(pairs)
        with open(self.meta_path, "w") as f:                  # meta written last = commit point
            json.dump({"dim": self.dim, "count": self.count, "model": self.model}, f)
        self._mm = np.memmap(self.emb_path, dtype=np.float32, mode="r",
                             shape=(self.count, self.dim))
        return len(pairs)

    def nbytes(self):
        return sum(os.path.getsize(p) for p in (self.emb_path, self.keys_path, self.meta_path)
                   if os.path.exists(p))


class MissingEmbeddings(KeyError):
    """Strict lookup found values absent from the embedding store."""
    def __init__(self, values: list[str]):
        super().__init__(f"{len(values)} values missing from embedding store")
        self.values = values


def resolve_embeddings(values, store, encode_fn=None, batch=256, persist=True):
    """Store-first embedding lookup; misses are encoded via encode_fn or raise MissingEmbeddings."""
    from tqdm import tqdm

    overlay: dict[str, np.ndarray] = {}
    missing = [v for v in dict.fromkeys(values) if v not in store]
    if missing:
        if encode_fn is None:
            raise MissingEmbeddings(missing)
        batches = range(0, len(missing), batch)
        if len(missing) > batch:
            batches = tqdm(batches, unit="batch", desc="sho encode")
        for i in batches:
            chunk = missing[i:i + batch]
            for v, e in zip(chunk, encode_fn(chunk)):
                overlay[v] = e
        if persist:
            store.append(overlay)
    def _get(v):
        got = store.get(v)
        return got if got is not None else overlay[v]
    return np.stack([_get(v) for v in values]).astype(np.float32)
