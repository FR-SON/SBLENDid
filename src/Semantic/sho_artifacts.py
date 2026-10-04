"""SHO artifact handle; hashing params come from manifest.json only, never from config."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from .config import SemanticConfig, SemanticOp
from .emb_store import PersistentEmbeddingStore, resolve_embeddings
from .simhash import codes_from_embeddings, get_hyperplanes

_REQUIRED_MANIFEST_KEYS = (
    "bits", "seed", "sample_seed", "row_cap", "skip_numeric", "encoder_model", "dim",
)


@dataclass
class ShoHandle:
    manifest: dict
    hyperplanes: np.ndarray
    store: PersistentEmbeddingStore
    device: str
    cfg: SemanticConfig
    _encode_fn: object = field(default=None, repr=False)
    _b2i: dict | None = field(default=None, repr=False)

    _CACHE: ClassVar[dict] = {}

    @property
    def basename_to_int(self) -> dict[str, int]:
        """Blend basename -> int TableId, from the basenames sidecar."""
        if self._b2i is None:
            import pandas as pd
            df = pd.read_parquet(self.cfg.blend_basenames_path)
            self._b2i = dict(zip(df["basename"].astype(str),
                                 df["table_int_id"].astype(int)))
        return self._b2i

    @classmethod
    def open(cls, cfg: SemanticConfig) -> "ShoHandle":
        key = cfg.signature()
        if key in cls._CACHE:
            return cls._CACHE[key]
        op = cfg.operator(SemanticOp.SHO)
        adir = cfg.approach_dir(op.approach, op.index_name)
        mpath = adir / "manifest.json"
        if not mpath.is_file():
            raise FileNotFoundError(
                f"SHO manifest not found: {mpath} — run scripts/create_sho_index.py "
                "for this dataset first")
        if not (adir / "codes.parquet").is_file():
            raise FileNotFoundError(
                f"SHO codes.parquet not found in {adir} — run "
                "create_sho_index.py --stage hash (and materialize) first")
        manifest = json.loads(mpath.read_text())
        missing = [k for k in _REQUIRED_MANIFEST_KEYS if k not in manifest]
        if missing:
            raise KeyError(f"SHO manifest {mpath} missing fields {missing}")
        store = PersistentEmbeddingStore(adir / "emb_store",
                                         model=manifest["encoder_model"])
        handle = cls(
            manifest=manifest,
            hyperplanes=get_hyperplanes(manifest["bits"], manifest["dim"],
                                        manifest["seed"]),
            store=store,
            device=cfg.device,
            cfg=cfg,
        )
        cls._CACHE[key] = handle
        return handle

    def _encoder(self):
        if self._encode_fn is None:
            from .encoders.sbert import build_encode_fn
            self._encode_fn = build_encode_fn(self.manifest["encoder_model"],
                                              device=self.device)
        return self._encode_fn

    def codes_for_values(self, values: list[str]) -> list[int]:
        """Codes for values: store lookup first, non-persisted local encode on miss."""
        try:
            emb = resolve_embeddings(values, self.store, encode_fn=None)
        except KeyError:
            emb = resolve_embeddings(values, self.store, encode_fn=self._encoder(),
                                     persist=False)
        return codes_from_embeddings(emb, self.hyperplanes).tolist()
