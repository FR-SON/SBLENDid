"""DeepJoinAdapter: HF SentenceTransformer ckpt over serialized column sentences."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Iterable, Iterator

import nltk
import numpy as np
import pandas as pd

from ..registry import ColumnRef

_SENTENCE_TOKEN_CAP = 512


def serialize_column_from_cells(col_name: str, cells: list[str]) -> str:
    """Replay of JUDIT DeepJoinAdapter._serialize_column_from_cells."""
    values = pd.Series([str(c) for c in cells])
    value_counts = values.value_counts()
    sorted_vals = value_counts.index.tolist()
    n = len(sorted_vals)
    col_str = ", ".join(sorted_vals)
    lengths = [len(v) for v in sorted_vals]
    max_l = max(lengths) if lengths else 0
    min_l = min(lengths) if lengths else 0
    avg_l = (sum(lengths) / len(lengths)) if lengths else 0
    sentence = (
        f"{col_name} contains {n} values "
        f"({max_l}, {min_l}, {avg_l}): {col_str}"
    )
    try:
        tokens = nltk.word_tokenize(sentence)
    except LookupError as e:
        raise RuntimeError(
            "nltk tokenizer data missing; set BLEND_DEEPJOIN_NLTK_PATH to a "
            "dir containing tokenizers/punkt (and punkt_tab)"
        ) from e
    return " ".join(tokens[:_SENTENCE_TOKEN_CAP])


def read_lake_cells(csv_path: Path, col_idx: int) -> list[str]:
    # verbatim DeepJoin/JUDIT: pandas defaults, so NaN becomes "nan" (unlike snoopy).
    df = pd.read_csv(csv_path)
    return df.iloc[:, col_idx].astype(str).tolist()


class DeepJoinAdapter:
    name = "deepjoin"
    dim = 768
    natively_normalized = True  # ckpt modules.json ends in Normalize()
    na_cell = "nan"

    def __init__(
        self,
        *,
        lake_root: Path,
        sentences_path: Path | None,
        nltk_data_dir: Path | None = None,
        device: str = "cpu",
        encoder_version: str = "deepjoin@unknown",
        batch_size: int = 64,
    ):
        self._lake_root = Path(lake_root)
        self._sentences_path = sentences_path
        self._device = device
        self.encoder_version = encoder_version
        self._batch_size = batch_size
        self._model = None
        self._cache: dict[tuple[str, int], str] | None = None
        self.n_fallback_total = 0
        if nltk_data_dir is not None:
            nltk.data.path.insert(0, str(nltk_data_dir))

    def load(self, ckpt_path: Path) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        device = self._device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = SentenceTransformer(str(ckpt_path), device=device)
        got = self._model.get_sentence_embedding_dimension()
        if got != self.dim:
            raise ValueError(
                f"ckpt {ckpt_path} embeds dim {got}, expected {self.dim}"
            )
        if self._sentences_path is not None:
            self._cache = pickle.loads(Path(self._sentences_path).read_bytes())

    def _sentence_for(self, ref: ColumnRef) -> str:
        if self._cache is not None:
            hit = self._cache.get((ref.table_id, ref.col_idx))
            if hit is not None:
                return hit
        self.n_fallback_total += 1
        if self.n_fallback_total == 1 and self._cache is not None:
            print(
                "[DEEPJOIN][SENTENCE-FALLBACK] sentence cache miss; "
                "serializing from lake CSVs (count reported at end)",
                file=sys.stderr, flush=True,
            )
        csv_path = self._lake_root / ref.source_path
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"deepjoin fallback: lake csv {csv_path} not found for "
                f"(table={ref.table_id!r}, col_idx={ref.col_idx})"
            )
        cells = read_lake_cells(csv_path, ref.col_idx)
        return serialize_column_from_cells(ref.col_name or "", cells)

    def encode_one(self, ref: ColumnRef, *, cells: list[str] | None = None) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("load() must be called before encode_one()")
        if cells is not None:
            return self.encode_batch([ref], cells_per_ref={ref.global_id: cells})[ref.global_id]
        vec = self._model.encode(
            self._sentence_for(ref), convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return np.asarray(vec, dtype=np.float32).reshape(self.dim)

    def encode_batch(
        self,
        refs: list[ColumnRef],
        *,
        cells_per_ref: dict[int, list[str]] | None = None,
    ) -> dict[int, np.ndarray]:
        """Encode many columns in one SentenceTransformer call; sentences come from cache/lake unless cells are supplied."""
        if self._model is None:
            raise RuntimeError("load() must be called before encode_batch()")
        if not refs:
            return {}
        gids = [r.global_id for r in refs]
        if len(set(gids)) != len(gids):
            from collections import Counter
            dupes = sorted(g for g, c in Counter(gids).items() if c > 1)
            raise ValueError(
                f"encode_batch requires unique global_ids in refs; duplicates: {dupes}"
            )
        if cells_per_ref is not None:
            supplied, required = set(cells_per_ref.keys()), set(gids)
            if supplied != required:
                raise ValueError(
                    f"cells_per_ref keys must match refs' global_ids exactly; "
                    f"missing={sorted(required - supplied)}, extra={sorted(supplied - required)}"
                )
            sentences = [
                serialize_column_from_cells(
                    r.col_name or "", [str(c) for c in cells_per_ref[r.global_id]])
                for r in refs
            ]
        else:
            sentences = [self._sentence_for(r) for r in refs]
        vecs = self._model.encode(
            sentences, batch_size=self._batch_size,
            convert_to_numpy=True, normalize_embeddings=False,
            show_progress_bar=False,
        )
        return {
            r.global_id: np.asarray(v, dtype=np.float32).reshape(self.dim)
            for r, v in zip(refs, np.asarray(vecs))
        }

    def encode(self, columns: Iterable[ColumnRef]) -> Iterator[tuple[int, np.ndarray]]:
        if self._model is None:
            raise RuntimeError("load() must be called before encode()")
        refs = list(columns)
        self.n_fallback_total = 0
        for i in range(0, len(refs), self._batch_size):
            chunk = refs[i:i + self._batch_size]
            sentences = [self._sentence_for(r) for r in chunk]
            vecs = self._model.encode(
                sentences, batch_size=self._batch_size,
                convert_to_numpy=True, normalize_embeddings=False,
                show_progress_bar=False,
            )
            for ref, vec in zip(chunk, np.asarray(vecs)):
                yield ref.global_id, np.asarray(vec, dtype=np.float32)
        if self.n_fallback_total:
            print(
                f"[DEEPJOIN] {self.n_fallback_total}/{len(refs)} sentences "
                "serialized on the fly (cache miss or no cache)",
                file=sys.stderr, flush=True,
            )
