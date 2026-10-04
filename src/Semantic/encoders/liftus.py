"""LiftusAdapter: wraps LIFTus's Net model for SegmentEncoder protocol."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn

from liftus.model import Net

from ..registry import ColumnRef


class LiftusAdapter:
    name = "liftus"
    natively_normalized = False
    na_cell = ""

    def __init__(
        self,
        left_aspects: list[str],
        right_aspects: list[str],
        hidden_size: int,
        num_heads: int,
        aspects_dir: Path,
        dataset: str,
        device: str = "cpu",
        encoder_version: str = "liftus@unknown",
    ):
        self._left_names = left_aspects
        self._right_names = right_aspects
        self._hidden = hidden_size
        self._num_heads = num_heads
        self.dim = hidden_size
        self.encoder_version = encoder_version
        self._device = torch.device(device)
        self._aspects_dir = aspects_dir
        self._dataset = dataset
        self._cache: dict[str, dict] = {}
        self._model: Net | None = None
        self._missing_per_aspect: dict[str, int] = {}

    # verbatim LIFTus run.py::emb_path templates; a wildcard would also match the <ds>_split_2_* sibling.
    _ASPECT_FILENAME_TEMPLATES: dict[str, str] = {
        "statistic": "{ds}_statistic.pickle",
        "paragraph": "{ds}_sample_32_paragraph__bert.pickle",
        "word":      "{ds}_word_emb_64_sample.pickle",
        "number":    "{ds}_number_emb_128_dim.pickle",
        "pattern":   "{ds}_pattern_emb.pickle",
    }

    def _load_aspect_pickle(self, aspect: str) -> dict:
        tpl = self._ASPECT_FILENAME_TEMPLATES.get(aspect)
        if tpl is None:
            raise KeyError(f"unknown aspect {aspect!r}; known: {sorted(self._ASPECT_FILENAME_TEMPLATES)}")
        path = self._aspects_dir / aspect / tpl.format(ds=self._dataset)
        if not path.is_file():
            raise FileNotFoundError(f"No aspect pickle for {aspect!r} at {path}")
        return pickle.loads(path.read_bytes())

    def _aspect(self, aspect: str) -> dict:
        if aspect not in self._cache:
            self._cache[aspect] = self._load_aspect_pickle(aspect)
        return self._cache[aspect]

    def _lookup(self, ref: ColumnRef, aspect: str) -> Any | None:
        d_table = self._aspect(aspect).get(ref.table_id)
        if d_table is None:
            key = f"{aspect}_table"
            self._missing_per_aspect[key] = (
                self._missing_per_aspect.get(key, 0) + 1
            )
            return None
        v = d_table.get(ref.col_name)
        if v is None:
            # verbatim LIFTus preprocess.py::getData_step: a missing column becomes a zero vector.
            self._missing_per_aspect[aspect] = (
                self._missing_per_aspect.get(aspect, 0) + 1
            )
            return None
        return v

    def load(self, ckpt_path: Path) -> None:
        input_size_left, input_size_right = self._infer_input_sizes()
        net = Net(
            input_size_left=input_size_left,
            input_size_right=input_size_right,
            num_heads=self._num_heads,
            hidden_size=self._hidden,
            criterion=nn.CrossEntropyLoss().to(self._device),
            device=self._device,
        ).to(self._device)
        state = torch.load(ckpt_path, map_location=self._device, weights_only=True)
        net.load_state_dict(state, strict=True)
        net.eval()
        self._model = net

    def _infer_input_sizes(self) -> tuple[list[Any], list[Any]]:
        sizes_map_left: dict[str, Any] = {"statistic": 99}
        sizes_map_right: dict[str, Any] = {
            "paragraph": [768, self._num_heads],
            "word": [300, self._num_heads],
            "number": [128, self._num_heads],
            "pattern": [128, self._num_heads],
        }
        left = [sizes_map_left[a] for a in self._left_names]
        right = [sizes_map_right[a] for a in self._right_names]
        return left, right

    def _reduce_statistic(self, v: np.ndarray) -> torch.Tensor:
        arr = np.asarray(v, dtype=np.float32)
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(self._device)

    def _reduce_paragraph(self, v: list, seq_len: int = 32) -> torch.Tensor:
        all_vecs = [np.asarray(t, dtype=np.float32) for sample in v for t in sample]
        if not all_vecs:
            mat = np.zeros((seq_len, 768), dtype=np.float32)
        elif len(all_vecs) >= seq_len:
            mat = np.stack(all_vecs[:seq_len])
        else:
            pad = np.zeros((seq_len - len(all_vecs), 768), dtype=np.float32)
            mat = np.vstack([np.stack(all_vecs), pad])
        return torch.from_numpy(mat).unsqueeze(0).to(self._device)

    def _reduce_word(self, v: list, seq_len: int = 64) -> torch.Tensor:
        vecs = [np.asarray(x[2], dtype=np.float32) for x in v]
        if not vecs:
            embed_dim = 300
            mat = np.zeros((seq_len, embed_dim), dtype=np.float32)
        else:
            embed_dim = vecs[0].shape[0]
            if len(vecs) >= seq_len:
                mat = np.stack(vecs[:seq_len])
            else:
                pad = np.zeros((seq_len - len(vecs), embed_dim), dtype=np.float32)
                mat = np.vstack([np.stack(vecs), pad])
        return torch.from_numpy(mat).unsqueeze(0).to(self._device)

    def _reduce_number(self, v: list, seq_len: int = 14) -> torch.Tensor:
        vecs = [np.asarray(x[1], dtype=np.float32) for x in v]
        if not vecs:
            mat = np.zeros((seq_len, 128), dtype=np.float32)
        elif len(vecs) >= seq_len:
            mat = np.stack(vecs[:seq_len])
        else:
            pad = np.zeros((seq_len - len(vecs), 128), dtype=np.float32)
            mat = np.vstack([np.stack(vecs), pad])
        return torch.from_numpy(mat).unsqueeze(0).to(self._device)

    def _reduce_pattern(self, v) -> torch.Tensor:
        if not isinstance(v, list) or not v:
            raise ValueError(
                f"pattern aspect: expected non-empty list[tuple], got {type(v).__name__}"
            )
        vecs = [np.asarray(t[3], dtype=np.float32) for t in v]
        embed_dim = int(vecs[0].shape[0])
        mat = np.zeros((self._SEQ_LEN_PATTERN, embed_dim), dtype=np.float32)
        n = min(len(vecs), self._SEQ_LEN_PATTERN)
        for i in range(n):
            mat[i] = vecs[i]
        return torch.from_numpy(mat).unsqueeze(0).to(self._device)

    _SEQ_LEN_PATTERN: int = 128

    _ASPECT_SEQ_LEN: dict[str, int] = {
        "paragraph": 32,
        "word": 64,
        "number": 14,
        "pattern": 128,
    }

    _REDUCERS = {
        "statistic": "_reduce_statistic",
        "paragraph": "_reduce_paragraph",
        "word":      "_reduce_word",
        "number":    "_reduce_number",
        "pattern":   "_reduce_pattern",
    }

    _ZERO_SHAPES: dict[str, tuple[int, int, int]] = {
        "statistic": (1, 1, 99),
        "paragraph": (1, 32, 768),
        "word":      (1, 64, 300),
        "number":    (1, 14, 128),
        "pattern":   (1, 128, 128),
    }

    def _to_tensor(self, ref: ColumnRef, aspect: str) -> torch.Tensor:
        v = self._lookup(ref, aspect)
        if v is None:
            shape = self._ZERO_SHAPES.get(aspect)
            if shape is None:
                raise ValueError(f"No zero-shape defined for aspect {aspect!r}")
            return torch.zeros(shape, dtype=torch.float32, device=self._device)
        reducer_name = self._REDUCERS.get(aspect)
        if reducer_name is None:
            raise ValueError(f"No reduction defined for aspect {aspect!r}")
        return getattr(self, reducer_name)(v)

    def encode_one(self, ref: ColumnRef, *, cells: list[str] | None = None) -> np.ndarray:
        if cells is not None:
            raise NotImplementedError(
                "liftus: query-time encoding is not implemented. LIFTus derives a column "
                "vector from five aspects that its preprocessing computes as a batch "
                "pipeline over a whole dataset, two of them sampled without a seed, so a "
                "vector encoded later is a different draw from the stored one and cannot "
                "be verified against it; the released lakes carry the aspects for their "
                "own tables, which is what in-lake queries use. Enrol the table in the "
                "index, or see docs/notes.md 'Queries outside the index' for the pathway."
            )
        if self._model is None:
            raise RuntimeError("load() must be called before encode_one()")
        left = [self._to_tensor(ref, a) for a in self._left_names]
        right = [self._to_tensor(ref, a) for a in self._right_names]
        with torch.no_grad():
            attn, _ = self._model.inference(left, right)
        out = attn.reshape(-1).detach().cpu().numpy().astype(np.float32)
        if out.shape != (self._hidden,):
            raise RuntimeError(
                f"liftus encode_one shape {out.shape}, expected ({self._hidden},)"
            )
        return out

    def _aggregate_attn_weight(self, attn_weight) -> dict[str, float]:
        w = attn_weight.squeeze(0).squeeze(0).detach().cpu().numpy()
        total_len = int(w.shape[0])
        out: dict[str, float] = {}
        cur = 0
        for aspect in self._right_names:
            try:
                n = self._ASPECT_SEQ_LEN[aspect]
            except KeyError as e:
                raise ValueError(
                    f"_ASPECT_SEQ_LEN missing entry for right-aspect {aspect!r}"
                ) from e
            seg = w[cur:cur + n]
            seg_sum = float(seg.sum())
            uniform_share = n / total_len if total_len > 0 else 0.0
            out[f"sum_{aspect}"] = seg_sum
            out[f"mean_{aspect}"] = float(seg.mean())
            out[f"max_{aspect}"] = float(seg.max())
            out[f"ratio_{aspect}"] = (
                seg_sum / uniform_share if uniform_share > 0 else 0.0
            )
            cur += n
        if cur != total_len:
            raise RuntimeError(
                f"attn_weight key_len {total_len} != expected {cur} "
                f"(check _ASPECT_SEQ_LEN vs reducer seq_lens)"
            )
        return out

    def encode_one_with_sdai(
        self, ref: ColumnRef,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Encode one column and return its per-right-aspect SDAI attention mass."""
        if self._model is None:
            raise RuntimeError(
                "load() must be called before encode_one_with_sdai()"
            )
        left = [self._to_tensor(ref, a) for a in self._left_names]
        right = [self._to_tensor(ref, a) for a in self._right_names]
        with torch.no_grad():
            attn, attn_weight = self._model.inference(left, right)
        out = attn.reshape(-1).detach().cpu().numpy().astype(np.float32)
        if out.shape != (self._hidden,):
            raise RuntimeError(
                f"liftus encode_one_with_sdai shape {out.shape}, "
                f"expected ({self._hidden},)"
            )
        aspects = self._aggregate_attn_weight(attn_weight)
        return out, aspects

    def encode_with_sdai(
        self, columns: list[ColumnRef],
    ) -> Iterator[tuple[int, np.ndarray, dict[str, float]]]:
        """Streaming variant of encode() that also yields per-column SDAI."""
        self._missing_per_aspect.clear()
        for ref in columns:
            vec, aspects = self.encode_one_with_sdai(ref)
            yield ref.global_id, vec, aspects
        if self._missing_per_aspect:
            msg = ", ".join(
                f"{a}={n}" for a, n in sorted(self._missing_per_aspect.items())
            )
            print(
                f"[liftus-encode] missing-column zero-fallbacks: {msg}",
                file=sys.stderr, flush=True,
            )

    def encode(self, columns: list[ColumnRef]) -> Iterator[tuple[int, np.ndarray]]:
        self._missing_per_aspect.clear()
        for ref in columns:
            yield ref.global_id, self.encode_one(ref)
        if self._missing_per_aspect:
            msg = ", ".join(
                f"{a}={n}" for a, n in sorted(self._missing_per_aspect.items())
            )
            print(
                f"[liftus-encode] missing-column zero-fallbacks: {msg}",
                file=sys.stderr, flush=True,
            )
