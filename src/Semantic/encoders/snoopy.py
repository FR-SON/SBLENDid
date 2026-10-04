"""ScorpionAdapter: wraps Snoopy's Scorpion model for SegmentEncoder protocol."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pandas as pd
import torch

from snoopy.model import Scorpion

from ..registry import ColumnRef


def _table_rng(dataset: str, table_basename: str, row_cap_seed: int) -> np.random.Generator:
    """Stable per-table RNG (blake2b, not salted hash()); verbatim judit.prep."""
    key = f"{dataset}\x00{table_basename}\x00{row_cap_seed}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    seed = int.from_bytes(digest, "big") % (2**32)
    return np.random.default_rng(seed)


def _sample_row_indices(
    n_rows: int, row_cap: int, rng: np.random.Generator,
) -> np.ndarray:
    """Sorted row-sample indices in [0, n_rows); verbatim judit.prep."""
    if n_rows <= row_cap:
        return np.arange(n_rows, dtype=np.int64)
    idx = rng.choice(n_rows, size=row_cap, replace=False)
    idx.sort()
    return idx


CellEmbedder = Callable[[list[str]], np.ndarray]


def _normalize_cells(cells: list[str], cap: int) -> list[str]:
    """Replay training-time cell transforms: JUDIT prep char cap, then snoopy's newline-to-space."""
    out: list[str] = []
    for c in cells:
        if len(c) > cap:
            c = c[:cap]
        if "\n" in c:
            c = c.replace("\n", " ")
        out.append(c)
    return out


def _default_fasttext_embedder(d: int, fasttext_path: Path) -> CellEmbedder:
    import fasttext  # type: ignore

    ft = fasttext.load_model(str(fasttext_path))

    def embed(cells: list[str]) -> np.ndarray:
        out = np.zeros((len(cells), d), dtype=np.float32)
        for i, c in enumerate(cells):
            out[i] = ft.get_sentence_vector(c).astype(np.float32)
        return out

    return embed


def _forward_chunked(
    model: Scorpion,
    x_all: torch.Tensor,
    index_list: list[int],
    chunk_size: int,
) -> torch.Tensor:
    """Fork of snoopy.model._forward_chunked; equals the unchunked call within fp32 tolerance."""
    n_cols = len(index_list)
    if chunk_size <= 0 or chunk_size >= n_cols:
        return model(x_all, torch.tensor(index_list))
    outs = []
    cell_offset = 0
    for col_start in range(0, n_cols, chunk_size):
        col_end = min(col_start + chunk_size, n_cols)
        chunk_idx = index_list[col_start:col_end]
        n_cells = sum(chunk_idx)
        x_chunk = x_all[cell_offset:cell_offset + n_cells]
        outs.append(model(x_chunk, torch.tensor(chunk_idx)))
        cell_offset += n_cells
    return torch.cat(outs, dim=0)


class ScorpionAdapter:
    name = "snoopy"
    natively_normalized = True
    na_cell = ""

    def __init__(
        self,
        n_proxy_sets: int,
        n_elements: int,
        d: int,
        device: str,
        lake_root: Path,
        cell_embedder: CellEmbedder,
        cell_char_cap: int,
        row_cap: int,
        row_cap_seed: int,
        dataset: str,
        eval_chunk_size: int,
        encoder_version: str = "snoopy@unknown",
    ):
        assert eval_chunk_size >= 1, (
            f"eval_chunk_size must be >= 1, got {eval_chunk_size}"
        )
        self._args = argparse.Namespace(
            n_proxy_sets=n_proxy_sets, n_elements=n_elements, d=d, device=device,
        )
        self.dim = n_proxy_sets
        self.encoder_version = encoder_version
        self._device = torch.device(device)
        self._model: Scorpion | None = None
        self._lake_root = lake_root
        self._embed = cell_embedder
        self._cell_char_cap = cell_char_cap
        self._row_cap = row_cap
        self._row_cap_seed = row_cap_seed
        self._dataset = dataset
        self._eval_chunk_size = eval_chunk_size

    def load(self, ckpt_path: Path) -> None:
        model = Scorpion(self._args).to(self._device)
        state = torch.load(ckpt_path, map_location=self._device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        self._model = model

    def _apply_row_cap(self, ref: ColumnRef, cells: list[str]) -> list[str]:
        n_rows = len(cells)
        basename = Path(ref.source_path).name
        rng = _table_rng(self._dataset, basename, self._row_cap_seed)
        idx = _sample_row_indices(n_rows, self._row_cap, rng)
        if idx.shape[0] < n_rows:
            cells = [cells[i] for i in idx.tolist()]
        return cells

    def encode_one(self, ref: ColumnRef, *, cells: list[str] | None = None) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("load() must be called before encode_one()")
        cells_per_ref = {ref.global_id: cells} if cells is not None else None
        return self.encode_batch([ref], cells_per_ref=cells_per_ref)[ref.global_id]

    def encode_batch(
        self,
        refs: list[ColumnRef],
        *,
        cells_per_ref: dict[int, list[str]] | None = None,
    ) -> dict[int, np.ndarray]:
        """Encode many columns via one Scorpion forward; cells come from the lake CSV unless supplied."""
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
            supplied = set(cells_per_ref.keys())
            required = set(gids)
            if supplied != required:
                raise ValueError(
                    f"cells_per_ref keys must match refs' global_ids exactly; "
                    f"missing={sorted(required - supplied)}, extra={sorted(supplied - required)}"
                )
            # A length mismatch would make _apply_row_cap draw a different row sample than at index time.
            mismatched = [
                (r.global_id, r.n_rows, len(cells_per_ref[r.global_id]))
                for r in refs if len(cells_per_ref[r.global_id]) != r.n_rows
            ]
            if mismatched:
                raise ValueError(
                    f"cells_per_ref length must match ref.n_rows for every ref; "
                    f"mismatches (gid, n_rows, supplied_len): {mismatched}"
                )

        per_ref_cells: list[list[str]] = []
        if cells_per_ref is None:
            df_by_source: dict[str, pd.DataFrame] = {}
            for ref in refs:
                if ref.source_path not in df_by_source:
                    df_by_source[ref.source_path] = pd.read_csv(
                        self._lake_root / ref.source_path,
                        dtype=str, keep_default_na=False,
                    )
            for ref in refs:
                cells = df_by_source[ref.source_path].iloc[:, ref.col_idx].astype(str).tolist()
                per_ref_cells.append(
                    _normalize_cells(self._apply_row_cap(ref, cells), self._cell_char_cap))
            del df_by_source
        else:
            for ref in refs:
                per_ref_cells.append(_normalize_cells(
                    self._apply_row_cap(ref, cells_per_ref[ref.global_id]), self._cell_char_cap))

        empty_ref_indices = [i for i, col in enumerate(per_ref_cells) if not col]
        if empty_ref_indices:
            empty_set = set(empty_ref_indices)
            kept_refs = [r for i, r in enumerate(refs) if i not in empty_set]
            kept_cells = [c for i, c in enumerate(per_ref_cells) if i not in empty_set]
        else:
            kept_refs = refs
            kept_cells = per_ref_cells

        flat: list[str] = [c for col in kept_cells for c in col]
        if not flat:
            return {r.global_id: np.zeros((self.dim,), dtype=np.float32) for r in refs}

        cell_mat = self._embed(flat)
        index_list = [len(col) for col in kept_cells]

        x = torch.from_numpy(cell_mat).to(self._device)
        with torch.no_grad():
            out = _forward_chunked(self._model, x, index_list, self._eval_chunk_size)
        out_cpu = out.detach().cpu().numpy().astype(np.float32)

        result = {r.global_id: out_cpu[i] for i, r in enumerate(kept_refs)}
        for i in empty_ref_indices:
            result[refs[i].global_id] = np.zeros((self.dim,), dtype=np.float32)
        return result

    def encode(self, columns: list[ColumnRef]) -> Iterator[tuple[int, np.ndarray]]:
        by_table: dict[str, list[ColumnRef]] = {}
        for ref in columns:
            by_table.setdefault(ref.source_path, []).append(ref)
        for refs in by_table.values():
            for gid, vec in self.encode_batch(refs).items():
                yield gid, vec
