"""Regenerates tests/fixtures/semantic_join/ckpt/tiny.{pth,json}."""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
from snoopy.model import Scorpion


_DIMS = dict(n_proxy_sets=4, n_elements=2, d=4)


def build(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(device="cpu", **_DIMS)
    model = Scorpion(args)
    torch.save(model.state_dict(), out_dir / "tiny.pth")
    sidecar = {
        **_DIMS,
        "cell_char_cap": 64,
        "row_cap": 8,
        "row_cap_seed": 0,
        "train_dataset": "fixture",
        "eval_chunk_size": 4,
    }
    (out_dir / "tiny.json").write_text(json.dumps(sidecar, indent=2))


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    build(here / "ckpt")
    print(f"wrote {here / 'ckpt' / 'tiny.pth'} + tiny.json")
