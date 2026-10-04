from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .aggregators.base import Aggregator
from .aggregators.decay_vote import DecayVoteAggregator
from .config import SemanticConfig
from .encoders.base import SegmentEncoder
# encoder modules (torch) are imported lazily in the build functions to keep this module torch-free.


@dataclass(frozen=True)
class ArtifactBundle:
    ckpt_path: Path
    sidecar: dict
    extras: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ApproachPlugin:
    name: str
    load_artifacts: Callable[[Path], ArtifactBundle]
    build_encoder: Callable[["ArtifactBundle", SemanticConfig], SegmentEncoder]
    build_aggregator: Callable[[], Aggregator]


_REGISTRY: dict[str, ApproachPlugin] = {}


def register(plugin: ApproachPlugin) -> None:
    _REGISTRY[plugin.name] = plugin


def get(name: str) -> ApproachPlugin:
    if name not in _REGISTRY:
        raise KeyError(f"unknown semantic approach {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


_SIDECAR_SUFFIXES = (".json", ".sidecar.json", ".hparams.json")


def _resolve_sidecar(ckpt: Path) -> Path:
    cands = [ckpt.with_name(ckpt.stem + sfx) for sfx in _SIDECAR_SUFFIXES]
    for cand in cands:
        if cand.is_file():
            return cand
    names = ", ".join(c.name for c in cands)
    raise FileNotFoundError(f"missing sidecar JSON next to {ckpt}: tried {names}")


def _liftus_load_artifacts(approach_dir: Path) -> ArtifactBundle:
    ckpt_dir = approach_dir / "ckpt"
    pths = sorted(ckpt_dir.glob("*.pth"))
    if not pths:
        raise FileNotFoundError(f"no .pth checkpoint in {ckpt_dir}")
    ckpt = pths[0]
    sidecar_path = _resolve_sidecar(ckpt)
    sidecar = json.loads(sidecar_path.read_text())
    for required in ("hidden_size", "num_heads", "left", "right", "train_dataset"):
        if required not in sidecar:
            raise KeyError(f"sidecar {sidecar_path} missing required field {required!r}")
    aspects_dir = approach_dir / "aspects"
    if not aspects_dir.is_dir():
        raise FileNotFoundError(f"missing aspects dir: {aspects_dir}")
    return ArtifactBundle(
        ckpt_path=ckpt,
        sidecar=sidecar,
        extras={"aspects_dir": aspects_dir, "dataset": sidecar["train_dataset"]},
    )


def _liftus_build_encoder(
    bundle: ArtifactBundle, cfg: SemanticConfig,
) -> SegmentEncoder:
    from .encoders.liftus import LiftusAdapter
    s = bundle.sidecar
    h = hashlib.sha256(bundle.ckpt_path.read_bytes()).hexdigest()[:7]
    enc = LiftusAdapter(
        left_aspects=s["left"],
        right_aspects=s["right"],
        hidden_size=s["hidden_size"],
        num_heads=s["num_heads"],
        aspects_dir=bundle.extras["aspects_dir"],
        dataset=bundle.extras["dataset"],
        device=cfg.device,
        encoder_version=f"liftus@{h}",
    )
    enc.load(bundle.ckpt_path)
    return enc


LIFTUS_PLUGIN = ApproachPlugin(
    name="liftus",
    load_artifacts=_liftus_load_artifacts,
    build_encoder=_liftus_build_encoder,
    build_aggregator=lambda: DecayVoteAggregator(),
)


def _snoopy_load_artifacts(approach_dir: Path) -> ArtifactBundle:
    ckpt_dir = approach_dir / "ckpt"
    pths = sorted(ckpt_dir.glob("*.pth"))
    if not pths:
        raise FileNotFoundError(f"no .pth checkpoint in {ckpt_dir}")
    ckpt = pths[0]
    sidecar_path = _resolve_sidecar(ckpt)
    sidecar = json.loads(sidecar_path.read_text())
    required = ("n_proxy_sets", "n_elements", "d", "cell_char_cap",
                "row_cap", "row_cap_seed", "train_dataset")
    for key in required:
        if key not in sidecar:
            raise KeyError(f"sidecar {sidecar_path} missing required field {key!r}")
    sidecar.setdefault("eval_chunk_size", 64)
    return ArtifactBundle(ckpt_path=ckpt, sidecar=sidecar)


def _snoopy_build_encoder(
    bundle: ArtifactBundle, cfg: SemanticConfig,
) -> SegmentEncoder:
    import os
    from .encoders.snoopy import ScorpionAdapter, _default_fasttext_embedder

    lake_root = cfg.dataset.dir() / "csvs"
    if not lake_root.is_dir():
        raise FileNotFoundError(
            f"snoopy build_encoder: lake_root {lake_root} does not exist. "
            f"Expected CSV directory under {cfg.dataset.dir()}; check "
            "[Dataset].name and [Dataset].root in config.ini."
        )
    ft_path_env = os.environ.get("BLEND_SNOOPY_FASTTEXT_PATH")
    if not ft_path_env:
        raise EnvironmentError("BLEND_SNOOPY_FASTTEXT_PATH is unset")
    ft_path = Path(ft_path_env)
    if not ft_path.is_file():
        raise FileNotFoundError(
            f"BLEND_SNOOPY_FASTTEXT_PATH={ft_path} does not exist"
        )
    s = bundle.sidecar
    h = hashlib.sha256(bundle.ckpt_path.read_bytes()).hexdigest()[:7]
    enc = ScorpionAdapter(
        n_proxy_sets=s["n_proxy_sets"],
        n_elements=s["n_elements"],
        d=s["d"],
        device=cfg.device,
        lake_root=lake_root,
        cell_embedder=_default_fasttext_embedder(d=s["d"], fasttext_path=ft_path),
        cell_char_cap=s["cell_char_cap"],
        row_cap=s["row_cap"],
        row_cap_seed=s["row_cap_seed"],
        dataset=s["train_dataset"],
        eval_chunk_size=s["eval_chunk_size"],
        encoder_version=f"snoopy@{h}",
    )
    enc.load(bundle.ckpt_path)
    return enc


SNOOPY_PLUGIN = ApproachPlugin(
    name="snoopy",
    load_artifacts=_snoopy_load_artifacts,
    build_encoder=_snoopy_build_encoder,
    build_aggregator=lambda: DecayVoteAggregator(),
)


def _deepjoin_load_artifacts(approach_dir: Path) -> ArtifactBundle:
    ckpt_dir = approach_dir / "ckpt"
    cands = []
    if ckpt_dir.is_dir():
        for d in sorted(ckpt_dir.iterdir()):
            if not d.is_dir() or d.name.startswith("._"):
                continue
            has_weights = (d / "model.safetensors").is_file() or (
                d / "pytorch_model.bin").is_file()
            if (d / "modules.json").is_file() and has_weights:
                cands.append(d)
    if len(cands) != 1:
        raise FileNotFoundError(
            f"expected exactly one HF SentenceTransformer dir under {ckpt_dir} "
            f"(modules.json + model weights, _trainer/ excluded by depth-1 "
            f"scan); found: {[d.name for d in cands]}"
        )
    hf_dir = cands[0]
    sidecar_path = hf_dir / "sidecar.json"
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"missing ckpt sidecar at {sidecar_path}")
    sidecar = json.loads(sidecar_path.read_text())
    if "dataset" not in sidecar:
        raise KeyError(
            f"sidecar {sidecar_path} missing required field 'dataset'"
        )
    sentences = approach_dir / "sentences" / f"{sidecar['dataset']}.flat.pkl"
    return ArtifactBundle(
        ckpt_path=hf_dir,
        sidecar=sidecar,
        extras={
            "dataset": sidecar["dataset"],
            "sentences_path": sentences if sentences.is_file() else None,
        },
    )


def _deepjoin_build_encoder(
    bundle: ArtifactBundle, cfg: SemanticConfig,
) -> SegmentEncoder:
    import os
    from .encoders.deepjoin import DeepJoinAdapter
    from .registry import sha256_ckpt

    nltk_dir = os.environ.get("BLEND_DEEPJOIN_NLTK_PATH")
    h = sha256_ckpt(bundle.ckpt_path)[:7]
    enc = DeepJoinAdapter(
        lake_root=cfg.dataset.dir() / "csvs",
        sentences_path=bundle.extras["sentences_path"],
        nltk_data_dir=Path(nltk_dir) if nltk_dir else None,
        device=cfg.device,
        encoder_version=f"deepjoin@{h}",
    )
    enc.load(bundle.ckpt_path)
    return enc


DEEPJOIN_PLUGIN = ApproachPlugin(
    name="deepjoin",
    load_artifacts=_deepjoin_load_artifacts,
    build_encoder=_deepjoin_build_encoder,
    build_aggregator=lambda: DecayVoteAggregator(),
)
