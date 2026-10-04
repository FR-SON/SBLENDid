"""Build the committed semantic test fixture from a dev-local _reference dir.

Run:
    python scripts/build_semantic_test_fixture.py --reference-dir _reference
"""
from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path


CSV_NAMES = [
    "SG_CSV0000000000000007.csv",
    "SG_CSV0000000000000008.csv",
    "SG_CSV0000000000000009.csv",
]
ASPECT_FILES = {
    "statistic": "opendata_statistic.pickle",
    "paragraph": "opendata_sample_32_paragraph__bert.pickle",
    "word":      "opendata_word_emb_64_sample.pickle",
    "number":    "opendata_number_emb_128_dim.pickle",
    "pattern":   "opendata_pattern_emb.pickle",
}
CKPT_PTH  = "opendata_judit_opendata_balanced.pth"
CKPT_JSON = "opendata_judit_opendata_balanced.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-dir", required=True, type=Path,
                    help="dev-local dir containing aspects/, csvs, and ckpt files")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("tests/fixtures/semantic"),
                    help="output fixture dir")
    args = ap.parse_args()

    ref: Path = args.reference_dir.resolve()
    out: Path = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "csvs").mkdir(exist_ok=True)
    (out / "ckpt").mkdir(exist_ok=True)
    (out / "aspects").mkdir(exist_ok=True)

    for name in CSV_NAMES:
        shutil.copyfile(ref / name, out / "csvs" / name)

    shutil.copyfile(ref / CKPT_PTH,  out / "ckpt" / CKPT_PTH)
    shutil.copyfile(ref / CKPT_JSON, out / "ckpt" / CKPT_JSON)

    keep = set(CSV_NAMES)
    for aspect, fname in ASPECT_FILES.items():
        src = ref / "aspects" / aspect / fname
        dst_dir = out / "aspects" / aspect
        dst_dir.mkdir(exist_ok=True)
        full = pickle.loads(src.read_bytes())
        filtered = {k: v for k, v in full.items() if k in keep}
        if not filtered:
            raise SystemExit(
                f"aspect {aspect!r}: none of {sorted(keep)} present in {src}; "
                "fixture would be empty"
            )
        (dst_dir / fname).write_bytes(pickle.dumps(filtered))
        print(f"{aspect}: kept {len(filtered)}/{len(full)} tables")

    print(f"wrote fixture to {out}")


if __name__ == "__main__":
    main()
