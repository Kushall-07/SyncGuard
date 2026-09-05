"""Build the ASVspoof 2019 LA manifest CSV from the official protocol files.

Usage (from the repo root)::

    python scripts/build_asvspoof_manifest.py --root data/asvspoof/LA \
        --out data/asvspoof/manifest.csv

Exits non-zero with a download hint if the dataset is not present.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.asvspoof_dataset import DOWNLOAD_HINT, build_asvspoof_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path,
                        help="Path to the 'LA' directory (contains ASVspoof2019_LA_cm_protocols/)")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "asvspoof" / "manifest.csv",
                        help="Output manifest CSV path")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "eval"],
                        choices=["train", "dev", "eval"])
    parser.add_argument("--no-verify-audio", action="store_true",
                        help="Skip checking that every referenced .flac exists")
    args = parser.parse_args()

    if not args.root.exists():
        print(DOWNLOAD_HINT, file=sys.stderr)
        return 2

    try:
        manifest = build_asvspoof_manifest(
            args.root,
            out_csv=args.out,
            splits=tuple(args.splits),
            verify_audio=not args.no_verify_audio,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"wrote {args.out}  ({len(manifest)} rows)")
    for split in args.splits:
        sub = manifest.split(split)
        by_label = Counter(r.label_name for r in sub)
        print(f"  {split:<6} n={len(sub):>7}  speakers={len(sub.speakers):>4}  "
              f"bonafide={by_label['bonafide']:>6}  spoof={by_label['spoof']:>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
