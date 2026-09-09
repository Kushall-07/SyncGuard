"""Phase 7: build the Celeb-DF-v2 train/dev/eval manifest CSV.

    python scripts/build_celebdf_manifest.py --root data/celebdf --out data/celebdf/manifest.csv

The eval split is ``List_of_testing_videos.txt`` verbatim (the file is only read);
the rest is split train/dev **strictly identity-disjoint** by identity
co-occurrence component (see src/data/celebdf_dataset.py). With
``--min-valid-frames`` (and ``--landmarks-dir``), videos with too few detected
faces are excluded from every split and recorded in ``<out>.audit.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.celebdf_dataset import build_celebdf_manifest, identities_of_sample  # noqa: E402
from src.data.manifests import LeakageError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "data" / "celebdf")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "celebdf" / "manifest.csv")
    parser.add_argument("--dev-identity-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--landmarks-dir", type=Path,
                        help="cached landmark dir (required with --min-valid-frames)")
    parser.add_argument("--min-valid-frames", type=int,
                        help="exclude videos with fewer detected-face frames than this")
    args = parser.parse_args()

    try:
        manifest, report = build_celebdf_manifest(
            args.root,
            out_csv=args.out,
            dev_identity_frac=args.dev_identity_frac,
            seed=args.seed,
            landmarks_dir=args.landmarks_dir,
            min_valid_frames=args.min_valid_frames,
            return_report=True,
        )
    except LeakageError as exc:  # pragma: no cover - defensive
        raise SystemExit(f"identity leakage: {exc}")

    audit_path = args.out.with_suffix(args.out.suffix + ".audit.json")
    audit_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    sizes = report["component_sizes"]
    top = ", ".join(str(s) for s in sizes[:8])
    singletons = sum(1 for s in sizes if s == 1)
    print(f"\nmanifest -> {args.out}    audit -> {audit_path}")
    print(f"total kept: {len(manifest)} videos    "
          f"identity components: [{top}{', ...' if len(sizes) > 8 else ''}]  "
          f"({len(sizes)} total, {singletons} singleton)")
    for split in ("train", "dev", "eval"):
        rows = list(manifest.split(split))
        pos = sum(1 for r in rows if r.label == 1)
        ids = {i for r in rows for i in identities_of_sample(r.sample_id)}
        print(f"  {split:5} n={len(rows):5}  real={pos:5}  fake={len(rows) - pos:5}  identities={len(ids)}")

    print(f"\ntrain identities: {report['train_identities']}   dev identities: {report['dev_identities']}   "
          f"train_dev_identity_overlap: {report['train_dev_identity_overlap']}")
    print("train INTERSECT dev = 0  (strict identity-disjoint, source + target)"
          if report["train_dev_identity_overlap"] == 0
          else "LEAK DETECTED - see audit")

    if report["min_valid_frames"] is not None:
        print(f"\nmin_valid_frames = {report['min_valid_frames']}")
        print(f"excluded videos: {report['n_excluded']}")
        print(f"  by split : {report['excluded_by_split']}")
        print(f"  by reason: {report['excluded_by_reason']}")
        for e in report["excluded"][:10]:
            print(f"    {e['split']:5} {e['sample_id']:32} n_valid={e['n_valid']:3}  {e['reason']}")
        if report["n_excluded"] > 10:
            print(f"    ... {report['n_excluded'] - 10} more in {audit_path}")
    else:
        print("\n(no --min-valid-frames given: no landmark-quality exclusion applied)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
