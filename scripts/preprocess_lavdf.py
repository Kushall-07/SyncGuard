"""Preprocessing CLI script for LAV-DF dataset (Phase 11A).

Usage (from repo root)::

    # Dry-run scan across train (4000) and dev (1000):
    python scripts/preprocess_lavdf.py --dry-run

    # Small validation run on 5 train and 5 dev samples:
    python scripts/preprocess_lavdf.py --limit 5

    # Full resumable preprocessing:
    python scripts/preprocess_lavdf.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing.lavdf import (
    process_single_sample,
    validate_audio_cache,
    validate_landmarks_cache,
)


def read_manifest_samples(manifest_path: Path) -> list[dict[str, str]]:
    """Read sample rows from a manifest CSV."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    samples = []
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(row)
    return samples


def find_mp4_path(extracted_dir: Path, row: dict[str, str]) -> Path:
    """Resolve MP4 path from manifest row."""
    split = row.get("split", "")
    filename = Path(row["path"]).name
    path1 = extracted_dir / split / filename
    if path1.is_file():
        return path1
    path2 = extracted_dir / filename
    if path2.is_file():
        return path2
    return path1


def run_dry_run(
    manifest_train: Path,
    manifest_dev: Path,
    extracted_dir: Path,
    out_dir: Path,
) -> dict:
    """Scan manifests and report processing/skip state without running extraction."""
    out_audio_dir = out_dir / "audio"
    out_landmarks_dir = out_dir / "landmarks"

    results = {}
    total_requested = 0
    total_can_skip = 0
    total_needs_processing = 0
    total_missing_source = 0
    total_corrupt_outputs = 0

    splits = [("train", manifest_train), ("dev", manifest_dev)]
    for split_name, manifest_path in splits:
        if not manifest_path.is_file():
            continue
        samples = read_manifest_samples(manifest_path)
        split_stats = {
            "requested": len(samples),
            "can_skip": 0,
            "needs_processing": 0,
            "missing_source": 0,
            "corrupt_outputs": 0,
            "missing_audio": 0,
            "missing_landmarks": 0,
        }

        for row in samples:
            sample_id = row["sample_id"]
            mp4_path = find_mp4_path(extracted_dir, row)
            if not mp4_path.is_file():
                split_stats["missing_source"] += 1
                split_stats["needs_processing"] += 1
                continue

            wav_path = out_audio_dir / f"{sample_id}.wav"
            npz_path = out_landmarks_dir / f"{sample_id}.npz"
            meta_path = out_landmarks_dir / f"{sample_id}.meta.json"

            audio_ok = validate_audio_cache(wav_path, meta_path)
            landmarks_ok = validate_landmarks_cache(npz_path)

            if wav_path.exists() and not audio_ok:
                split_stats["corrupt_outputs"] += 1
            if npz_path.exists() and not landmarks_ok:
                split_stats["corrupt_outputs"] += 1

            if not audio_ok:
                split_stats["missing_audio"] += 1
            if not landmarks_ok:
                split_stats["missing_landmarks"] += 1

            if audio_ok and landmarks_ok and meta_path.is_file():
                split_stats["can_skip"] += 1
            else:
                split_stats["needs_processing"] += 1

        results[split_name] = split_stats
        total_requested += split_stats["requested"]
        total_can_skip += split_stats["can_skip"]
        total_needs_processing += split_stats["needs_processing"]
        total_missing_source += split_stats["missing_source"]
        total_corrupt_outputs += split_stats["corrupt_outputs"]

    results["total"] = {
        "requested": total_requested,
        "can_skip": total_can_skip,
        "needs_processing": total_needs_processing,
        "missing_source": total_missing_source,
        "corrupt_outputs": total_corrupt_outputs,
    }

    print("==================================================")
    print("LAV-DF PREPROCESSING DRY RUN REPORT")
    print("==================================================")
    for split_name in ("train", "dev"):
        if split_name in results:
            st = results[split_name]
            print(f"Split: {split_name.upper()}")
            print(f"  Requested samples:      {st['requested']}")
            print(f"  Can skip (valid cache): {st['can_skip']}")
            print(f"  Needs processing:       {st['needs_processing']}")
            print(f"  Missing audio:          {st['missing_audio']}")
            print(f"  Missing landmarks:      {st['missing_landmarks']}")
            print(f"  Missing source MP4:     {st['missing_source']}")
            print(f"  Corrupt/invalid cache:  {st['corrupt_outputs']}")
            print("--------------------------------------------------")

    print("TOTAL SUMMARY:")
    tot = results["total"]
    print(f"  Total requested:        {tot['requested']}")
    print(f"  Total can skip:         {tot['can_skip']}")
    print(f"  Total needs processing: {tot['needs_processing']}")
    print(f"  Total missing source:   {tot['missing_source']}")
    print(f"  Total corrupt cache:    {tot['corrupt_outputs']}")
    print("==================================================")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="LAV-DF Preprocessing Pipeline")
    parser.add_argument(
        "--manifest-train",
        type=Path,
        default=REPO_ROOT / "data" / "lavdf" / "manifest_train.csv",
        help="Path to manifest_train.csv",
    )
    parser.add_argument(
        "--manifest-dev",
        type=Path,
        default=REPO_ROOT / "data" / "lavdf" / "manifest_dev.csv",
        help="Path to manifest_dev.csv",
    )
    parser.add_argument(
        "--extracted-dir",
        type=Path,
        default=REPO_ROOT / "data" / "lavdf" / "extracted",
        help="Directory with extracted MP4 files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data" / "lavdf" / "processed",
        help="Output directory for processed audio and landmarks",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan cache status without processing files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples per split (e.g., 5 for validation)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force re-processing even if valid cache exists",
    )
    args = parser.parse_args()

    if args.dry_run:
        run_dry_run(
            args.manifest_train,
            args.manifest_dev,
            args.extracted_dir,
            args.out_dir,
        )
        return

    out_audio_dir = args.out_dir / "audio"
    out_landmarks_dir = args.out_dir / "landmarks"
    out_audio_dir.mkdir(parents=True, exist_ok=True)
    out_landmarks_dir.mkdir(parents=True, exist_ok=True)

    splits = [("train", args.manifest_train), ("dev", args.manifest_dev)]

    # Lazy initialization of landmarker
    from scripts.extract_celebdf_landmarks import DEFAULT_MODEL, ensure_model, make_landmarker

    model_path = ensure_model(DEFAULT_MODEL)
    landmarker = make_landmarker(model_path)

    total_processed = 0
    total_skipped = 0
    total_audio_failed = 0
    total_landmarks_failed = 0

    for split_name, manifest_path in splits:
        if not manifest_path.is_file():
            print(f"Warning: manifest for split '{split_name}' not found: {manifest_path}")
            continue

        samples = read_manifest_samples(manifest_path)
        if args.limit is not None and args.limit > 0:
            samples = samples[: args.limit]

        print(f"\nProcessing {len(samples)} samples for split '{split_name}'...")

        for i, row in enumerate(samples, 1):
            sample_id = row["sample_id"]
            mp4_path = find_mp4_path(args.extracted_dir, row)

            meta = process_single_sample(
                sample_id=sample_id,
                mp4_path=mp4_path,
                out_audio_dir=out_audio_dir,
                out_landmarks_dir=out_landmarks_dir,
                landmarker=landmarker,
                force=args.overwrite,
            )

            if meta.get("skipped"):
                total_skipped += 1
            else:
                total_processed += 1

            if not meta.get("audio_valid"):
                total_audio_failed += 1
            if not meta.get("landmarks_valid"):
                total_landmarks_failed += 1

            if i % 10 == 0 or i == len(samples):
                print(
                    f"  [{i}/{len(samples)}] sample {sample_id} | "
                    f"audio_valid={meta.get('audio_valid')} | "
                    f"landmarks_valid={meta.get('landmarks_valid')}"
                )

    print("\nProcessing Complete!")
    print(f"  Processed:         {total_processed}")
    print(f"  Skipped (cached):  {total_skipped}")
    print(f"  Audio failures:    {total_audio_failed}")
    print(f"  Landmark failures: {total_landmarks_failed}")


if __name__ == "__main__":
    main()
