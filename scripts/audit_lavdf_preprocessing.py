"""Audit utility for LAV-DF preprocessing pipeline (Phase 11A).

Usage::

    python scripts/audit_lavdf_preprocessing.py

Scans train and dev splits separately and reports:
- Requested samples
- Successful audio outputs / failed audio / missing audio
- Successful landmark outputs / failed landmarks / missing landmarks
- FPS statistics (min, max, mean, std)
- Duration statistics (min, max, mean, std)
- Corrupt outputs / incomplete outputs
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing.lavdf import (
    validate_audio_cache,
    validate_landmarks_cache,
)


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def audit_split(
    split_name: str,
    manifest_path: Path,
    out_dir: Path,
) -> dict:
    samples = read_manifest(manifest_path)
    out_audio_dir = out_dir / "audio"
    out_landmarks_dir = out_dir / "landmarks"

    stats = {
        "split": split_name,
        "requested": len(samples),
        "successful_audio": 0,
        "failed_audio": 0,
        "missing_audio": 0,
        "successful_landmarks": 0,
        "failed_landmarks": 0,
        "missing_landmarks": 0,
        "corrupt_outputs": 0,
        "incomplete_outputs": 0,
        "fps_list": [],
        "duration_list": [],
    }

    for row in samples:
        sample_id = row["sample_id"]
        wav_path = out_audio_dir / f"{sample_id}.wav"
        npz_path = out_landmarks_dir / f"{sample_id}.npz"
        meta_path = out_landmarks_dir / f"{sample_id}.meta.json"

        meta = None
        if meta_path.is_file():
            try:
                with meta_path.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                pass

        # Audio Audit
        audio_ok = validate_audio_cache(wav_path, meta_path)
        if audio_ok:
            stats["successful_audio"] += 1
        elif meta and not meta.get("audio_valid", True):
            stats["failed_audio"] += 1
        elif wav_path.exists() and not audio_ok:
            stats["corrupt_outputs"] += 1
            stats["failed_audio"] += 1
        else:
            stats["missing_audio"] += 1

        # Landmark Audit
        landmarks_ok = validate_landmarks_cache(npz_path)
        if landmarks_ok:
            stats["successful_landmarks"] += 1
            try:
                with np.load(npz_path) as data:
                    fps = float(data["fps"])
                    n_frames = int(data["points"].shape[0])
                    duration = n_frames / fps if fps > 0 else 0.0
                    stats["fps_list"].append(fps)
                    stats["duration_list"].append(duration)
            except Exception:
                pass
        elif meta and not meta.get("landmarks_valid", True):
            stats["failed_landmarks"] += 1
        elif npz_path.exists() and not landmarks_ok:
            stats["corrupt_outputs"] += 1
            stats["failed_landmarks"] += 1
        else:
            stats["missing_landmarks"] += 1

        if (not audio_ok) or (not landmarks_ok):
            stats["incomplete_outputs"] += 1

    return stats


def print_audit_report(stats: dict) -> None:
    print("==================================================")
    print(f"LAV-DF PREPROCESSING AUDIT REPORT — {stats['split'].upper()}")
    print("==================================================")
    print(f"Requested samples:          {stats['requested']}")
    print(f"Successful audio outputs:   {stats['successful_audio']}")
    print(f"Failed audio:               {stats['failed_audio']}")
    print(f"Missing audio:              {stats['missing_audio']}")
    print(f"Successful landmark outputs:{stats['successful_landmarks']}")
    print(f"Failed landmarks:           {stats['failed_landmarks']}")
    print(f"Missing landmarks:          {stats['missing_landmarks']}")
    print(f"Corrupt outputs:            {stats['corrupt_outputs']}")
    print(f"Incomplete samples:         {stats['incomplete_outputs']}")

    if stats["fps_list"]:
        fps_arr = np.array(stats["fps_list"])
        print("\nFPS Statistics:")
        print(f"  Min:  {fps_arr.min():.2f}")
        print(f"  Max:  {fps_arr.max():.2f}")
        print(f"  Mean: {fps_arr.mean():.2f}")
        print(f"  Std:  {fps_arr.std():.2f}")

    if stats["duration_list"]:
        dur_arr = np.array(stats["duration_list"])
        print("\nDuration Statistics (seconds):")
        print(f"  Min:  {dur_arr.min():.2f}")
        print(f"  Max:  {dur_arr.max():.2f}")
        print(f"  Mean: {dur_arr.mean():.2f}")
        print(f"  Std:  {dur_arr.std():.2f}")

    print("==================================================")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit LAV-DF Preprocessing Pipeline")
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
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data" / "lavdf" / "processed",
        help="Processed data output directory",
    )
    args = parser.parse_args()

    train_stats = audit_split("train", args.manifest_train, args.out_dir)
    print_audit_report(train_stats)

    dev_stats = audit_split("dev", args.manifest_dev, args.out_dir)
    print_audit_report(dev_stats)


if __name__ == "__main__":
    main()
