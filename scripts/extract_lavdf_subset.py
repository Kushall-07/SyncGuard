"""Selective extraction of LAV-DF subset from TAR archive based on manifest files.

Usage (from the repo root)::

    python scripts/extract_lavdf_subset.py --tar data/lavdf/LAV-DF.tar \\
        --manifest data/lavdf/manifest_train.csv \\
        --manifest data/lavdf/manifest_dev.csv \\
        --out-dir data/lavdf/extracted

Extracts ONLY files listed in the manifests, preserving train/dev directory structure.
Supports dry-run mode (--dry-run) to preview extraction without writing files.
"""

from __future__ import annotations

import argparse
import csv
import os
import tarfile
from collections import Counter
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO_ROOT))


def read_manifest_paths(manifest_path: Path) -> set[str]:
    """Read all paths from a manifest CSV file."""
    paths: set[str] = set()
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            paths.add(row["path"])
    return paths


def validate_tar_path(path: str) -> str | None:
    """Validate and normalize a TAR path for safe extraction.

    Returns None if the path is invalid (traversal attempt, wrong split, etc.).
    Otherwise returns the normalized path.
    """
    # Reject path traversal attempts
    if ".." in path or path.startswith("/"):
        return None

    # Ensure path starts with LAV-DF/
    if not path.startswith("LAV-DF/"):
        return None

    # Extract the split component
    parts = path.split("/")
    if len(parts) < 2:
        return None

    split = parts[1]  # e.g., "train", "dev", "test"

    # Reject test split
    if split == "test":
        return None

    # Only allow train and dev
    if split not in ("train", "dev"):
        return None

    return path


def extract_subset(
    tar_path: Path,
    manifest_paths: set[str],
    out_dir: Path,
    *,
    dry_run: bool = False,
) -> dict:
    """Extract files from TAR that are listed in manifest_paths.

    Args:
        tar_path: Path to LAV-DF.tar
        manifest_paths: Set of canonical paths to extract (e.g., "LAV-DF/train/file.mp4")
        out_dir: Output directory for extracted files
        dry_run: If True, don't actually extract files

    Returns:
        Dictionary with extraction statistics
    """
    stats = {
        "requested": len(manifest_paths),
        "extracted": 0,
        "already_present": 0,
        "failed": 0,
        "total_bytes": 0,
        "missing_from_tar": [],
    }

    if not tar_path.exists():
        raise FileNotFoundError(f"TAR file not found: {tar_path}")

    # Create output directories
    for split in ("train", "dev"):
        (out_dir / split).mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "r") as tar:
        tar_members = {m.name: m for m in tar.getmembers()}

        for path in manifest_paths:
            # Validate path
            validated = validate_tar_path(path)
            if validated is None:
                stats["failed"] += 1
                continue

            # Check if path exists in TAR
            if validated not in tar_members:
                stats["missing_from_tar"].append(validated)
                stats["failed"] += 1
                continue

            member = tar_members[validated]

            # Skip non-regular files (directories, symlinks, etc.)
            if not member.isfile():
                stats["failed"] += 1
                continue

            # Determine output path
            # Remove LAV-DF/ prefix for output directory structure
            relative_path = validated.removeprefix("LAV-DF/")
            output_path = out_dir / relative_path

            # Skip if file already exists and is non-empty
            if output_path.exists() and output_path.stat().st_size > 0:
                stats["already_present"] += 1
                continue

            if dry_run:
                stats["extracted"] += 1
                stats["total_bytes"] += member.size
                continue

            # Extract the file manually to control output path
            try:
                # Ensure parent directory exists
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Extract file content
                f = tar.extractfile(member)
                if f is None:
                    stats["failed"] += 1
                    continue

                # Write to output path
                with output_path.open("wb") as out_f:
                    out_f.write(f.read())

                # Verify the extracted file
                if not output_path.exists():
                    stats["failed"] += 1
                    continue

                if output_path.stat().st_size == 0:
                    stats["failed"] += 1
                    output_path.unlink()
                    continue

                stats["extracted"] += 1
                stats["total_bytes"] += member.size
            except Exception as e:
                stats["failed"] += 1
                # Clean up partial extraction if possible
                if output_path.exists():
                    try:
                        output_path.unlink()
                    except Exception:
                        pass

    return stats


def print_statistics(stats: dict, dry_run: bool = False) -> None:
    """Print extraction statistics."""
    print(f"Requested: {stats['requested']}")
    print(f"Extracted: {stats['extracted']}")
    print(f"Already present: {stats['already_present']}")
    print(f"Failed: {stats['failed']}")

    if stats["missing_from_tar"]:
        print(f"Missing from TAR: {len(stats['missing_from_tar'])}")
        for path in stats["missing_from_tar"][:5]:
            print(f"  - {path}")
        if len(stats["missing_from_tar"]) > 5:
            print(f"  ... and {len(stats['missing_from_tar']) - 5} more")

    total_mb = stats["total_bytes"] / (1024 * 1024)
    print(f"Total bytes: {stats['total_bytes']:,} ({total_mb:.2f} MB)")

    if dry_run:
        print("\nDry run: no files extracted")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tar",
        type=Path,
        required=True,
        help="Path to LAV-DF.tar archive",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        required=True,
        help="Path to manifest CSV file (can be specified multiple times)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data" / "lavdf" / "extracted",
        help="Output directory for extracted files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview extraction without writing files",
    )

    args = parser.parse_args()

    # Read all manifest paths
    all_paths: set[str] = set()
    for manifest_path in args.manifest:
        if not manifest_path.exists():
            print(f"Manifest not found: {manifest_path}", file=__import__("sys").stderr)
            return 2
        paths = read_manifest_paths(manifest_path)
        all_paths.update(paths)
        print(f"Loaded {len(paths)} paths from {manifest_path}")

    if not all_paths:
        parser.error("No paths found in manifests")

    print(f"Total unique paths to process: {len(all_paths)}")

    try:
        stats = extract_subset(
            args.tar,
            all_paths,
            args.out_dir,
            dry_run=args.dry_run,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=__import__("sys").stderr)
        return 2
    except Exception as exc:
        print(f"Error during extraction: {exc}", file=__import__("sys").stderr)
        return 1

    print()
    print_statistics(stats, dry_run=args.dry_run)

    if stats["failed"] > 0:
        print(f"\nWarning: {stats['failed']} files failed to extract", file=__import__("sys").stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
