"""Build LAV-DF subset manifest CSV from metadata.min.json inside the TAR archive.

Usage (from the repo root)::

    python scripts/build_lavdf_manifest.py --tar data/lavdf/LAV-DF.tar \\
        --out data/lavdf/manifest.csv --real 1000 --audio-only 500 \\
        --video-only 500 --audio-video 500 --seed 42

The script reads metadata.min.json directly from the TAR without extracting it.
Supports dry-run mode (--dry-run) to print statistics without writing output.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import tarfile
from collections import Counter
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class LAVDFRow:
    """Single LAV-DF sample row for the manifest."""
    sample_id: str
    path: str
    label: int  # 1 = real, 0 = manipulated
    label_name: str  # "real" | "manipulated"
    split: str  # "train" | "dev" | "test"
    modify_audio: bool
    modify_video: bool
    n_fakes: int
    fake_periods: str  # JSON array as string, preserved for temporal eval
    duration: float
    original: str
    video_frames: int
    audio_frames: int

    def __post_init__(self) -> None:
        if self.label not in (0, 1):
            raise ValueError(f"label must be 0 or 1, got {self.label!r}")
        expected = "real" if self.label == 1 else "manipulated"
        if self.label_name != expected:
            raise ValueError(
                f"label_name {self.label_name!r} != {expected!r} for label {self.label}"
            )
        if self.split not in ("train", "dev", "test"):
            raise ValueError(f"split must be train/dev/test, got {self.split!r}")


class LAVDFManifest:
    """LAV-DF manifest with CSV IO and selection helpers."""

    columns: tuple[str, ...] = tuple(f.name for f in fields(LAVDFRow))

    def __init__(self, rows: Iterable[LAVDFRow]) -> None:
        self.rows: list[LAVDFRow] = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[LAVDFRow]:
        return iter(self.rows)

    def __getitem__(self, index: int) -> LAVDFRow:
        return self.rows[index]

    def write_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.columns)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({c: getattr(row, c) for c in self.columns})
        return path

    def filter(self, **conditions: object) -> "LAVDFManifest":
        """Return rows matching every condition. A value may be a scalar or a set/list."""

        def matches(row: LAVDFRow) -> bool:
            for key, wanted in conditions.items():
                value = getattr(row, key)
                if isinstance(wanted, (set, frozenset, list, tuple)):
                    if value not in wanted:
                        return False
                elif value != wanted:
                    return False
            return True

        return LAVDFManifest(r for r in self.rows if matches(r))

    def split(self, name: str) -> "LAVDFManifest":
        return self.filter(split=name)

    def subset(
        self,
        n: int,
        *,
        seed: int = 0,
        stratify_by: str | None = None,
    ) -> "LAVDFManifest":
        """Deterministic sub-sample of at most n rows."""
        if n >= len(self.rows):
            return LAVDFManifest(self.rows)

        rng = random.Random(seed)
        pool = list(self.rows)
        rng.shuffle(pool)

        if not stratify_by or n >= len(pool):
            return LAVDFManifest(pool[:n])

        from collections import defaultdict

        groups: dict[object, list[LAVDFRow]] = defaultdict(list)
        for row in pool:
            groups[getattr(row, stratify_by)].append(row)

        total = len(pool)
        picked: list[LAVDFRow] = []
        for key, group in groups.items():
            take = round(n * len(group) / total)
            picked.extend(group[:take])

        if len(picked) > n:
            picked = picked[:n]
        elif len(picked) < min(n, len(pool)):
            remaining = [r for r in pool if r not in set(picked)]
            picked.extend(remaining[: n - len(picked)])

        picked.sort(key=lambda r: pool.index(r))
        return LAVDFManifest(picked)

    def statistics(self) -> dict:
        """Compute manifest statistics."""
        durations = [r.duration for r in self.rows]
        return {
            "total": len(self.rows),
            "split_sizes": dict(Counter(r.split for r in self.rows)),
            "real": sum(1 for r in self.rows if r.label == 1),
            "manipulated": sum(1 for r in self.rows if r.label == 0),
            "audio_only": sum(
                1
                for r in self.rows
                if r.label == 0 and r.modify_audio and not r.modify_video
            ),
            "video_only": sum(
                1
                for r in self.rows
                if r.label == 0 and not r.modify_audio and r.modify_video
            ),
            "audio_video": sum(
                1
                for r in self.rows
                if r.label == 0 and r.modify_audio and r.modify_video
            ),
            "duration_min": min(durations) if durations else 0.0,
            "duration_max": max(durations) if durations else 0.0,
            "duration_mean": sum(durations) / len(durations) if durations else 0.0,
        }


def parse_metadata_from_tar(tar_path: Path) -> list[dict]:
    """Read metadata.min.json from inside the TAR without extracting."""
    if not tar_path.exists():
        raise FileNotFoundError(f"TAR file not found: {tar_path}")

    with tarfile.open(tar_path, "r") as tar:
        # Try both possible paths
        for member_name in ["LAV-DF/metadata.min.json", "metadata.min.json"]:
            try:
                member = tar.getmember(member_name)
                f = tar.extractfile(member)
                if f is None:
                    continue
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and "samples" in data:
                    return data["samples"]
                else:
                    raise ValueError(
                        f"Unexpected metadata format in {member_name}"
                    )
            except KeyError:
                continue

    raise FileNotFoundError(
        f"metadata.min.json not found in TAR {tar_path}. "
        "Expected 'LAV-DF/metadata.min.json' or 'metadata.min.json'"
    )


def classify_category(entry: dict) -> str:
    """Classify a metadata entry into one of four categories."""
    n_fakes = entry.get("n_fakes", 0)
    modify_audio = entry.get("modify_audio", False)
    modify_video = entry.get("modify_video", False)

    if n_fakes == 0:
        return "real"
    elif modify_audio and not modify_video:
        return "audio_only"
    elif not modify_audio and modify_video:
        return "video_only"
    elif modify_audio and modify_video:
        return "audio_video"
    else:
        # n_fakes > 0 but both modify flags false - shouldn't happen in valid data
        return "unknown"


def metadata_to_row(entry: dict) -> LAVDFRow:
    """Convert a metadata entry to a LAVDFRow."""
    file_path = entry["file"]
    n_fakes = entry.get("n_fakes", 0)
    label = 1 if n_fakes == 0 else 0
    label_name = "real" if label == 1 else "manipulated"

    # Preserve fake_periods as JSON string
    fake_periods = entry.get("fake_periods", [])
    if isinstance(fake_periods, list):
        fake_periods_str = json.dumps(fake_periods)
    else:
        fake_periods_str = str(fake_periods)

    # Normalize path to canonical form with LAV-DF prefix
    # Metadata may have "train/file.mp4" but TAR has "LAV-DF/train/file.mp4"
    if not file_path.startswith("LAV-DF/"):
        canonical_path = f"LAV-DF/{file_path}"
    else:
        canonical_path = file_path

    return LAVDFRow(
        sample_id=Path(file_path).stem,
        path=canonical_path,
        label=label,
        label_name=label_name,
        split=entry["split"],
        modify_audio=entry.get("modify_audio", False),
        modify_video=entry.get("modify_video", False),
        n_fakes=n_fakes,
        fake_periods=fake_periods_str,
        duration=float(entry.get("duration", 0.0)),
        original=entry.get("original", ""),
        video_frames=int(entry.get("video_frames", 0)),
        audio_frames=int(entry.get("audio_frames", 0)),
    )


def verify_tar_members(tar_path: Path, rows: list[LAVDFRow]) -> list[str]:
    """Verify that all selected paths exist as TAR members. Returns missing paths."""
    with tarfile.open(tar_path, "r") as tar:
        members = {m.name for m in tar.getmembers()}
    missing = [r.path for r in rows if r.path not in members]
    return missing


def build_lavdf_manifest(
    tar_path: Path,
    category_counts: dict[str, int],
    *,
    seed: int = 0,
    splits: tuple[str, ...] = ("train", "dev"),
    verify_members: bool = True,
) -> LAVDFManifest:
    """Build LAV-DF manifest with category-based sampling.

    Args:
        tar_path: Path to LAV-DF.tar
        category_counts: Dict mapping category name to desired count.
            Categories: real, audio_only, video_only, audio_video
        seed: Random seed for deterministic sampling
        splits: Which splits to include (test is never included)
        verify_members: If True, verify all paths exist in TAR

    Returns:
        LAVDFManifest with selected samples
    """
    if "test" in splits:
        raise ValueError("test split must never be used for development subsets")

    metadata = parse_metadata_from_tar(tar_path)

    # Convert to rows and filter by allowed splits
    all_rows = [metadata_to_row(entry) for entry in metadata]
    allowed_rows = [r for r in all_rows if r.split in splits]

    # Group by category
    by_category: dict[str, list[LAVDFRow]] = {}
    for row in allowed_rows:
        cat = classify_category(
            {
                "n_fakes": row.n_fakes,
                "modify_audio": row.modify_audio,
                "modify_video": row.modify_video,
            }
        )
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(row)

    # Sample from each category
    selected: list[LAVDFRow] = []
    for cat, count in category_counts.items():
        if count <= 0:
            continue
        if cat not in by_category:
            raise ValueError(
                f"Category '{cat}' requested but no samples found in metadata"
            )
        pool = by_category[cat]
        if len(pool) < count:
            raise ValueError(
                f"Requested {count} samples for category '{cat}' "
                f"but only {len(pool)} available"
            )
        rng = random.Random(seed)
        sampled = rng.sample(pool, count)
        selected.extend(sampled)

    # Verify TAR members if requested
    if verify_members:
        missing = verify_tar_members(tar_path, selected)
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} paths not found in TAR: {missing[:5]}..."
            )

    return LAVDFManifest(selected)


def print_statistics(manifest: LAVDFManifest) -> None:
    """Print manifest statistics to stdout."""
    stats = manifest.statistics()
    print(f"Total selected: {stats['total']}")
    print(f"Split distribution: {stats['split_sizes']}")
    print(f"Real: {stats['real']}")
    print(f"Manipulated: {stats['manipulated']}")
    print(f"Audio-only: {stats['audio_only']}")
    print(f"Video-only: {stats['video_only']}")
    print(f"Audio+Video: {stats['audio_video']}")
    print(
        f"Duration: min={stats['duration_min']:.2f}s "
        f"max={stats['duration_max']:.2f}s "
        f"mean={stats['duration_mean']:.2f}s"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tar",
        type=Path,
        required=True,
        help="Path to LAV-DF.tar archive",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "lavdf" / "manifest.csv",
        help="Output manifest CSV path",
    )
    parser.add_argument(
        "--real",
        type=int,
        default=0,
        help="Number of real samples to include (n_fakes == 0)",
    )
    parser.add_argument(
        "--audio-only",
        type=int,
        default=0,
        help="Number of audio-only manipulated samples",
    )
    parser.add_argument(
        "--video-only",
        type=int,
        default=0,
        help="Number of video-only manipulated samples",
    )
    parser.add_argument(
        "--audio-video",
        type=int,
        default=0,
        help="Number of audio+video manipulated samples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "dev"],
        choices=["train", "dev", "test"],
        help="Which splits to include (test is excluded by default)",
    )
    parser.add_argument(
        "--no-verify-members",
        action="store_true",
        help="Skip TAR member existence verification",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print statistics without writing output file",
    )

    args = parser.parse_args()

    # Build category counts
    category_counts = {
        "real": args.real,
        "audio_only": args.audio_only,
        "video_only": args.video_only,
        "audio_video": args.audio_video,
    }

    # Remove zero-count categories
    category_counts = {k: v for k, v in category_counts.items() if v > 0}

    if not category_counts:
        parser.error("At least one category count must be positive")

    # Ensure test is not used
    if "test" in args.splits:
        print("WARNING: test split should never be used for development subsets", file=__import__("sys").stderr)

    try:
        manifest = build_lavdf_manifest(
            args.tar,
            category_counts,
            seed=args.seed,
            splits=tuple(args.splits),
            verify_members=not args.no_verify_members,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=__import__("sys").stderr)
        return 2
    except ValueError as exc:
        print(str(exc), file=__import__("sys").stderr)
        return 1

    print_statistics(manifest)

    if not args.dry_run:
        manifest.write_csv(args.out)
        print(f"\nWrote manifest to: {args.out}")
    else:
        print("\nDry run: no file written")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
