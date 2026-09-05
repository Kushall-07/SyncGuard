"""Dataset-agnostic sample manifest (Phase 3).

A *manifest* is a flat table where one row describes one audio sample: where it
lives on disk, its label, which speaker produced it, and which split it belongs
to. Every dataset in SyncGuard (ASVspoof now, the DFDC audio track later) is
converted to this one representation so the ``Dataset`` / ``Trainer`` code never
has to know a dataset's native protocol format.

Label convention (kept consistent with :mod:`src.evaluation.metrics`, where the
positive class is ``1``): ``1 = bonafide/genuine``, ``0 = spoof/synthetic``.

The :func:`synthetic_manifest` helper builds a manifest of generated WAV files
with speaker-disjoint splits, so the loader and training wiring can be exercised
before any real dataset is downloaded.
"""

from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np

from src.preprocessing.synthetic import chirp, multi_tone, sine, white_noise, write_wav

__all__ = [
    "ManifestRow",
    "Manifest",
    "LeakageError",
    "assert_speaker_disjoint",
    "synthetic_manifest",
    "BONAFIDE",
    "SPOOF",
    "SPLITS",
]

BONAFIDE = 1
SPOOF = 0
_LABEL_NAME = {BONAFIDE: "bonafide", SPOOF: "spoof"}
SPLITS = ("train", "dev", "eval")


class LeakageError(RuntimeError):
    """Raised when the same speaker appears in more than one split."""


@dataclass(frozen=True)
class ManifestRow:
    sample_id: str
    path: str
    label: int            # 1 = bonafide, 0 = spoof
    label_name: str       # "bonafide" | "spoof"
    speaker: str
    split: str            # "train" | "dev" | "eval"
    source: str = ""      # dataset tag, e.g. "asvspoof2019-la"
    attack: str = "-"     # system / attack id ("-" for bonafide)

    def __post_init__(self) -> None:
        if self.label not in _LABEL_NAME:
            raise ValueError(f"label must be 0 or 1, got {self.label!r}")
        expected = _LABEL_NAME[self.label]
        if self.label_name != expected:
            raise ValueError(f"label_name {self.label_name!r} != {expected!r} for label {self.label}")

    @classmethod
    def make(
        cls,
        *,
        sample_id: str,
        path: str | Path,
        label: int,
        speaker: str,
        split: str,
        source: str = "",
        attack: str = "-",
    ) -> "ManifestRow":
        return cls(
            sample_id=sample_id,
            path=str(path),
            label=int(label),
            label_name=_LABEL_NAME[int(label)],
            speaker=speaker,
            split=split,
            source=source,
            attack=attack,
        )


class Manifest:
    """An ordered collection of :class:`ManifestRow` with CSV IO and split helpers."""

    columns: tuple[str, ...] = tuple(f.name for f in fields(ManifestRow))

    def __init__(self, rows: Iterable[ManifestRow]) -> None:
        self.rows: list[ManifestRow] = list(rows)

    # ------------------------------------------------------------- sequence api

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[ManifestRow]:
        return iter(self.rows)

    def __getitem__(self, index: int) -> ManifestRow:
        return self.rows[index]

    def __repr__(self) -> str:
        dist = self.label_distribution()
        return (
            f"Manifest(n={len(self)}, speakers={len(self.speakers)}, "
            f"splits={sorted(self.split_sizes())}, labels={dist})"
        )

    # -------------------------------------------------------------------- io

    def write_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.columns)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({c: getattr(row, c) for c in self.columns})
        return path

    @classmethod
    def read_csv(cls, path: str | Path) -> "Manifest":
        path = Path(path)
        with path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            missing = set(cls.columns) - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path}: manifest missing columns {sorted(missing)}")
            rows = [
                ManifestRow.make(
                    sample_id=r["sample_id"],
                    path=r["path"],
                    label=int(r["label"]),
                    speaker=r["speaker"],
                    split=r["split"],
                    source=r.get("source", ""),
                    attack=r.get("attack", "-"),
                )
                for r in reader
            ]
        return cls(rows)

    @classmethod
    def concat(cls, *manifests: "Manifest") -> "Manifest":
        out: list[ManifestRow] = []
        for m in manifests:
            out.extend(m.rows)
        return cls(out)

    # ------------------------------------------------------------- inspection

    @property
    def speakers(self) -> set[str]:
        return {r.speaker for r in self.rows}

    def split_sizes(self) -> dict[str, int]:
        counter: Counter[str] = Counter(r.split for r in self.rows)
        return dict(counter)

    def label_distribution(self) -> dict[str, int]:
        counter: Counter[str] = Counter(r.label_name for r in self.rows)
        return dict(counter)

    # --------------------------------------------------------------- selection

    def filter(self, **conditions: object) -> "Manifest":
        """Return rows matching every condition. A value may be a scalar or a set/list."""

        def matches(row: ManifestRow) -> bool:
            for key, wanted in conditions.items():
                value = getattr(row, key)
                if isinstance(wanted, (set, frozenset, list, tuple)):
                    if value not in wanted:
                        return False
                elif value != wanted:
                    return False
            return True

        return Manifest(r for r in self.rows if matches(r))

    def split(self, name: str) -> "Manifest":
        return self.filter(split=name)

    def subset(
        self,
        n: int,
        *,
        seed: int = 0,
        stratify_by: str | None = "label",
        per_speaker_cap: int | None = None,
    ) -> "Manifest":
        """A deterministic sub-sample of at most ``n`` rows.

        With ``stratify_by`` set, the class (or other field) proportions of the
        source are preserved as closely as integer counts allow. ``per_speaker_cap``
        limits how many rows any one speaker may contribute (useful for quick,
        speaker-balanced development subsets).
        """

        if n >= len(self.rows) and per_speaker_cap is None:
            return Manifest(self.rows)

        rng = random.Random(seed)
        pool = list(self.rows)
        rng.shuffle(pool)

        if per_speaker_cap is not None:
            seen: Counter[str] = Counter()
            capped = []
            for row in pool:
                if seen[row.speaker] < per_speaker_cap:
                    capped.append(row)
                    seen[row.speaker] += 1
            pool = capped

        if not stratify_by or n >= len(pool):
            return Manifest(pool[:n])

        groups: dict[object, list[ManifestRow]] = defaultdict(list)
        for row in pool:
            groups[getattr(row, stratify_by)].append(row)

        total = len(pool)
        picked: list[ManifestRow] = []
        for key, group in groups.items():
            take = round(n * len(group) / total)
            picked.extend(group[:take])

        # Correct rounding drift back to exactly n (or as close as the pool allows).
        if len(picked) > n:
            picked = picked[:n]
        elif len(picked) < min(n, len(pool)):
            remaining = [r for r in pool if r not in set(picked)]
            picked.extend(remaining[: n - len(picked)])

        picked.sort(key=lambda r: pool.index(r))
        return Manifest(picked)

    # ------------------------------------------------------------- leakage check

    def check_disjoint(self, other: "Manifest", *, on: str = "speaker") -> bool:
        mine = {getattr(r, on) for r in self.rows}
        theirs = {getattr(r, on) for r in other.rows}
        return mine.isdisjoint(theirs)

    def assert_splits_speaker_disjoint(self) -> None:
        by_split: dict[str, set[str]] = defaultdict(set)
        for row in self.rows:
            by_split[row.split].add(row.speaker)
        assert_speaker_disjoint(**by_split)


def assert_speaker_disjoint(**named_speaker_sets: Sequence[str] | set[str]) -> None:
    """Raise :class:`LeakageError` if any two named groups share a speaker.

    Accepts either sets of speaker ids or :class:`Manifest` objects.
    """

    resolved: dict[str, set[str]] = {}
    for name, value in named_speaker_sets.items():
        if isinstance(value, Manifest):
            resolved[name] = value.speakers
        else:
            resolved[name] = set(value)

    names = list(resolved)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlap = resolved[a] & resolved[b]
            if overlap:
                sample = sorted(overlap)[:5]
                raise LeakageError(
                    f"{len(overlap)} speaker(s) appear in both '{a}' and '{b}' "
                    f"(e.g. {sample}) - splits would leak"
                )


# --------------------------------------------------------------------- synthetic

# Deterministic label -> generator mapping for the synthetic dataset. The exact
# choice is arbitrary; it only has to be consistent and give the two classes
# visibly different spectra.
_BONAFIDE_GENS = (lambda s: sine(220.0, seed=s), lambda s: chirp(80.0, 3800.0, seed=s))
_SPOOF_GENS = (lambda s: white_noise(seed=s), lambda s: multi_tone((180.0, 540.0, 1500.0), seed=s))


def synthetic_manifest(
    out_dir: str | Path,
    *,
    n_per_split: Mapping[str, int] | int = 24,
    n_speakers_per_split: int = 4,
    seed: int = 0,
    sample_rate: int = 16000,
    duration_s: float = 1.0,
    source: str = "synthetic",
) -> Manifest:
    """Generate WAV files with speaker-disjoint splits and return their manifest.

    Speakers are named ``SYN-<split>-<k>`` so they can never collide across
    splits. Labels are balanced within each split.
    """

    out_dir = Path(out_dir)
    if isinstance(n_per_split, int):
        n_per_split = {s: n_per_split for s in SPLITS}

    rng = random.Random(seed)
    rows: list[ManifestRow] = []

    for split, count in n_per_split.items():
        speakers = [f"SYN-{split}-{k:02d}" for k in range(n_speakers_per_split)]
        for i in range(count):
            label = BONAFIDE if i % 2 == 0 else SPOOF
            gen_pool = _BONAFIDE_GENS if label == BONAFIDE else _SPOOF_GENS
            gen = gen_pool[i % len(gen_pool)]
            gen_seed = rng.randint(0, 2**31 - 1)
            samples, sr = gen(gen_seed)
            if sr != sample_rate:
                raise ValueError("synthetic generators must already be at sample_rate")
            # Trim/tile to the requested duration for predictable shapes.
            target = int(round(duration_s * sample_rate))
            if samples.shape[0] < target:
                reps = target // samples.shape[0] + 1
                samples = np.tile(samples, reps)[:target].astype("float32")
            else:
                samples = samples[:target]

            speaker = speakers[i % len(speakers)]
            sample_id = f"{split}_{i:04d}"
            path = out_dir / split / f"{sample_id}.wav"
            write_wav(path, samples, sample_rate)
            rows.append(
                ManifestRow.make(
                    sample_id=sample_id,
                    path=path,
                    label=label,
                    speaker=speaker,
                    split=split,
                    source=source,
                    attack="-" if label == BONAFIDE else f"SYN{i % 3:02d}",
                )
            )

    manifest = Manifest(rows)
    manifest.assert_splits_speaker_disjoint()
    return manifest
