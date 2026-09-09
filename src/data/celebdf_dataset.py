"""Celeb-DF-v2 protocol parsing and the landmark ``Dataset`` (Phase 7).

The raw dataset is not bundled; :data:`DOWNLOAD_HINT` explains how to obtain it.
Once ``data/celebdf/`` holds the three video folders and
``List_of_testing_videos.txt``, :func:`build_celebdf_manifest` converts the
official split into a :class:`~src.data.manifests.Manifest`, and
:class:`CelebDFLandmarkDataset` turns that manifest into ``(features, label)``
pairs where ``features`` is the normalised MediaPipe face/mouth landmark sequence
for one video (see :mod:`src.preprocessing.landmarks`).

Label convention matches the audio branch's ``1 = bonafide``: here ``1 = real``
(Celeb-real + YouTube-real), ``0 = fake`` (Celeb-synthesis face swaps).

Split protocol
--------------
* ``eval``  - the 518 videos listed in ``List_of_testing_videos.txt``, verbatim.
* ``train`` / ``dev`` - every other video, partitioned **identity-disjoint**: the
  celebrity identity pool ``id0..id61`` (plus one synthetic identity per
  YouTube-real clip) is split by :attr:`VideoDataConfig.dev_identity_frac`, and a
  video goes to whichever split owns its identity. A Celeb-synthesis clip
  ``id{src}_id{tgt}_*`` is keyed by its **target** identity ``id{tgt}`` (the
  person whose head/pose the video shows); the residual source-identity leakage
  is documented in ``docs/decisions/0002-video-deepfake-detection.md``.
Celeb-DF reuses the same identities in its test list, so ``train`` and ``eval``
unavoidably share identities - that limitation is recorded in the decision note.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import VideoDataConfig
from src.data.manifests import BONAFIDE, SPOOF, LeakageError, Manifest, ManifestRow
from src.preprocessing.landmarks import (
    N_FACEMESH_POINTS,
    interpolate_invalid,
    normalize_landmarks,
    region_indices,
)

__all__ = [
    "DOWNLOAD_HINT",
    "SOURCE_TAG",
    "REAL_TAG",
    "FAKE_ATTACK",
    "VIDEO_DIRS",
    "parse_testing_list",
    "video_sample_id",
    "video_identity",
    "video_identities",
    "identities_of_sample",
    "build_celebdf_manifest",
    "synthetic_landmark_manifest",
    "CelebDFLandmarkDataset",
    "build_celebdf_datasets",
]

SOURCE_TAG = "celeb-df-v2"
REAL_TAG = "-"
FAKE_ATTACK = "celebdf-fs"          # single face-swap synthesis method
TESTING_LIST = "List_of_testing_videos.txt"
VIDEO_DIRS = ("Celeb-real", "Celeb-synthesis", "YouTube-real")
_REAL_DIRS = ("Celeb-real", "YouTube-real")

DOWNLOAD_HINT = (
    "Celeb-DF-v2 was not found. Request it from the authors "
    "(https://github.com/yuezunli/celeb-deepfakeforensics), extract so that "
    "<root> contains 'Celeb-real/', 'Celeb-synthesis/', 'YouTube-real/' and "
    "'List_of_testing_videos.txt'."
)


# --------------------------------------------------------------------- protocol

def _rel_posix(path: str) -> str:
    return path.replace("\\", "/").strip().lstrip("./")


def video_sample_id(rel_path: str) -> str:
    """``"Celeb-synthesis/id0_id1_0003.mp4"`` -> ``"Celeb-synthesis__id0_id1_0003"``."""

    rel = _rel_posix(rel_path)
    folder, _, name = rel.rpartition("/")
    stem = name[:-4] if name.lower().endswith(".mp4") else name
    return f"{folder}__{stem}" if folder else stem


def video_identities(rel_path: str) -> set[str]:
    """*Every* identity a video involves.

    ``Celeb-real/id3_0009.mp4`` -> ``{id3}``; ``YouTube-real/00170.mp4`` ->
    ``{yt-00170}``; ``Celeb-synthesis/id0_id1_0003.mp4`` -> ``{id0, id1}`` -
    both the source identity (whose face is rendered in) and the target identity
    (whose head/pose the clip shows). The strict train/dev partition keys on this
    full set so a source identity can never straddle the split.
    """

    rel = _rel_posix(rel_path)
    folder, _, name = rel.rpartition("/")
    stem = name[:-4] if name.lower().endswith(".mp4") else name
    if folder == "YouTube-real":
        return {f"yt-{stem}"}
    ids = [p for p in stem.split("_") if p.startswith("id")]
    if not ids:
        raise ValueError(f"cannot parse identity from {rel_path!r}")
    return set(ids)


def video_identity(rel_path: str) -> str:
    """The single identity a video is *keyed* by (its ``speaker`` field).

    The *target* identity for a Celeb-synthesis clip (last ``id`` token), the sole
    identity otherwise. Retained for per-identity result slicing and CSV
    readability; disjointness is enforced on the full :func:`video_identities`
    set, not on this value.
    """

    rel = _rel_posix(rel_path)
    folder, _, name = rel.rpartition("/")
    stem = name[:-4] if name.lower().endswith(".mp4") else name
    if folder == "YouTube-real":
        return f"yt-{stem}"
    ids = [p for p in stem.split("_") if p.startswith("id")]
    if not ids:
        raise ValueError(f"cannot parse identity from {rel_path!r}")
    return ids[-1]


def parse_testing_list(path: str | Path) -> list[dict[str, object]]:
    """Parse ``List_of_testing_videos.txt`` -> ``[{rel_path, label}]`` (label 1=real)."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"testing list not found: {path}")
    entries: list[dict[str, object]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2 or parts[0] not in ("0", "1"):
            raise ValueError(f"{path}:{lineno}: expected '<0|1> <path>', got {line!r}")
        entries.append({"rel_path": _rel_posix(parts[1]), "label": int(parts[0])})
    if not entries:
        raise ValueError(f"{path}: no entries parsed")
    return entries


def _iter_videos(root: Path) -> list[str]:
    rels: list[str] = []
    for folder in VIDEO_DIRS:
        d = root / folder
        if not d.is_dir():
            raise FileNotFoundError(f"{DOWNLOAD_HINT}\n(missing {d})")
        rels += sorted(f"{folder}/{p.name}" for p in d.glob("*.mp4"))
    return rels


def _label_for(rel_path: str) -> int:
    return SPOOF if rel_path.startswith("Celeb-synthesis/") else BONAFIDE


def _identity_components(pool_rel: list[str]) -> list[set[str]]:
    """Connected components of the identity co-occurrence graph.

    Two identities share an edge when they appear together in some pool video
    (a Celeb-synthesis clip links its source and target identity). A component is
    atomic for the train/dev split: putting any of its identities in one split
    forces the whole component there, so no identity - source identities
    included - can appear in both splits.
    """

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for rel in pool_rel:
        ids = sorted(video_identities(rel))
        find(ids[0])                       # register singletons too
        for other in ids[1:]:
            union(ids[0], other)

    groups: dict[str, set[str]] = {}
    for ident in list(parent):
        groups.setdefault(find(ident), set()).add(ident)
    return list(groups.values())


def _count_valid_frames(landmarks_dir: Path, sample_id: str) -> int:
    """Detected-face frame count for a cached video (``-1`` if no cache present)."""

    npz = landmarks_dir / f"{sample_id}.npz"
    if npz.is_file():
        with np.load(npz) as data:
            return int(np.asarray(data["valid"], dtype=bool).sum())
    meta = landmarks_dir / f"{sample_id}.meta.json"
    if meta.is_file():
        return int(json.loads(meta.read_text(encoding="utf-8")).get("n_detected", 0))
    return -1


def build_celebdf_manifest(
    root: str | Path,
    *,
    out_csv: str | Path | None = None,
    dev_identity_frac: float = 0.15,
    seed: int = 1337,
    landmarks_dir: str | Path | None = None,
    min_valid_frames: int | None = None,
    return_report: bool = False,
) -> "Manifest | tuple[Manifest, dict]":
    """Build the train/dev/eval manifest for Celeb-DF-v2.

    Split protocol
    --------------
    * ``eval`` = ``List_of_testing_videos.txt`` verbatim (the file is only read).
    * ``train`` / ``dev`` = every other video, partitioned **strictly
      identity-disjoint**: identities are grouped into connected components by
      co-occurrence (:func:`_identity_components` - a synthesis clip links its
      source and target identity), whole components are assigned to ``dev`` until
      ``dev_identity_frac`` of the pool identities is reached, and each video
      follows its component. No identity - source identities of synthesised clips
      included - can appear in both ``train`` and ``dev``. The choice is
      deterministic in ``seed``.

    Landmark filtering
    ------------------
    When ``min_valid_frames`` is set, ``landmarks_dir`` is required and any video
    with fewer than that many detected-face frames (or no cache at all) is
    **excluded from every split** rather than interpolated in. Occasional gaps in
    otherwise-valid videos are still interpolated at load time. With
    ``return_report=True`` the call returns ``(manifest, report)`` where
    ``report`` carries the exclusion counts / reasons / ids and the final
    identity sets per split.
    """

    root = Path(root)
    lm_dir = Path(landmarks_dir) if landmarks_dir is not None else None
    if min_valid_frames is not None and lm_dir is None:
        raise ValueError("min_valid_frames requires landmarks_dir")

    test_entries = parse_testing_list(root / TESTING_LIST)
    eval_rel = [e["rel_path"] for e in test_entries]
    eval_set = set(eval_rel)

    all_rel = _iter_videos(root)
    missing_from_disk = sorted(eval_set - set(all_rel))
    if missing_from_disk:
        raise FileNotFoundError(
            f"{len(missing_from_disk)} testing-list videos are not on disk, e.g. "
            f"{missing_from_disk[:5]}"
        )
    pool_rel = [r for r in all_rel if r not in eval_set]

    # -- strict identity-disjoint component partition (train vs dev) ----------
    components = _identity_components(pool_rel)
    n_pool_ids = sum(len(c) for c in components)
    target = round(dev_identity_frac * n_pool_ids)

    def has_fake(comp: set[str]) -> bool:
        return any(_label_for(r) == SPOOF and video_identities(r) <= comp for r in pool_rel)

    rng = random.Random(seed)
    fake_comps = [c for c in components if has_fake(c)]
    real_comps = [c for c in components if not has_fake(c)]
    rng.shuffle(real_comps)
    # smallest fake-bearing component seeds dev (keeps dev small); ties broken
    # deterministically by the seeded RNG.
    fake_comps.sort(key=lambda c: (len(c), rng.random()))
    if len(fake_comps) < 2:
        raise ValueError(
            f"pool has {len(fake_comps)} fake-bearing identity component(s); need >=2 "
            f"so that both train and dev can hold fakes - adjust the data or seed"
        )

    # dev = one fake component + real-only components up to the identity target.
    # Further fake components are NOT added (each is large; one is enough for dev).
    dev_ids: set[str] = set(fake_comps[0])
    for comp in real_comps:
        if len(dev_ids) >= target:
            break
        dev_ids |= comp

    # -- landmark cache checks --------------------------------------------
    if lm_dir is not None and min_valid_frames is None:
        all_rel_kept = pool_rel + eval_rel
        absent = [video_sample_id(r) for r in all_rel_kept
                  if not (lm_dir / f"{video_sample_id(r)}.npz").is_file()]
        if absent:
            raise FileNotFoundError(
                f"{len(absent)} videos have no landmark .npz under {lm_dir}, e.g. "
                f"{absent[:5]} - run scripts/extract_celebdf_landmarks.py first "
                f"(or pass min_valid_frames to exclude them instead)"
            )

    # -- optional landmark-quality exclusion --------------------------------
    excluded: list[dict[str, object]] = []

    def keep(rel_path: str) -> bool:
        if min_valid_frames is None:
            return True
        sid = video_sample_id(rel_path)
        n_valid = _count_valid_frames(lm_dir, sid)
        if n_valid < min_valid_frames:
            excluded.append({
                "sample_id": sid,
                "rel_path": rel_path,
                "split": "eval" if rel_path in eval_set else (
                    "dev" if video_identities(rel_path) <= dev_ids else "train"),
                "n_valid": n_valid,
                "reason": "missing_landmarks" if n_valid < 0 else "insufficient_valid_frames",
            })
            return False
        return True

    rows: list[ManifestRow] = []

    def add(rel_path: str, split: str) -> None:
        label = _label_for(rel_path)
        rows.append(ManifestRow.make(
            sample_id=video_sample_id(rel_path),
            path=str(root / rel_path),
            label=label,
            speaker=video_identity(rel_path),
            split=split,
            source=SOURCE_TAG,
            attack=REAL_TAG if label == BONAFIDE else FAKE_ATTACK,
        ))

    for rel_path in pool_rel:
        if keep(rel_path):
            add(rel_path, "dev" if video_identities(rel_path) <= dev_ids else "train")
    for rel_path in eval_rel:                       # official order preserved
        if keep(rel_path):
            add(rel_path, "eval")

    manifest = Manifest(rows)

    # -- disjointness guarantees ------------------------------------------
    train_ids = {i for r in manifest.split("train") for i in identities_of_sample(r.sample_id)}
    dev_ids_final = {i for r in manifest.split("dev") for i in identities_of_sample(r.sample_id)}
    overlap = train_ids & dev_ids_final
    if overlap:
        raise LeakageError(
            f"{len(overlap)} identit(y/ies) appear in both train and dev "
            f"(e.g. {sorted(overlap)[:5]}) - strict partition failed"
        )
    for split in ("train", "dev"):
        labels = {r.label for r in manifest.split(split)}
        if labels != {BONAFIDE, SPOOF}:
            raise ValueError(
                f"'{split}' split has labels {labels}, need both real and fake - "
                f"adjust dev_identity_frac or seed"
            )

    report = {
        "n_pool_identities": n_pool_ids,
        "n_identity_components": len(components),
        "component_sizes": sorted((len(c) for c in components), reverse=True),
        "dev_identity_target": target,
        "train_identities": len(train_ids),
        "dev_identities": len(dev_ids_final),
        "train_dev_identity_overlap": len(overlap),
        "split_sizes": manifest.split_sizes(),
        "min_valid_frames": min_valid_frames,
        "n_excluded": len(excluded),
        "excluded_by_split": _count_by(excluded, "split"),
        "excluded_by_reason": _count_by(excluded, "reason"),
        "excluded": excluded,
    }

    if out_csv is not None:
        manifest.write_csv(out_csv)
    return (manifest, report) if return_report else manifest


def identities_of_sample(sample_id: str) -> set[str]:
    """Identity set from a manifest ``sample_id`` (``"<folder>__<stem>"``)."""

    folder, _, stem = sample_id.partition("__")
    return video_identities(f"{folder}/{stem}.mp4")


def _count_by(items: list[dict[str, object]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for it in items:
        out[str(it[key])] = out.get(str(it[key]), 0) + 1
    return out


# --------------------------------------------------------------------- synthetic

def synthetic_landmark_manifest(
    out_dir: str | Path,
    *,
    n_per_split: dict[str, int] | int = 24,
    n_ids_per_split: int = 4,
    num_frames: int = 8,
    seed: int = 0,
) -> Manifest:
    """Write random ``[T, 478, 3]`` landmark ``.npz`` files with speaker-disjoint
    splits so the visual pipeline can be exercised with no dataset and no
    MediaPipe. Mirrors :func:`src.data.manifests.synthetic_manifest`.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(n_per_split, int):
        n_per_split = {"train": n_per_split, "dev": n_per_split, "eval": n_per_split}

    rng = np.random.default_rng(seed)
    rows: list[ManifestRow] = []
    for split, count in n_per_split.items():
        speakers = [f"SYN-{split}-{k:02d}" for k in range(n_ids_per_split)]
        for i in range(count):
            label = BONAFIDE if i % 2 == 0 else SPOOF
            sample_id = f"SYN-{split}-{i:04d}"
            # Fake class gets slightly noisier landmarks so the two are separable.
            noise = 0.02 if label == BONAFIDE else 0.05
            base = rng.normal(0.0, 0.3, size=(N_FACEMESH_POINTS, 3)).astype(np.float32)
            pts = base[None] + rng.normal(0.0, noise, size=(num_frames, N_FACEMESH_POINTS, 3))
            np.savez_compressed(
                out_dir / f"{sample_id}.npz",
                points=pts.astype(np.float32),
                valid=np.ones(num_frames, dtype=bool),
                fps=np.float32(25.0),
                frame_idx=np.arange(num_frames, dtype=np.int64),
            )
            rows.append(
                ManifestRow.make(
                    sample_id=sample_id,
                    path=str(out_dir / f"{sample_id}.npz"),
                    label=label,
                    speaker=speakers[i % len(speakers)],
                    split=split,
                    source="synthetic",
                    attack=REAL_TAG if label == BONAFIDE else FAKE_ATTACK,
                )
            )
    manifest = Manifest(rows)
    manifest.assert_splits_speaker_disjoint()
    return manifest


# ----------------------------------------------------------------------- dataset

def _select_frames(n_available: int, n_want: int, *, rng: random.Random | None) -> np.ndarray:
    if n_available <= 0:
        raise ValueError("landmark sequence has no frames")
    if rng is not None:
        if n_available >= n_want:
            return np.array(sorted(rng.sample(range(n_available), n_want)), dtype=np.int64)
        return np.array(sorted(rng.choices(range(n_available), k=n_want)), dtype=np.int64)
    return np.linspace(0, n_available - 1, n_want).round().astype(np.int64)


class CelebDFLandmarkDataset(Dataset):
    """Manifest-backed landmark dataset yielding ``(features[T, N, C], label)``."""

    def __init__(
        self,
        manifest: Manifest,
        video_cfg: VideoDataConfig,
        *,
        landmarks_dir: str | Path | None = None,
        num_frames: int | None = None,
        coords: int = 3,
        random_sample: bool = False,
        seed: int = 0,
        frame_transform=None,
    ) -> None:
        if len(manifest) == 0:
            raise ValueError("manifest is empty")
        self.manifest = manifest
        self.video_cfg = video_cfg
        self.landmarks_dir = Path(landmarks_dir or video_cfg.landmarks_dir)
        self.num_frames = int(num_frames or video_cfg.num_frames)
        self.coords = int(coords)
        self.random_sample = random_sample
        self._seed = seed
        self.frame_transform = frame_transform
        self._region = region_indices(video_cfg.regions)

    def __len__(self) -> int:
        return len(self.manifest)

    def row(self, index: int) -> ManifestRow:
        return self.manifest[index]

    def timing(self, index: int) -> dict[str, float | int]:
        """Per-clip temporal metadata for Phase 9 audio-visual alignment.

        Returns ``{"fps": float, "frame_idx0": int, "window_seconds": float}``:

        * ``fps`` - the source video frame rate (from the cached ``.npz``);
        * ``frame_idx0`` - source frame index of the first frame the model sees;
        * ``window_seconds`` - wall-clock length of the model's frame window,
          ``num_frames / fps``. Exact for the dense contiguous cache
          (``num_frames == cached frames``); an approximation if a shorter
          ``num_frames`` re-subsamples a longer cache.

        Non-breaking: ``__getitem__`` still returns ``(features, label)``.
        """

        row = self.manifest[index]
        npz_path = self.landmarks_dir / f"{row.sample_id}.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(f"sample {row.sample_id!r}: no landmarks at {npz_path}")
        with np.load(npz_path) as data:
            fps = float(np.asarray(data["fps"]))
            frame_idx = np.asarray(data["frame_idx"], dtype=np.int64)

        sel = _select_frames(int(frame_idx.size), self.num_frames, rng=None)
        return {
            "fps": fps,
            "frame_idx0": int(frame_idx[sel[0]]),
            "window_seconds": float(self.num_frames) / fps,
        }

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.manifest[index]
        npz_path = self.landmarks_dir / f"{row.sample_id}.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(f"sample {row.sample_id!r}: no landmarks at {npz_path}")

        with np.load(npz_path) as data:
            points = np.asarray(data["points"], dtype=np.float32)   # [Tall, 478, 3]
            valid = np.asarray(data["valid"], dtype=bool)

        points = interpolate_invalid(points, valid)

        rng = random.Random(self._seed * 1_000_003 + index) if self.random_sample else None
        idx = _select_frames(points.shape[0], self.num_frames, rng=rng)
        points = points[idx]                                        # [T, 478, 3]

        points = normalize_landmarks(
            points,
            method=self.video_cfg.normalize,
            align_rotation=self.video_cfg.align_rotation,
            coords=self.coords,
        )
        if self.frame_transform is not None:
            points = self.frame_transform(points)

        feat = np.ascontiguousarray(points[:, self._region, :])     # [T, N, C]
        return torch.from_numpy(feat).float(), int(row.label)

    def label_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights ``[w_fake, w_real]`` for a weighted loss."""

        counts = torch.zeros(2)
        for r in self.manifest:
            counts[r.label] += 1
        counts = counts.clamp(min=1.0)
        return counts.sum() / (2.0 * counts)


def build_celebdf_datasets(
    manifest: Manifest | str | Path,
    video_cfg: VideoDataConfig,
    *,
    splits: tuple[str, ...] = ("train", "dev", "eval"),
    coords: int = 3,
    seed: int = 0,
    landmarks_dir: str | Path | None = None,
) -> dict[str, CelebDFLandmarkDataset]:
    """Wrap each split in a :class:`CelebDFLandmarkDataset` (random frame sampling
    for ``train`` only, deterministic evenly-spaced sampling elsewhere)."""

    if not isinstance(manifest, Manifest):
        manifest = Manifest.read_csv(manifest)

    present = {r.split for r in manifest}
    wanted = [s for s in splits if s in present]
    if not wanted:
        raise ValueError(f"manifest has splits {sorted(present)}, none of {splits}")

    out: dict[str, CelebDFLandmarkDataset] = {}
    for split in wanted:
        out[split] = CelebDFLandmarkDataset(
            manifest.split(split),
            video_cfg,
            landmarks_dir=landmarks_dir,
            coords=coords,
            random_sample=(split == "train"),
            seed=seed,
        )
    return out
