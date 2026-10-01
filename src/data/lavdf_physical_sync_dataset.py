"""LAV-DF dataset for PHYSICAL audio-visual synchronization training.

This is the corrected training-pair construction for the SyncGuard sync-detection
fix. It differs from `src.data.lavdf_dataset.LAVDFSyncDataset` /
`src.data.sync_pairs` (the original Phase 11 baseline, left unmodified and still
importable for old-vs-new comparison) in three required ways:

1. **Physical waveform shift, not token reassignment.** Negatives are built by
   calling `src.data.audio_shift.shift_waveform` on the clip's *raw waveform*
   before the mel spectrogram / audio encoder ever see it - the same operation
   used to build `demo/video/desync/*` (see `docs/windowed_av_inference.md`
   Section 11). `LAVDFSyncDataset` instead leaves the waveform/mel/audio-tokens
   untouched and re-buckets already-encoded tokens onto a shifted time grid
   (`src.data.sync_pairs.compute_shifted_alignment`) - a token-space operation
   the frozen audio encoder never actually sees applied to its input.

2. **Consecutive 32-frame production windows, not sparse whole-clip sampling.**
   `LAVDFSyncDataset._select_frames` densely subsamples 32 tokens via
   `np.linspace` across the *entire* clip, then aligns only the clip's *first*
   `32/fps` seconds of audio to those scattered tokens (see
   `src/inference/windowing.py` module docstring for why that mismatches
   production). This dataset instead reuses
   `src.inference.windowing.generate_windows` / `select_frame_indices` - the
   exact functions production inference uses - to carve each clip into the same
   consecutive, fixed-size windows the deployed model actually sees.

3. **An explicit cross-clip negative category** (video from clip A + native,
   unshifted audio from a different clip B in the same split), tracked
   separately from physical-shift negatives via `negative_type`, per the "do not
   rely only on silence-based shifts" requirement. Physical-shift remains the
   primary negative category; cross-clip is additional.

Manifest loading is shared with `LAVDFSyncDataset` via
`src.data.lavdf_dataset.load_lavdf_manifest` (same split-isolation guarantees:
a dataset instance only ever contains rows for one manifest `split`, so no
cross-split window, shift, or cross-clip pairing is possible - every derived
sample of a given source clip stays in that clip's split).

LAV-DF manipulation labels (`label_name`, `modify_audio`, `modify_video`,
`fake_periods`) are loaded as metadata only, exactly as in `LAVDFSyncDataset`,
and are NEVER used to derive the sync/desync target here either.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.config import AudioConfig, VideoDataConfig
from src.data.audio_shift import shift_waveform
from src.data.lavdf_dataset import LAVDFSyncSample, crop_audio_window, load_lavdf_manifest
from src.features.mel_spectrogram import MelSpectrogramExtractor
from src.inference.windowing import WindowSpec, generate_windows, select_frame_indices
from src.preprocessing.audio import preprocess_audio
from src.preprocessing.landmarks import interpolate_invalid, normalize_landmarks, region_indices

__all__ = [
    "PhysicalSyncPairConfig",
    "LAVDFPhysicalSyncConfig",
    "LAVDFPhysicalSyncDataset",
    "build_lavdf_physical_datasets",
    "physical_sync_collate_fn",
]

_DEFAULT_SHIFT_MAGNITUDES: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class PhysicalSyncPairConfig:
    """Configuration for physical-shift / cross-clip negative construction."""

    shift_magnitudes_seconds: tuple[float, ...] = _DEFAULT_SHIFT_MAGNITUDES
    negative_pair_probability: float = 0.5
    cross_clip_probability: float = 0.2  # fraction of NEGATIVES that are cross-clip
    window_frames: int = 32
    stride_frames: int = 32

    def __post_init__(self) -> None:
        if not self.shift_magnitudes_seconds or any(m <= 0 for m in self.shift_magnitudes_seconds):
            raise ValueError("shift_magnitudes_seconds must be a non-empty list of positive values")
        if not 0.0 <= self.negative_pair_probability <= 1.0:
            raise ValueError("negative_pair_probability must be in [0, 1]")
        if not 0.0 <= self.cross_clip_probability <= 1.0:
            raise ValueError("cross_clip_probability must be in [0, 1]")
        if self.window_frames <= 0 or self.stride_frames <= 0:
            raise ValueError("window_frames and stride_frames must be positive")


@dataclass(frozen=True)
class LAVDFPhysicalSyncConfig:
    manifest_path: str | Path
    video_dir: str | Path
    audio_dir: str | Path
    landmarks_dir: str | Path
    split: str = "train"
    pair_config: PhysicalSyncPairConfig = field(default_factory=PhysicalSyncPairConfig)
    seed: int = 0
    # Every LAV-DF clip yields several consecutive 32-frame windows (production
    # would evaluate all of them); capping how many of a single clip's windows
    # are used per training pass keeps wall-clock time and per-clip audio
    # decode cost bounded while still covering every clip at least once, chosen
    # deterministically (evenly spaced across the clip), not randomly dropped.
    max_windows_per_clip: int | None = None

    def __post_init__(self) -> None:
        if self.split not in ("train", "dev", "test"):
            raise ValueError(f"split must be one of ('train', 'dev', 'test'), got {self.split!r}")
        if self.max_windows_per_clip is not None and self.max_windows_per_clip <= 0:
            raise ValueError("max_windows_per_clip must be positive or None")


@dataclass(frozen=True)
class _WindowEntry:
    sample_idx: int
    window: WindowSpec
    fps: float
    n_frames: int


class LAVDFPhysicalSyncDataset(Dataset):
    """One item = one consecutive production-style window from one LAV-DF clip,
    labeled SYNC (native audio) or DESYNC (physically shifted, or cross-clip,
    audio), matching the exact temporal regime `SyncGuardPredictor` uses at
    inference time.
    """

    def __init__(self, config: LAVDFPhysicalSyncConfig, audio_cfg: AudioConfig, video_cfg: VideoDataConfig) -> None:
        self.config = config
        self.audio_cfg = audio_cfg
        self.video_cfg = video_cfg
        self.pair_cfg = config.pair_config

        self.samples: list[LAVDFSyncSample] = load_lavdf_manifest(
            config.manifest_path, config.video_dir, config.audio_dir, config.split
        )
        self._region = region_indices(video_cfg.regions)
        self._mel_extractor = MelSpectrogramExtractor(audio_cfg)

        self._entries: list[_WindowEntry] = self._build_window_index()
        if not self._entries:
            raise ValueError(f"No usable windows found for split={config.split!r}")

    # ------------------------------------------------------------------ index

    def _landmarks_path(self, sample: LAVDFSyncSample) -> Path:
        return Path(self.config.landmarks_dir) / f"{sample.sample_id}.npz"

    def _read_fps_and_frame_count(self, sample: LAVDFSyncSample) -> tuple[float, int]:
        path = self._landmarks_path(sample)
        if not path.exists():
            raise FileNotFoundError(f"Landmarks file not found: {path}")
        with np.load(path) as data:
            fps = float(np.asarray(data["fps"]))
            # Prefer the smallest array on disk that still carries frame count.
            if "frame_idx" in data.files:
                n = int(data["frame_idx"].shape[0])
            elif "valid" in data.files:
                n = int(data["valid"].shape[0])
            else:
                n = int(data["points"].shape[0])
        return fps, n

    def _build_window_index(self) -> list[_WindowEntry]:
        entries: list[_WindowEntry] = []
        cap = self.config.max_windows_per_clip
        for idx, sample in enumerate(self.samples):
            fps, n_frames = self._read_fps_and_frame_count(sample)
            windows = generate_windows(
                n_frames, fps, window_frames=self.pair_cfg.window_frames, stride_frames=self.pair_cfg.stride_frames
            )
            if cap is not None and len(windows) > cap:
                keep = np.linspace(0, len(windows) - 1, cap).round().astype(np.int64)
                windows = [windows[i] for i in sorted(set(keep.tolist()))]
            for w in windows:
                entries.append(_WindowEntry(sample_idx=idx, window=w, fps=fps, n_frames=n_frames))
        return entries

    def __len__(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------ data

    def _load_landmarks_window(self, sample: LAVDFSyncSample, window: WindowSpec) -> Tensor:
        path = self._landmarks_path(sample)
        with np.load(path) as data:
            points = np.asarray(data["points"], dtype=np.float32)
            valid = np.asarray(data["valid"], dtype=bool)

        points = interpolate_invalid(points, valid)
        points = normalize_landmarks(
            points, method=self.video_cfg.normalize, align_rotation=self.video_cfg.align_rotation, coords=3
        )
        points = points[:, self._region, :]

        sel = select_frame_indices(window.n_valid_frames, window.window_frames) + window.start_frame
        return torch.from_numpy(points[sel]).float()

    def _load_full_waveform(self, sample: LAVDFSyncSample) -> Tensor:
        if not sample.audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {sample.audio_path}")
        return preprocess_audio(sample.audio_path, self.audio_cfg, device="cpu")

    def _feasible_shift(
        self, rng: random.Random, sample: LAVDFSyncSample, window: WindowSpec
    ) -> float:
        """Pick a signed shift magnitude from the config's candidate list.

        A shift that leaves *some* real (non-silence) audio inside the
        analysis window is a valid, informative negative even if part of the
        window is boundary silence - that is exactly the physical construction
        `demo/video/desync/*` uses. The only case this deterministically avoids
        is a shift large enough to make the *entire* window fall in the
        silence-padded region (a degenerate, all-zero-audio sample), per the
        "skip too-short shifts deterministically" requirement: it falls back to
        the next-smaller candidate magnitude in that case.
        """
        candidates = sorted(self.pair_cfg.shift_magnitudes_seconds)
        sign = rng.choice((-1.0, 1.0))
        magnitude = rng.choice(candidates)

        def _degenerate(mag: float) -> bool:
            if sign > 0:  # delay: window is fully silent iff it ends before revealed content starts
                return mag >= window.end_time
            # advance: window is fully silent iff no original content remains at/after start_time
            return mag > sample.duration - window.start_time

        if _degenerate(magnitude):
            for mag in candidates:  # ascending: smallest first
                if not _degenerate(mag):
                    magnitude = mag
                    break
            # else: even the smallest magnitude is degenerate (a very short clip) -
            # keep it; an all-silence window is still a valid, if extreme, negative.
        return sign * magnitude

    def build_fixed(self, index: int, *, shift_seconds: float = 0.0) -> dict:
        """Deterministically build the window at `index` under an EXPLICIT,
        caller-chosen shift, bypassing the random negative-type/shift selection
        `__getitem__` uses for training. Used by
        `scripts/evaluate_physical_sync.py` to evaluate every shift category
        (native + each +/- magnitude) against the exact same underlying
        (clip, window) set, for a clean per-category comparison.
        """
        entry = self._entries[index]
        sample = self.samples[entry.sample_idx]
        window = entry.window
        window_seconds = window.window_frames / entry.fps

        landmarks = self._load_landmarks_window(sample, window)
        waveform = self._load_full_waveform(sample)
        if shift_seconds != 0.0:
            waveform = shift_waveform(waveform, self.audio_cfg.sample_rate, shift_seconds)
        audio_window = crop_audio_window(
            waveform, self.audio_cfg.sample_rate, start_seconds=window.start_time, window_seconds=window_seconds
        )
        with torch.no_grad():
            mel_window = self._mel_extractor(audio_window)

        return {
            "mel_window": mel_window,
            "landmarks": landmarks,
            "fps": entry.fps,
            "window_seconds": window_seconds,
            "label": 1.0 if shift_seconds == 0.0 else 0.0,
            "negative_type": "none" if shift_seconds == 0.0 else "physical_shift",
            "shift_seconds": shift_seconds,
            "sample_id": sample.sample_id,
            "partner_sample_id": "",
            "window_index": window.index,
            "split": sample.split,
        }

    def __getitem__(self, index: int) -> dict:
        entry = self._entries[index]
        sample = self.samples[entry.sample_idx]
        window = entry.window
        window_seconds = window.window_frames / entry.fps

        rng = random.Random(self.config.seed + index)
        is_negative = rng.random() < self.pair_cfg.negative_pair_probability
        negative_type = "none"
        shift_seconds = 0.0
        partner_sample_id = ""

        landmarks = self._load_landmarks_window(sample, window)

        if not is_negative:
            waveform = self._load_full_waveform(sample)
            audio_window = crop_audio_window(
                waveform, self.audio_cfg.sample_rate, start_seconds=window.start_time, window_seconds=window_seconds
            )
        elif rng.random() < self.pair_cfg.cross_clip_probability and len(self.samples) > 1:
            negative_type = "cross_clip"
            partner_idx = rng.randrange(len(self.samples) - 1)
            if partner_idx >= entry.sample_idx:
                partner_idx += 1
            partner_sample = self.samples[partner_idx]
            partner_sample_id = partner_sample.sample_id
            partner_fps, partner_n_frames = self._read_fps_and_frame_count(partner_sample)
            partner_windows = generate_windows(
                partner_n_frames, partner_fps,
                window_frames=self.pair_cfg.window_frames, stride_frames=self.pair_cfg.stride_frames,
            )
            partner_window = rng.choice(partner_windows)
            partner_waveform = self._load_full_waveform(partner_sample)
            audio_window = crop_audio_window(
                partner_waveform, self.audio_cfg.sample_rate,
                start_seconds=partner_window.start_time,
                window_seconds=partner_window.window_frames / partner_fps,
            )
        else:
            negative_type = "physical_shift"
            shift_seconds = self._feasible_shift(rng, sample, window)
            waveform = self._load_full_waveform(sample)
            shifted = shift_waveform(waveform, self.audio_cfg.sample_rate, shift_seconds)
            audio_window = crop_audio_window(
                shifted, self.audio_cfg.sample_rate, start_seconds=window.start_time, window_seconds=window_seconds
            )

        with torch.no_grad():
            mel_window = self._mel_extractor(audio_window)

        return {
            "mel_window": mel_window,
            "landmarks": landmarks,
            "fps": entry.fps,
            "window_seconds": window_seconds,
            "label": 0.0 if is_negative else 1.0,
            "negative_type": negative_type,
            "shift_seconds": shift_seconds,
            "sample_id": sample.sample_id,
            "partner_sample_id": partner_sample_id,
            "window_index": window.index,
            "split": sample.split,
        }


def physical_sync_collate_fn(batch: list[dict]) -> dict:
    """Pad variable-length mel windows (clip fps varies -> window duration in
    samples varies slightly); landmarks are always exactly `window_frames`
    long so they stack directly. Module-scope for Windows `spawn` picklability.
    """
    max_mel_len = max(item["mel_window"].shape[1] for item in batch)
    mel_windows = []
    for item in batch:
        mel = item["mel_window"]
        if mel.shape[1] < max_mel_len:
            mel = torch.nn.functional.pad(mel, (0, max_mel_len - mel.shape[1]))
        mel_windows.append(mel)

    return {
        "mel_window": torch.stack(mel_windows),
        "landmarks": torch.stack([item["landmarks"] for item in batch]),
        "fps": [item["fps"] for item in batch],
        "window_seconds": [item["window_seconds"] for item in batch],
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.float32),
        "negative_type": [item["negative_type"] for item in batch],
        "shift_seconds": [item["shift_seconds"] for item in batch],
        "sample_id": [item["sample_id"] for item in batch],
        "partner_sample_id": [item["partner_sample_id"] for item in batch],
        "window_index": [item["window_index"] for item in batch],
        "split": [item["split"] for item in batch],
    }


def build_lavdf_physical_datasets(
    manifest_path: str | Path,
    video_dir: str | Path,
    audio_dir: str | Path,
    landmarks_dir: str | Path,
    audio_cfg: AudioConfig,
    video_cfg: VideoDataConfig,
    *,
    splits: tuple[str, ...] = ("train", "dev"),
    pair_config: PhysicalSyncPairConfig | None = None,
    seed: int = 0,
    max_windows_per_clip: int | None = None,
) -> dict[str, LAVDFPhysicalSyncDataset]:
    """Build `LAVDFPhysicalSyncDataset`s for the requested manifest splits.

    Each returned dataset only ever contains windows/shifts/cross-clip partners
    drawn from its own `split`'s manifest rows (see `load_lavdf_manifest`), so
    train/dev/test isolation is preserved automatically - no derived sample of a
    train clip can appear in the dev dataset or vice versa.
    """
    pair_config = pair_config or PhysicalSyncPairConfig()
    datasets: dict[str, LAVDFPhysicalSyncDataset] = {}
    for split in splits:
        config = LAVDFPhysicalSyncConfig(
            manifest_path=manifest_path,
            video_dir=video_dir,
            audio_dir=audio_dir,
            landmarks_dir=landmarks_dir,
            split=split,
            pair_config=pair_config,
            seed=seed,
            max_windows_per_clip=max_windows_per_clip,
        )
        datasets[split] = LAVDFPhysicalSyncDataset(config, audio_cfg, video_cfg)
    return datasets
