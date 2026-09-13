"""LAV-DF dataset loader for audio-visual synchronization (Phase 11).

Loads extracted LAV-DF clips and produces training-ready inputs for the existing
AVEncoder: log-mel spectrograms, normalized landmark sequences, per-clip timing
metadata, and sync-pair shift labels.

IMPORTANT: This dataset loader does NOT use LAV-DF fake_periods as sync labels.
Manipulation is not equivalent to temporal desynchronization.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.config import AudioConfig, VideoDataConfig
from src.data.sync_pairs import SyncPairConfig, parse_fake_periods, sample_shift_seconds
from src.features.mel_spectrogram import MelSpectrogramExtractor
from src.preprocessing.audio import load_audio, preprocess_audio
from src.preprocessing.landmarks import interpolate_invalid, normalize_landmarks, region_indices

__all__ = [
    "LAVDFSyncConfig",
    "LAVDFSyncSample",
    "LAVDFSyncDataset",
    "build_lavdf_datasets",
    "crop_audio_window",
    "compute_window_seconds",
    "lavdf_collate_fn",
]


def compute_window_seconds(n_video_tokens: int, fps: float) -> float:
    """Wall-clock duration of the model's dense video-token window."""
    if n_video_tokens <= 0:
        raise ValueError("n_video_tokens must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    return float(n_video_tokens) / float(fps)


def crop_audio_window(
    waveform: Tensor,
    sample_rate: int,
    start_seconds: float,
    window_seconds: float,
) -> Tensor:
    """Crop ``waveform`` ``[C, N]`` to ``[start_seconds, start_seconds + window_seconds)``.

    Right-pads with zeros when the window extends past the clip end.
    """
    if waveform.ndim != 2:
        raise ValueError(f"waveform must be [C, N], got {tuple(waveform.shape)}")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if start_seconds < 0:
        raise ValueError("start_seconds must be non-negative")
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")

    start = int(round(start_seconds * sample_rate))
    end = start + int(round(window_seconds * sample_rate))
    n = waveform.shape[-1]

    if start >= n:
        return waveform.new_zeros(waveform.shape[0], end - start)

    chunk = waveform[..., start:min(end, n)]
    if chunk.shape[-1] < end - start:
        pad = waveform.new_zeros(waveform.shape[0], (end - start) - chunk.shape[-1])
        chunk = torch.cat([chunk, pad], dim=-1)
    return chunk


@dataclass(frozen=True)
class LAVDFSyncConfig:
    """Configuration for LAV-DF sync dataset."""

    manifest_path: str | Path
    video_dir: str | Path
    audio_dir: str | Path
    landmarks_dir: str | Path
    n_video_tokens: int = 32
    sync_pair_config: SyncPairConfig | None = None
    use_negative_pairs: bool = True
    negative_pair_probability: float = 0.5
    split: str = "train"  # train, dev, or test
    random_sample: bool = False  # random frame sampling for training
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_video_tokens <= 0:
            raise ValueError("n_video_tokens must be positive")
        if not 0 <= self.negative_pair_probability <= 1:
            raise ValueError("negative_pair_probability must be between 0 and 1")
        if self.split not in ("train", "dev", "test"):
            raise ValueError(f"split must be one of ('train', 'dev', 'test'), got {self.split!r}")
        if self.use_negative_pairs and self.sync_pair_config is None:
            object.__setattr__(self, "sync_pair_config", SyncPairConfig())


@dataclass
class LAVDFSyncSample:
    """A single LAV-DF sample with metadata."""

    sample_id: str
    video_path: Path
    audio_path: Path
    label: int  # 1 = real, 0 = manipulated
    label_name: str
    split: str
    modify_audio: bool
    modify_video: bool
    n_fakes: int
    fake_periods: list[list[float]]  # Parsed but NOT used as sync labels
    duration: float
    original: str
    video_frames: int
    audio_frames: int


class LAVDFSyncDataset(Dataset):
    """Dataset for LAV-DF audio-visual synchronization.

    Returns log-mel spectrograms and normalized landmarks aligned to a deterministic
    temporal window, plus per-clip FPS and sync-shift metadata. Enforces split
    isolation to prevent train/dev leakage.
    """

    def __init__(
        self,
        config: LAVDFSyncConfig,
        audio_cfg: AudioConfig,
        video_cfg: VideoDataConfig,
    ) -> None:
        self.config = config
        self.audio_cfg = audio_cfg
        self.video_cfg = video_cfg

        self.samples = self._load_manifest()
        self._region = region_indices(video_cfg.regions)
        self._mel_extractor = MelSpectrogramExtractor(audio_cfg)
        self._rng = random.Random(config.seed)

        for sample in self.samples:
            if sample.split != config.split:
                raise ValueError(
                    f"Sample {sample.sample_id} has split '{sample.split}' "
                    f"but dataset config requires '{config.split}'"
                )

    def _load_manifest(self) -> list[LAVDFSyncSample]:
        """Load samples from manifest CSV, filtering by configured split."""
        manifest_path = Path(self.config.manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        samples: list[LAVDFSyncSample] = []
        with manifest_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] != self.config.split:
                    continue

                fake_periods = parse_fake_periods(row.get("fake_periods", "[]"))
                video_path = Path(self.config.video_dir) / Path(row["path"]).name
                audio_path = Path(self.config.audio_dir) / f"{Path(row['path']).stem}.wav"

                samples.append(
                    LAVDFSyncSample(
                        sample_id=row["sample_id"],
                        video_path=video_path,
                        audio_path=audio_path,
                        label=int(row["label"]),
                        label_name=row["label_name"],
                        split=row["split"],
                        modify_audio=row["modify_audio"] == "True",
                        modify_video=row["modify_video"] == "True",
                        n_fakes=int(row["n_fakes"]),
                        fake_periods=fake_periods,
                        duration=float(row["duration"]),
                        original=row.get("original", ""),
                        video_frames=int(row["video_frames"]),
                        audio_frames=int(row["audio_frames"]),
                    )
                )

        if not samples:
            raise ValueError(
                f"No samples found for split={self.config.split!r} in {manifest_path}"
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _select_frames(self, n_available: int, n_want: int) -> np.ndarray:
        """Select frame indices for temporal windowing."""
        if n_available <= 0:
            raise ValueError("landmark sequence has no frames")
        if n_available >= n_want:
            if self.config.random_sample:
                return np.array(
                    sorted(self._rng.sample(range(n_available), n_want)), dtype=np.int64
                )
            return np.linspace(0, n_available - 1, n_want).round().astype(np.int64)
        return np.array(
            sorted(self._rng.choices(range(n_available), k=n_want)), dtype=np.int64
        )

    def _sample_shift_seconds(self, index: int) -> float:
        """Sample a sync shift for this index (0.0 = positive pair)."""
        if not self.config.use_negative_pairs:
            return 0.0
        if self._rng.random() >= self.config.negative_pair_probability:
            return 0.0
        pair_cfg = self.config.sync_pair_config
        if pair_cfg is None:
            return 0.0
        # Per-index seed keeps sampling deterministic across __getitem__ calls
        rng = random.Random(self.config.seed + index)
        return sample_shift_seconds(pair_cfg, rng, include_positive=False)

    def timing(self, index: int) -> dict[str, float | int]:
        """Per-clip temporal metadata for Phase 9 audio-visual alignment.

        Returns ``fps``, ``frame_idx0``, and ``window_seconds`` for the model window.
        FPS is read from the landmark NPZ (not assumed to be 25).
        """
        sample = self.samples[index]
        landmarks_path = Path(self.config.landmarks_dir) / f"{sample.sample_id}.npz"
        if not landmarks_path.exists():
            raise FileNotFoundError(f"Landmarks file not found: {landmarks_path}")

        with np.load(landmarks_path) as data:
            fps = float(np.asarray(data["fps"]))
            if "frame_idx" in data:
                frame_idx = np.asarray(data["frame_idx"], dtype=np.int64)
            else:
                frame_idx = np.arange(int(data["points"].shape[0]), dtype=np.int64)

        sel = self._select_frames(int(frame_idx.size), self.config.n_video_tokens)
        window_seconds = compute_window_seconds(self.config.n_video_tokens, fps)
        return {
            "fps": fps,
            "frame_idx0": int(frame_idx[sel[0]]),
            "window_seconds": window_seconds,
        }

    def __getitem__(self, index: int) -> dict[str, Tensor | float | int | str | bool | list]:
        """Load and preprocess a single training sample."""
        sample = self.samples[index]
        timing = self.timing(index)
        fps = float(timing["fps"])
        frame_idx0 = int(timing["frame_idx0"])
        window_seconds = float(timing["window_seconds"])

        if not sample.audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {sample.audio_path}")

        # Full-clip audio preprocessing (Phase 2A), then crop to the video window
        audio_waveform = preprocess_audio(sample.audio_path, self.audio_cfg, device="cpu")
        audio_window = crop_audio_window(
            audio_waveform,
            self.audio_cfg.sample_rate,
            start_seconds=frame_idx0 / fps,
            window_seconds=window_seconds,
        )

        with torch.no_grad():
            mel_window = self._mel_extractor(audio_window)  # [n_mels, T_mel]

        # Landmarks for the same temporal window
        landmarks_path = Path(self.config.landmarks_dir) / f"{sample.sample_id}.npz"
        with np.load(landmarks_path) as data:
            points = np.asarray(data["points"], dtype=np.float32)
            valid = np.asarray(data["valid"], dtype=bool)
            if "frame_idx" in data:
                frame_idx = np.asarray(data["frame_idx"], dtype=np.int64)
            else:
                frame_idx = np.arange(points.shape[0], dtype=np.int64)

        points = interpolate_invalid(points, valid)
        sel = self._select_frames(int(frame_idx.size), self.config.n_video_tokens)
        points = points[frame_idx[sel]]

        points = normalize_landmarks(
            points,
            method=self.video_cfg.normalize,
            align_rotation=self.video_cfg.align_rotation,
            coords=3,
        )
        points = points[:, self._region, :]

        shift_seconds = self._sample_shift_seconds(index)
        is_positive = shift_seconds == 0.0

        return {
            "mel_window": mel_window,
            "landmarks": torch.from_numpy(points).float(),
            "fps": fps,
            "window_seconds": window_seconds,
            "frame_idx0": frame_idx0,
            "duration": sample.duration,
            "label": sample.label,
            "label_name": sample.label_name,
            "sample_id": sample.sample_id,
            "split": sample.split,
            "modify_audio": sample.modify_audio,
            "modify_video": sample.modify_video,
            "n_fakes": sample.n_fakes,
            "fake_periods": sample.fake_periods,
            "shift_seconds": shift_seconds,
            "is_positive": is_positive,
        }

    def get_sample_by_id(self, sample_id: str) -> LAVDFSyncSample | None:
        """Get sample by ID."""
        for sample in self.samples:
            if sample.sample_id == sample_id:
                return sample
        return None

    @property
    def sample_ids(self) -> list[str]:
        return [s.sample_id for s in self.samples]

    @property
    def audio_paths(self) -> list[Path]:
        return [s.audio_path for s in self.samples]


def lavdf_collate_fn(batch: list[dict]) -> dict:
    """Custom collate function to handle variable-sized mel windows.

    Defined at module scope (rather than inline in a training script) so it is
    picklable by the ``spawn`` start method that Windows' multiprocessing
    ``DataLoader`` workers require.
    """
    # Find max mel length in batch
    max_mel_len = max(item["mel_window"].shape[1] for item in batch)
    n_mels = batch[0]["mel_window"].shape[0]

    # Pad mel windows to max length
    mel_windows = []
    for item in batch:
        mel = item["mel_window"]
        if mel.shape[1] < max_mel_len:
            padding = max_mel_len - mel.shape[1]
            mel = torch.nn.functional.pad(mel, (0, padding), mode='constant', value=0)
        mel_windows.append(mel)

    # Stack all tensors
    return {
        "mel_window": torch.stack(mel_windows),
        "landmarks": torch.stack([item["landmarks"] for item in batch]),
        "fps": [item["fps"] for item in batch],
        "window_seconds": [item["window_seconds"] for item in batch],
        "shift_seconds": [item["shift_seconds"] for item in batch],
        "is_positive": [item["is_positive"] for item in batch],
    }


def build_lavdf_datasets(
    manifest_path: str | Path,
    video_dir: str | Path,
    audio_dir: str | Path,
    landmarks_dir: str | Path,
    audio_cfg: AudioConfig,
    video_cfg: VideoDataConfig,
    *,
    splits: tuple[str, ...] = ("train", "dev"),
    **kwargs,
) -> dict[str, LAVDFSyncDataset]:
    """Build LAV-DF datasets for specified splits.

    Training pipelines must load ``splits=('train',)`` for training and
    ``splits=('dev',)`` for validation to preserve official split separation.
    """
    manifest_path = Path(manifest_path)
    datasets: dict[str, LAVDFSyncDataset] = {}

    for split in splits:
        config = LAVDFSyncConfig(
            manifest_path=manifest_path,
            video_dir=video_dir,
            audio_dir=audio_dir,
            landmarks_dir=landmarks_dir,
            split=split,
            **kwargs,
        )
        datasets[split] = LAVDFSyncDataset(config, audio_cfg, video_cfg)

    return datasets
