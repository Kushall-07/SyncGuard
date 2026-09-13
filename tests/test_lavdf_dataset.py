"""Tests for LAV-DF sync dataset loader (Phase 11)."""

from __future__ import annotations

import csv
import wave
from pathlib import Path

import numpy as np
import pytest
import torch

from src.config import AudioConfig, VideoDataConfig
from src.data.lavdf_dataset import (
    LAVDFSyncConfig,
    LAVDFSyncDataset,
    build_lavdf_datasets,
    compute_window_seconds,
    crop_audio_window,
)
from src.data.sync_pairs import SyncPairConfig


def _write_wav(path: Path, *, sample_rate: int = 16000, duration: float = 2.0) -> None:
    n = int(sample_rate * duration)
    samples = (np.sin(np.linspace(0, 40 * np.pi, n)) * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


def _write_landmarks_npz(
    path: Path,
    *,
    n_frames: int = 100,
    fps: float = 30.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.random.randn(n_frames, 478, 3).astype(np.float32)
    valid = np.ones(n_frames, dtype=bool)
    frame_idx = np.arange(n_frames, dtype=np.int64)
    np.savez(path, points=points, valid=valid, fps=np.array(fps), frame_idx=frame_idx)


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    columns = [
        "sample_id",
        "path",
        "label",
        "label_name",
        "split",
        "modify_audio",
        "modify_video",
        "n_fakes",
        "fake_periods",
        "duration",
        "original",
        "video_frames",
        "audio_frames",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@pytest.fixture
def lavdf_fixture(tmp_path: Path) -> dict[str, Path]:
    manifest = tmp_path / "manifest.csv"
    video_dir = tmp_path / "videos"
    audio_dir = tmp_path / "audio"
    landmarks_dir = tmp_path / "landmarks"

    rows = [
        {
            "sample_id": "train__real_001",
            "path": "train/real_001.mp4",
            "label": "1",
            "label_name": "real",
            "split": "train",
            "modify_audio": "False",
            "modify_video": "False",
            "n_fakes": "0",
            "fake_periods": "[]",
            "duration": "2.0",
            "original": "",
            "video_frames": "60",
            "audio_frames": "32000",
        },
        {
            "sample_id": "train__av_fake_001",
            "path": "train/av_fake_001.mp4",
            "label": "0",
            "label_name": "manipulated",
            "split": "train",
            "modify_audio": "True",
            "modify_video": "True",
            "n_fakes": "1",
            "fake_periods": "[[0.5, 1.0]]",
            "duration": "2.0",
            "original": "real_001.mp4",
            "video_frames": "60",
            "audio_frames": "32000",
        },
        {
            "sample_id": "dev__real_002",
            "path": "dev/real_002.mp4",
            "label": "1",
            "label_name": "real",
            "split": "dev",
            "modify_audio": "False",
            "modify_video": "False",
            "n_fakes": "0",
            "fake_periods": "[]",
            "duration": "2.0",
            "original": "",
            "video_frames": "60",
            "audio_frames": "32000",
        },
    ]
    _write_manifest(manifest, rows)

    for row in rows:
        _write_wav(audio_dir / f"{Path(row['path']).stem}.wav")
        _write_landmarks_npz(landmarks_dir / f"{row['sample_id']}.npz", fps=30.0)

    return {
        "manifest": manifest,
        "video_dir": video_dir,
        "audio_dir": audio_dir,
        "landmarks_dir": landmarks_dir,
    }


def test_crop_audio_window_padding() -> None:
    waveform = torch.ones(1, 8000)
    cropped = crop_audio_window(waveform, 16000, start_seconds=0.0, window_seconds=1.0)
    assert cropped.shape == (1, 16000)


def test_compute_window_seconds() -> None:
    assert compute_window_seconds(32, 25.0) == pytest.approx(1.28)


def test_dataset_returns_training_ready_fields(lavdf_fixture: dict[str, Path]) -> None:
    cfg = LAVDFSyncConfig(
        manifest_path=lavdf_fixture["manifest"],
        video_dir=lavdf_fixture["video_dir"],
        audio_dir=lavdf_fixture["audio_dir"],
        landmarks_dir=lavdf_fixture["landmarks_dir"],
        split="train",
        use_negative_pairs=False,
        n_video_tokens=32,
    )
    ds = LAVDFSyncDataset(cfg, AudioConfig(), VideoDataConfig())

    item = ds[0]
    assert set(item.keys()) >= {
        "mel_window",
        "landmarks",
        "fps",
        "window_seconds",
        "frame_idx0",
        "shift_seconds",
        "is_positive",
        "fake_periods",
        "sample_id",
        "split",
    }
    assert item["mel_window"].ndim == 2
    assert item["landmarks"].shape[0] == 32
    assert item["fps"] == 30.0
    assert item["window_seconds"] == pytest.approx(32 / 30.0)
    assert item["is_positive"] is True
    assert item["shift_seconds"] == 0.0
    assert item["fake_periods"] == []


def test_timing_uses_npz_fps_not_default(lavdf_fixture: dict[str, Path]) -> None:
    cfg = LAVDFSyncConfig(
        manifest_path=lavdf_fixture["manifest"],
        video_dir=lavdf_fixture["video_dir"],
        audio_dir=lavdf_fixture["audio_dir"],
        landmarks_dir=lavdf_fixture["landmarks_dir"],
        split="train",
        use_negative_pairs=False,
    )
    ds = LAVDFSyncDataset(cfg, AudioConfig(), VideoDataConfig())
    timing = ds.timing(0)
    assert timing["fps"] == 30.0
    assert timing["window_seconds"] == pytest.approx(32 / 30.0)


def test_train_loader_contains_only_train_rows(lavdf_fixture: dict[str, Path]) -> None:
    train_ds = build_lavdf_datasets(
        lavdf_fixture["manifest"],
        lavdf_fixture["video_dir"],
        lavdf_fixture["audio_dir"],
        lavdf_fixture["landmarks_dir"],
        AudioConfig(),
        VideoDataConfig(),
        splits=("train",),
        use_negative_pairs=False,
    )["train"]

    assert len(train_ds) == 2
    assert all(s.split == "train" for s in train_ds.samples)
    assert {s.sample_id for s in train_ds.samples} == {"train__real_001", "train__av_fake_001"}


def test_dev_loader_contains_only_dev_rows(lavdf_fixture: dict[str, Path]) -> None:
    dev_ds = build_lavdf_datasets(
        lavdf_fixture["manifest"],
        lavdf_fixture["video_dir"],
        lavdf_fixture["audio_dir"],
        lavdf_fixture["landmarks_dir"],
        AudioConfig(),
        VideoDataConfig(),
        splits=("dev",),
        use_negative_pairs=False,
    )["dev"]

    assert len(dev_ds) == 1
    assert dev_ds.samples[0].split == "dev"
    assert dev_ds.samples[0].sample_id == "dev__real_002"


def test_no_path_overlap_between_train_and_dev(lavdf_fixture: dict[str, Path]) -> None:
    datasets = build_lavdf_datasets(
        lavdf_fixture["manifest"],
        lavdf_fixture["video_dir"],
        lavdf_fixture["audio_dir"],
        lavdf_fixture["landmarks_dir"],
        AudioConfig(),
        VideoDataConfig(),
        splits=("train", "dev"),
        use_negative_pairs=False,
    )
    train_paths = {str(p) for p in datasets["train"].audio_paths}
    dev_paths = {str(p) for p in datasets["dev"].audio_paths}
    assert train_paths.isdisjoint(dev_paths)

    train_ids = set(datasets["train"].sample_ids)
    dev_ids = set(datasets["dev"].sample_ids)
    assert train_ids.isdisjoint(dev_ids)


def test_fake_periods_preserved_not_used_as_sync_labels(lavdf_fixture: dict[str, Path]) -> None:
    cfg = LAVDFSyncConfig(
        manifest_path=lavdf_fixture["manifest"],
        video_dir=lavdf_fixture["video_dir"],
        audio_dir=lavdf_fixture["audio_dir"],
        landmarks_dir=lavdf_fixture["landmarks_dir"],
        split="train",
        use_negative_pairs=False,
    )
    ds = LAVDFSyncDataset(cfg, AudioConfig(), VideoDataConfig())
    item = ds[1]  # manipulated sample with fake_periods
    assert item["fake_periods"] == [[0.5, 1.0]]
    assert item["is_positive"] is True  # sync label comes from shift, not manipulation


def test_negative_shift_sampling(lavdf_fixture: dict[str, Path]) -> None:
    cfg = LAVDFSyncConfig(
        manifest_path=lavdf_fixture["manifest"],
        video_dir=lavdf_fixture["video_dir"],
        audio_dir=lavdf_fixture["audio_dir"],
        landmarks_dir=lavdf_fixture["landmarks_dir"],
        split="train",
        use_negative_pairs=True,
        negative_pair_probability=1.0,
        sync_pair_config=SyncPairConfig(shift_seconds=[0.5]),
        seed=42,
    )
    ds = LAVDFSyncDataset(cfg, AudioConfig(), VideoDataConfig())
    item = ds[0]
    assert item["shift_seconds"] == 0.5
    assert item["is_positive"] is False


def test_empty_split_raises(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "sample_id": "dev__real_002",
                "path": "dev/real_002.mp4",
                "label": "1",
                "label_name": "real",
                "split": "dev",
                "modify_audio": "False",
                "modify_video": "False",
                "n_fakes": "0",
                "fake_periods": "[]",
                "duration": "2.0",
                "original": "",
                "video_frames": "60",
                "audio_frames": "32000",
            }
        ],
    )
    cfg = LAVDFSyncConfig(
        manifest_path=manifest,
        video_dir=tmp_path / "videos",
        audio_dir=tmp_path / "audio",
        landmarks_dir=tmp_path / "landmarks",
        split="train",
        use_negative_pairs=False,
    )
    with pytest.raises(ValueError, match="No samples found"):
        LAVDFSyncDataset(cfg, AudioConfig(), VideoDataConfig())
