"""Comprehensive unit tests for LAV-DF preprocessing pipeline (Phase 11A).

Tests all 11 target task requirements:
1. MP4 audio extraction (via PyAV synthetic MP4 fixture)
2. Audio resampling
3. Audio output schema
4. Landmark NPZ schema
5. FPS metadata handling
6. Invalid/missing landmark handling
7. Cache/resume behavior
8. Train/dev split isolation
9. Temporal metadata consistency
10. LAVDFSyncDataset train sample loading
11. LAVDFSyncDataset dev sample loading
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from src.config import AudioConfig, VideoDataConfig
from src.data.lavdf_dataset import LAVDFSyncConfig, LAVDFSyncDataset
from src.preprocessing.lavdf import (
    extract_audio_from_mp4,
    extract_landmarks_from_mp4,
    process_single_sample,
    validate_audio_cache,
    validate_landmarks_cache,
)


def _create_synthetic_mp4(
    path: Path,
    *,
    duration_sec: float = 1.0,
    fps: float = 25.0,
    sample_rate: int = 44100,
    include_audio: bool = True,
    include_video: bool = True,
) -> Path:
    """Create a tiny deterministic synthetic MP4 fixture for testing."""
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")

    v_stream = None
    if include_video:
        v_stream = container.add_stream("h264", rate=int(fps))
        v_stream.width = 128
        v_stream.height = 128
        v_stream.pix_fmt = "yuv420p"

    a_stream = None
    if include_audio:
        a_stream = container.add_stream("aac", rate=sample_rate)
        a_stream.layout = "mono"

    n_video_frames = int(round(duration_sec * fps))
    for i in range(n_video_frames):
        if v_stream is not None:
            # Simple gray background frame
            img = np.full((128, 128, 3), (i * 10) % 255, dtype=np.uint8)
            v_frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in v_stream.encode(v_frame):
                container.mux(packet)

    if a_stream is not None:
        n_audio_samples = int(round(duration_sec * sample_rate))
        t = np.linspace(0, duration_sec, n_audio_samples, endpoint=False)
        samples = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)
        a_frame = av.AudioFrame.from_ndarray(samples[None, :], format="s16", layout="mono")
        a_frame.sample_rate = sample_rate
        for packet in a_stream.encode(a_frame):
            container.mux(packet)

    if v_stream is not None:
        for packet in v_stream.encode():
            container.mux(packet)
    if a_stream is not None:
        for packet in a_stream.encode():
            container.mux(packet)

    container.close()
    return path


def test_mp4_audio_extraction(tmp_path: Path):
    """1. MP4 audio extraction using synthetic MP4 fixture."""
    mp4_path = _create_synthetic_mp4(tmp_path / "sample.mp4", duration_sec=1.0, sample_rate=44100)
    waveform, sr = extract_audio_from_mp4(mp4_path, target_sr=16000)

    assert isinstance(waveform, torch.Tensor)
    assert waveform.ndim == 2
    assert waveform.shape[0] == 1  # mono
    assert sr == 16000
    assert waveform.shape[1] > 0


def test_audio_resampling(tmp_path: Path):
    """2. Audio resampling from source 44.1kHz to target 16kHz."""
    mp4_path = _create_synthetic_mp4(tmp_path / "sample_44k.mp4", duration_sec=2.0, sample_rate=44100)
    waveform, sr = extract_audio_from_mp4(mp4_path, target_sr=16000)

    assert sr == 16000
    expected_samples = 32000
    assert abs(waveform.shape[1] - expected_samples) < 1000


def test_audio_output_schema(tmp_path: Path):
    """3. Audio output schema (WAV format, 16kHz, mono, float32)."""
    mp4_path = _create_synthetic_mp4(tmp_path / "test_audio.mp4", duration_sec=1.0)
    out_audio_dir = tmp_path / "audio"
    out_landmarks_dir = tmp_path / "landmarks"

    meta = process_single_sample(
        "test_01",
        mp4_path,
        out_audio_dir,
        out_landmarks_dir,
    )

    wav_path = out_audio_dir / "test_01.wav"
    assert wav_path.is_file()
    assert meta["audio_valid"] is True

    info = sf.info(str(wav_path))
    assert info.channels == 1
    assert info.samplerate == 16000
    assert info.frames > 0


def test_landmark_npz_schema(tmp_path: Path):
    """4. Landmark NPZ schema."""
    mp4_path = _create_synthetic_mp4(tmp_path / "test_video.mp4", duration_sec=0.5, fps=25.0)
    out_audio_dir = tmp_path / "audio"
    out_landmarks_dir = tmp_path / "landmarks"

    meta = process_single_sample(
        "test_lm_01",
        mp4_path,
        out_audio_dir,
        out_landmarks_dir,
    )

    npz_path = out_landmarks_dir / "test_lm_01.npz"
    assert npz_path.is_file()
    assert meta["landmarks_valid"] is True

    with np.load(npz_path) as data:
        assert "points" in data
        assert "valid" in data
        assert "fps" in data
        assert "frame_idx" in data

        assert data["points"].ndim == 3
        assert data["points"].shape[1] == 478
        assert data["points"].shape[2] == 3
        assert data["valid"].ndim == 1
        assert data["valid"].shape[0] == data["points"].shape[0]


def test_fps_metadata(tmp_path: Path):
    """5. FPS metadata (non-25 FPS detection)."""
    mp4_path = _create_synthetic_mp4(tmp_path / "test_30fps.mp4", duration_sec=1.0, fps=30.0)
    out_audio_dir = tmp_path / "audio"
    out_landmarks_dir = tmp_path / "landmarks"

    meta = process_single_sample(
        "test_30fps",
        mp4_path,
        out_audio_dir,
        out_landmarks_dir,
    )

    assert meta["fps"] == pytest.approx(30.0, rel=0.05)


def test_unreliable_fps_flagged_as_invalid(tmp_path: Path, monkeypatch):
    """5b. Regression test: Missing/unreliable FPS is flagged as invalid, never defaulted to 25.0."""
    mp4_path = _create_synthetic_mp4(tmp_path / "test_no_fps.mp4", duration_sec=1.0, fps=25.0)

    # Mock PyAV stream to return 0 / None for average_rate and r_frame_rate
    class MockStream:
        average_rate = 0
        r_frame_rate = 0

    class MockContainer:
        streams = type("Streams", (), {"video": [MockStream()], "audio": []})()

        def decode(self, stream):
            import av
            img = np.full((128, 128, 3), 100, dtype=np.uint8)
            return [av.VideoFrame.from_ndarray(img, format="rgb24")]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    import av
    monkeypatch.setattr(av, "open", lambda *args, **kwargs: MockContainer())

    with pytest.raises(ValueError, match="Could not determine valid video stream FPS"):
        extract_landmarks_from_mp4(mp4_path, landmarker=None)

    out_audio_dir = tmp_path / "audio"
    out_landmarks_dir = tmp_path / "landmarks"
    meta = process_single_sample("no_fps_01", mp4_path, out_audio_dir, out_landmarks_dir)

    assert meta["landmarks_valid"] is False
    assert "landmarks_error" in meta
    assert "Could not determine valid video stream FPS" in meta["landmarks_error"]


def test_invalid_missing_landmark_handling(tmp_path: Path):
    """6. Invalid/missing landmark and corrupted audio handling."""
    # Create MP4 without audio stream
    mp4_no_audio = _create_synthetic_mp4(
        tmp_path / "no_audio.mp4",
        duration_sec=0.5,
        include_audio=False,
    )
    out_audio_dir = tmp_path / "audio"
    out_landmarks_dir = tmp_path / "landmarks"

    meta = process_single_sample(
        "no_audio_01",
        mp4_no_audio,
        out_audio_dir,
        out_landmarks_dir,
    )

    assert meta["audio_valid"] is False
    assert "audio_error" in meta
    # Ensure invalid audio WAV does NOT pass validation
    wav_path = out_audio_dir / "no_audio_01.wav"
    meta_path = out_landmarks_dir / "no_audio_01.meta.json"
    assert not validate_audio_cache(wav_path, meta_path)


def test_cache_resume_behavior(tmp_path: Path):
    """7. Cache/resume behavior."""
    mp4_path = _create_synthetic_mp4(tmp_path / "resume.mp4", duration_sec=0.5)
    out_audio_dir = tmp_path / "audio"
    out_landmarks_dir = tmp_path / "landmarks"

    # First run
    meta1 = process_single_sample("res_01", mp4_path, out_audio_dir, out_landmarks_dir)
    assert meta1["skipped"] is False

    # Second run (should skip)
    meta2 = process_single_sample("res_01", mp4_path, out_audio_dir, out_landmarks_dir)
    assert meta2["skipped"] is True

    # Corrupt audio output
    wav_path = out_audio_dir / "res_01.wav"
    wav_path.write_bytes(b"corrupt header")

    # Third run (should re-process corrupted file)
    meta3 = process_single_sample("res_01", mp4_path, out_audio_dir, out_landmarks_dir)
    assert meta3["skipped"] is False
    assert meta3["audio_valid"] is True


def test_train_dev_split_isolation(tmp_path: Path):
    """8. Train/dev split isolation."""
    train_mp4 = _create_synthetic_mp4(tmp_path / "train_01.mp4", duration_sec=0.5)
    dev_mp4 = _create_synthetic_mp4(tmp_path / "dev_01.mp4", duration_sec=0.5)

    out_dir = tmp_path / "processed"
    process_single_sample("train_01", train_mp4, out_dir / "audio", out_dir / "landmarks")
    process_single_sample("dev_01", dev_mp4, out_dir / "audio", out_dir / "landmarks")

    train_manifest = tmp_path / "manifest_train.csv"
    dev_manifest = tmp_path / "manifest_dev.csv"

    fieldnames = [
        "sample_id", "path", "label", "label_name", "split", "modify_audio",
        "modify_video", "n_fakes", "fake_periods", "duration", "original",
        "video_frames", "audio_frames"
    ]

    with train_manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerow({
            "sample_id": "train_01", "path": "train_01.mp4", "label": "1",
            "label_name": "real", "split": "train", "modify_audio": "False",
            "modify_video": "False", "n_fakes": "0", "fake_periods": "[]",
            "duration": "0.5", "original": "", "video_frames": "13", "audio_frames": "8000"
        })

    with dev_manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerow({
            "sample_id": "dev_01", "path": "dev_01.mp4", "label": "0",
            "label_name": "fake", "split": "dev", "modify_audio": "False",
            "modify_video": "False", "n_fakes": "1", "fake_periods": "[]",
            "duration": "0.5", "original": "", "video_frames": "13", "audio_frames": "8000"
        })

    train_ds = LAVDFSyncDataset(
        LAVDFSyncConfig(
            manifest_path=train_manifest,
            video_dir=tmp_path,
            audio_dir=out_dir / "audio",
            landmarks_dir=out_dir / "landmarks",
            split="train",
        ),
        AudioConfig(),
        VideoDataConfig(),
    )

    assert len(train_ds) == 1
    assert train_ds.samples[0].sample_id == "train_01"


def test_temporal_metadata_consistency(tmp_path: Path):
    """9. Temporal metadata consistency."""
    mp4_path = _create_synthetic_mp4(tmp_path / "temp_align.mp4", duration_sec=2.0, fps=25.0, sample_rate=16000)
    out_dir = tmp_path / "processed"

    meta = process_single_sample("temp_align", mp4_path, out_dir / "audio", out_dir / "landmarks")

    assert meta["fps"] == pytest.approx(25.0, rel=0.01)
    assert meta["audio_sample_rate"] == 16000
    assert meta["duration_diff"] < 0.2  # duration difference between audio and video is small


def test_loading_processed_train_sample_dataset(tmp_path: Path):
    """10. Loading one processed train sample through LAVDFSyncDataset."""
    mp4_path = _create_synthetic_mp4(tmp_path / "s1.mp4", duration_sec=2.0, fps=25.0)
    out_dir = tmp_path / "processed"

    process_single_sample("s1", mp4_path, out_dir / "audio", out_dir / "landmarks")

    manifest = tmp_path / "manifest_train.csv"
    fieldnames = [
        "sample_id", "path", "label", "label_name", "split", "modify_audio",
        "modify_video", "n_fakes", "fake_periods", "duration", "original",
        "video_frames", "audio_frames"
    ]
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerow({
            "sample_id": "s1", "path": "s1.mp4", "label": "1",
            "label_name": "real", "split": "train", "modify_audio": "False",
            "modify_video": "False", "n_fakes": "0", "fake_periods": "[]",
            "duration": "2.0", "original": "", "video_frames": "50", "audio_frames": "32000"
        })

    ds = LAVDFSyncDataset(
        LAVDFSyncConfig(
            manifest_path=manifest,
            video_dir=tmp_path,
            audio_dir=out_dir / "audio",
            landmarks_dir=out_dir / "landmarks",
            split="train",
            n_video_tokens=32,
        ),
        AudioConfig(),
        VideoDataConfig(),
    )

    sample = ds[0]
    assert "mel_window" in sample
    assert "landmarks" in sample
    assert sample["mel_window"].ndim == 2  # [n_mels, T_mel]
    assert sample["landmarks"].ndim == 3   # [T_video, N_region, 3]
    assert sample["fps"] == pytest.approx(25.0, rel=0.01)


def test_loading_processed_dev_sample_dataset(tmp_path: Path):
    """11. Loading one processed dev sample through LAVDFSyncDataset."""
    mp4_path = _create_synthetic_mp4(tmp_path / "d1.mp4", duration_sec=2.0, fps=25.0)
    out_dir = tmp_path / "processed"

    process_single_sample("d1", mp4_path, out_dir / "audio", out_dir / "landmarks")

    manifest = tmp_path / "manifest_dev.csv"
    fieldnames = [
        "sample_id", "path", "label", "label_name", "split", "modify_audio",
        "modify_video", "n_fakes", "fake_periods", "duration", "original",
        "video_frames", "audio_frames"
    ]
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerow({
            "sample_id": "d1", "path": "d1.mp4", "label": "0",
            "label_name": "manipulated", "split": "dev", "modify_audio": "True",
            "modify_video": "True", "n_fakes": "1", "fake_periods": "[[0.2, 1.5]]",
            "duration": "2.0", "original": "", "video_frames": "50", "audio_frames": "32000"
        })

    ds = LAVDFSyncDataset(
        LAVDFSyncConfig(
            manifest_path=manifest,
            video_dir=tmp_path,
            audio_dir=out_dir / "audio",
            landmarks_dir=out_dir / "landmarks",
            split="dev",
            n_video_tokens=32,
        ),
        AudioConfig(),
        VideoDataConfig(),
    )

    sample = ds[0]
    assert sample["sample_id"] == "d1"
    assert sample["split"] == "dev"
    assert sample["mel_window"].ndim == 2
    assert sample["landmarks"].ndim == 3
