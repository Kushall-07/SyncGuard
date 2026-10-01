"""Tests for src/data/lavdf_physical_sync_dataset.py (physical-shift training data).

Covers the mandatory regression list from the sync-detection fix: physical
(waveform-level) negatives vs. positives, cross-clip negatives, deterministic
window/shift/partner selection, split isolation, and production-matching
32-frame consecutive windowing.
"""

from __future__ import annotations

import csv
import wave
from pathlib import Path

import numpy as np
import pytest
import torch

from src.config import AudioConfig, VideoDataConfig
from src.data.lavdf_physical_sync_dataset import (
    LAVDFPhysicalSyncConfig,
    LAVDFPhysicalSyncDataset,
    PhysicalSyncPairConfig,
    build_lavdf_physical_datasets,
    physical_sync_collate_fn,
)


def _write_wav(path: Path, *, sample_rate: int = 16000, duration: float = 2.0, freq: float = 220.0) -> None:
    # A pure stationary tone is a poor test fixture here: mel spectrograms are
    # magnitude-only, so shifting a perfectly periodic signal by any amount is
    # literally invisible to a mel-based encoder (every analysis frame of a
    # steady-state sinusoid looks the same). Use deterministic band-limited
    # noise with a slowly varying envelope instead, so a temporal shift
    # actually moves distinguishable energy into/out of the analysis window.
    n = int(sample_rate * duration)
    rng = np.random.RandomState(int(freq))
    t = np.arange(n) / sample_rate
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t)  # slow amplitude envelope, not periodic w.r.t. window length
    noise = rng.randn(n)
    samples = (envelope * noise / max(np.abs(envelope * noise).max(), 1e-6) * 32767 * 0.9).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


def _write_landmarks_npz(path: Path, *, n_frames: int = 60, fps: float = 30.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    points = rng.randn(n_frames, 478, 3).astype(np.float32)
    valid = np.ones(n_frames, dtype=bool)
    frame_idx = np.arange(n_frames, dtype=np.int64)
    np.savez(path, points=points, valid=valid, fps=np.array(fps), frame_idx=frame_idx)


_COLUMNS = [
    "sample_id", "path", "label", "label_name", "split", "modify_audio", "modify_video",
    "n_fakes", "fake_periods", "duration", "original", "video_frames", "audio_frames",
]


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _row(sample_id: str, split: str, freq: float = 220.0) -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "path": f"{split}/{sample_id}.mp4",
        "label": "1",
        "label_name": "real",
        "split": split,
        "modify_audio": "False",
        "modify_video": "False",
        "n_fakes": "0",
        "fake_periods": "[]",
        "duration": "2.0",
        "original": "",
        "video_frames": "60",
        "audio_frames": "32000",
    }


@pytest.fixture
def fixture(tmp_path: Path) -> dict[str, Path]:
    manifest = tmp_path / "manifest.csv"
    video_dir = tmp_path / "videos"
    audio_dir = tmp_path / "audio"
    landmarks_dir = tmp_path / "landmarks"

    rows = [
        _row("train_001", "train", freq=220.0),
        _row("train_002", "train", freq=330.0),
        _row("train_003", "train", freq=440.0),
        _row("dev_001", "dev", freq=550.0),
    ]
    _write_manifest(manifest, rows)
    for row, freq in zip(rows, (220.0, 330.0, 440.0, 550.0)):
        _write_wav(audio_dir / f"{row['sample_id']}.wav", freq=freq)
        _write_landmarks_npz(landmarks_dir / f"{row['sample_id']}.npz")

    return {"manifest": manifest, "video_dir": video_dir, "audio_dir": audio_dir, "landmarks_dir": landmarks_dir}


def _make_dataset(fixture: dict[str, Path], pair_cfg: PhysicalSyncPairConfig, split: str = "train", seed: int = 0) -> LAVDFPhysicalSyncDataset:
    cfg = LAVDFPhysicalSyncConfig(
        manifest_path=fixture["manifest"],
        video_dir=fixture["video_dir"],
        audio_dir=fixture["audio_dir"],
        landmarks_dir=fixture["landmarks_dir"],
        split=split,
        pair_config=pair_cfg,
        seed=seed,
    )
    return LAVDFPhysicalSyncDataset(cfg, AudioConfig(), VideoDataConfig())


def test_window_count_matches_production_windowing(fixture: dict[str, Path]) -> None:
    # 60 frames @ 32-frame/32-stride windows -> [0,32) and [32,60): 2 windows/clip.
    ds = _make_dataset(fixture, PhysicalSyncPairConfig(negative_pair_probability=0.0))
    assert len(ds) == 3 * 2  # 3 train clips


def test_max_windows_per_clip_caps_and_covers_every_clip(fixture: dict[str, Path]) -> None:
    cfg = LAVDFPhysicalSyncConfig(
        manifest_path=fixture["manifest"], video_dir=fixture["video_dir"], audio_dir=fixture["audio_dir"],
        landmarks_dir=fixture["landmarks_dir"], split="train",
        pair_config=PhysicalSyncPairConfig(negative_pair_probability=0.0),
        max_windows_per_clip=1,
    )
    ds = LAVDFPhysicalSyncDataset(cfg, AudioConfig(), VideoDataConfig())
    assert len(ds) == 3  # one window per clip, all 3 clips still represented
    sample_idxs = {e.sample_idx for e in ds._entries}
    assert sample_idxs == {0, 1, 2}


def test_positive_item_fields(fixture: dict[str, Path]) -> None:
    ds = _make_dataset(fixture, PhysicalSyncPairConfig(negative_pair_probability=0.0))
    item = ds[0]
    assert item["label"] == 1.0
    assert item["negative_type"] == "none"
    assert item["shift_seconds"] == 0.0
    assert item["landmarks"].shape[0] == 32
    assert item["mel_window"].ndim == 2


def test_physical_shift_negative_has_nonzero_shift(fixture: dict[str, Path]) -> None:
    ds = _make_dataset(
        fixture, PhysicalSyncPairConfig(negative_pair_probability=1.0, cross_clip_probability=0.0)
    )
    item = ds[0]
    assert item["label"] == 0.0
    assert item["negative_type"] == "physical_shift"
    assert item["shift_seconds"] != 0.0
    assert abs(item["shift_seconds"]) in PhysicalSyncPairConfig().shift_magnitudes_seconds


def test_cross_clip_negative_uses_different_sample(fixture: dict[str, Path]) -> None:
    ds = _make_dataset(
        fixture, PhysicalSyncPairConfig(negative_pair_probability=1.0, cross_clip_probability=1.0)
    )
    item = ds[0]
    assert item["label"] == 0.0
    assert item["negative_type"] == "cross_clip"
    assert item["partner_sample_id"] != ""
    assert item["partner_sample_id"] != item["sample_id"]
    assert item["shift_seconds"] == 0.0  # cross-clip is not a shift


def test_audio_differs_between_positive_and_physical_shift_negative(fixture: dict[str, Path]) -> None:
    pos_ds = _make_dataset(fixture, PhysicalSyncPairConfig(negative_pair_probability=0.0), seed=7)
    neg_ds = _make_dataset(
        fixture, PhysicalSyncPairConfig(negative_pair_probability=1.0, cross_clip_probability=0.0), seed=7
    )
    pos_mel = pos_ds[0]["mel_window"]
    neg_mel = neg_ds[0]["mel_window"]
    assert pos_mel.shape == neg_mel.shape
    assert not torch.allclose(pos_mel, neg_mel)


def test_deterministic_across_dataset_instances(fixture: dict[str, Path]) -> None:
    cfg = PhysicalSyncPairConfig(negative_pair_probability=0.5, cross_clip_probability=0.2)
    ds_a = _make_dataset(fixture, cfg, seed=3)
    ds_b = _make_dataset(fixture, cfg, seed=3)
    for i in range(len(ds_a)):
        a, b = ds_a[i], ds_b[i]
        assert a["negative_type"] == b["negative_type"]
        assert a["shift_seconds"] == b["shift_seconds"]
        assert torch.equal(a["mel_window"], b["mel_window"])
        assert torch.equal(a["landmarks"], b["landmarks"])


def test_split_isolation_train_and_dev(fixture: dict[str, Path]) -> None:
    datasets = build_lavdf_physical_datasets(
        fixture["manifest"], fixture["video_dir"], fixture["audio_dir"], fixture["landmarks_dir"],
        AudioConfig(), VideoDataConfig(), splits=("train", "dev"),
    )
    train_ids = {s.sample_id for s in datasets["train"].samples}
    dev_ids = {s.sample_id for s in datasets["dev"].samples}
    assert train_ids.isdisjoint(dev_ids)
    # cross-clip partners can only ever come from within the same dataset's own sample list
    ds = datasets["train"]
    cfg = PhysicalSyncPairConfig(negative_pair_probability=1.0, cross_clip_probability=1.0)
    ds2 = _make_dataset(fixture, cfg, split="train")
    for i in range(len(ds2)):
        partner = ds2[i]["partner_sample_id"]
        if partner:
            assert partner in train_ids


def test_collate_fn_stacks_batch(fixture: dict[str, Path]) -> None:
    ds = _make_dataset(fixture, PhysicalSyncPairConfig(negative_pair_probability=0.5))
    batch = physical_sync_collate_fn([ds[0], ds[1], ds[2]])
    assert batch["mel_window"].shape[0] == 3
    assert batch["landmarks"].shape == (3, 32, batch["landmarks"].shape[2], 3)
    assert batch["label"].shape == (3,)


def test_build_fixed_native_matches_positive(fixture: dict[str, Path]) -> None:
    ds = _make_dataset(fixture, PhysicalSyncPairConfig(negative_pair_probability=0.0))
    fixed = ds.build_fixed(0, shift_seconds=0.0)
    assert fixed["label"] == 1.0
    assert fixed["negative_type"] == "none"
    assert fixed["shift_seconds"] == 0.0


def test_build_fixed_shift_is_deterministic_and_labeled_desync(fixture: dict[str, Path]) -> None:
    ds = _make_dataset(fixture, PhysicalSyncPairConfig(negative_pair_probability=0.0))
    a = ds.build_fixed(0, shift_seconds=0.5)
    b = ds.build_fixed(0, shift_seconds=0.5)
    assert a["label"] == 0.0
    assert a["negative_type"] == "physical_shift"
    assert torch.equal(a["mel_window"], b["mel_window"])
    native = ds.build_fixed(0, shift_seconds=0.0)
    assert not torch.allclose(a["mel_window"], native["mel_window"])


def test_label_ignores_misleading_sample_id_or_path(fixture: dict[str, Path]) -> None:
    """The label must come only from the constructed shift, never from the
    sample_id/filename - a sample deliberately named to look like a desync
    example must still be labeled SYNC when no shift was actually applied.
    """
    rows = [_row("desync_0001_looks_like_desync_but_is_not", "train")]
    manifest = fixture["manifest"].parent / "manifest_misleading.csv"
    _write_manifest(manifest, rows)
    _write_wav(fixture["audio_dir"] / f"{rows[0]['sample_id']}.wav")
    _write_landmarks_npz(fixture["landmarks_dir"] / f"{rows[0]['sample_id']}.npz")
    cfg = LAVDFPhysicalSyncConfig(
        manifest_path=manifest, video_dir=fixture["video_dir"], audio_dir=fixture["audio_dir"],
        landmarks_dir=fixture["landmarks_dir"], split="train",
        pair_config=PhysicalSyncPairConfig(negative_pair_probability=0.0),
    )
    ds = LAVDFPhysicalSyncDataset(cfg, AudioConfig(), VideoDataConfig())
    assert ds.build_fixed(0, shift_seconds=0.0)["label"] == 1.0


def test_manipulation_label_not_used_as_sync_target(fixture: dict[str, Path]) -> None:
    # Even a manipulated-content sample's positive (unshifted) window must be SYNC.
    rows = [_row("train_fake", "train")]
    rows[0]["label"] = "0"
    rows[0]["label_name"] = "manipulated"
    rows[0]["modify_audio"] = "True"
    rows[0]["modify_video"] = "True"
    manifest = fixture["manifest"].parent / "manifest2.csv"
    _write_manifest(manifest, rows)
    _write_wav(fixture["audio_dir"] / "train_fake.wav")
    _write_landmarks_npz(fixture["landmarks_dir"] / "train_fake.npz")
    cfg = LAVDFPhysicalSyncConfig(
        manifest_path=manifest, video_dir=fixture["video_dir"], audio_dir=fixture["audio_dir"],
        landmarks_dir=fixture["landmarks_dir"], split="train",
        pair_config=PhysicalSyncPairConfig(negative_pair_probability=0.0),
    )
    ds = LAVDFPhysicalSyncDataset(cfg, AudioConfig(), VideoDataConfig())
    assert ds[0]["label"] == 1.0
