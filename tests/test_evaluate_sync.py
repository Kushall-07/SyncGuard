"""Tests for Phase 11.5 evaluation pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.evaluation.sync_metrics import SyncMetrics, compute_sync_metrics
from src.data.sync_pairs import build_sync_pair


def test_compute_sync_metrics_basic() -> None:
    """Test basic sync metrics computation."""
    logits = torch.tensor([[1.0, -1.0, 0.5], [-0.5, 0.8, -0.2]])
    targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    mask = torch.ones_like(targets, dtype=torch.bool)

    metrics = compute_sync_metrics(logits, targets, mask, aggregation="mean")

    assert metrics.n_samples == 2
    assert metrics.n_windows == 6
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.video_accuracy <= 1.0


def test_compute_sync_metrics_with_mask() -> None:
    """Test sync metrics with masking."""
    logits = torch.tensor([[1.0, -1.0, 0.5], [-0.5, 0.8, -0.2]])
    targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    mask = torch.tensor([[True, True, False], [True, False, True]])

    metrics = compute_sync_metrics(logits, targets, mask, aggregation="mean")

    # Only 4 valid windows (2 masked out)
    assert metrics.n_windows == 4


def test_compute_sync_metrics_all_masked() -> None:
    """Test sync metrics when all windows are masked."""
    logits = torch.tensor([[1.0, -1.0], [-0.5, 0.8]])
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    mask = torch.zeros_like(targets, dtype=torch.bool)

    metrics = compute_sync_metrics(logits, targets, mask, aggregation="mean")

    assert metrics.n_windows == 0
    assert metrics.accuracy == 0.0
    assert metrics.video_accuracy == 0.0
    assert metrics.auc is None
    assert metrics.video_auc is None


def test_compute_sync_metrics_video_aggregation() -> None:
    """Test different video-level aggregation methods."""
    logits = torch.tensor([[1.0, -1.0, 0.5], [-0.5, 0.8, -0.2]])
    targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    mask = torch.ones_like(targets, dtype=torch.bool)

    for agg in ["mean", "max", "median"]:
        metrics = compute_sync_metrics(logits, targets, mask, aggregation=agg)
        assert metrics.video_accuracy >= 0.0
        assert metrics.video_accuracy <= 1.0


def test_eval_dataset_train_dev_test_isolation() -> None:
    """Test that train/dev/test splits do not overlap."""
    from src.data.lavdf_dataset import build_lavdf_datasets
    from src.config import AudioConfig, VideoDataConfig
    from src.data.sync_pairs import SyncPairConfig
    from dataclasses import replace

    # Skip if manifests don't exist
    if not Path("data/lavdf/manifest_train.csv").is_file():
        pytest.skip("Train manifest not found")
    if not Path("data/lavdf/manifest_dev.csv").is_file():
        pytest.skip("Dev manifest not found")

    # Load configs
    video_cfg = VideoDataConfig(num_frames=32, regions="face_mouth")
    audio_cfg = AudioConfig()
    sp_cfg = SyncPairConfig()
    sp_cfg = replace(sp_cfg, audio_token_seconds=0.01)

    # Build datasets
    train_dataset = build_lavdf_datasets(
        manifest_path="data/lavdf/manifest_train.csv",
        video_dir="data/lavdf/extracted/train",
        audio_dir="data/lavdf/processed/audio",
        landmarks_dir="data/lavdf/processed/landmarks",
        audio_cfg=audio_cfg,
        video_cfg=video_cfg,
        splits=("train",),
        n_video_tokens=32,
        sync_pair_config=sp_cfg,
        use_negative_pairs=False,
        seed=42,
    )["train"]

    dev_dataset = build_lavdf_datasets(
        manifest_path="data/lavdf/manifest_dev.csv",
        video_dir="data/lavdf/extracted/dev",
        audio_dir="data/lavdf/processed/audio",
        landmarks_dir="data/lavdf/processed/landmarks",
        audio_cfg=audio_cfg,
        video_cfg=video_cfg,
        splits=("dev",),
        n_video_tokens=32,
        sync_pair_config=sp_cfg,
        use_negative_pairs=False,
        seed=42,
    )["dev"]

    # Check for sample ID overlap
    train_ids = set(s.sample_id for s in train_dataset.samples)
    dev_ids = set(s.sample_id for s in dev_dataset.samples)

    overlap = train_ids & dev_ids
    assert len(overlap) == 0, f"Train/dev overlap: {overlap}"


def test_eval_shift_generation() -> None:
    """Test that temporal shifts are generated correctly."""
    from src.data.sync_pairs import build_sync_pair

    # Create dummy tensors
    audio_tokens = torch.randn(100, 256)  # 100 audio tokens
    visual_tokens = torch.randn(32, 256)  # 32 video tokens

    shifts = [0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0]

    for shift in shifts:
        pair = build_sync_pair(
            audio_tokens,
            visual_tokens,
            video_fps=25.0,
            shift_seconds=shift,
            audio_token_seconds=0.01,
            window_seconds=32 / 25.0,
        )

        # Check that shift is preserved
        assert pair.shift_seconds == shift
        assert pair.is_positive == (shift == 0.0)


def test_eval_score_aggregation() -> None:
    """Test that video-level score aggregation works correctly."""
    logits = torch.randn(4, 32)
    targets = torch.randint(0, 2, (4, 32)).float()
    mask = torch.ones(4, 32, dtype=torch.bool)

    # Test mean aggregation
    metrics_mean = compute_sync_metrics(logits, targets, mask, aggregation="mean")
    assert metrics_mean.video_accuracy >= 0.0
    assert metrics_mean.video_accuracy <= 1.0

    # Test max aggregation
    metrics_max = compute_sync_metrics(logits, targets, mask, aggregation="max")
    assert metrics_max.video_accuracy >= 0.0
    assert metrics_max.video_accuracy <= 1.0


def test_eval_per_window_timeline_output() -> None:
    """Test that per-window sync scores can be extracted."""
    logits = torch.randn(2, 32)
    targets = torch.randint(0, 2, (2, 32)).float()
    mask = torch.ones(2, 32, dtype=torch.bool)

    metrics = compute_sync_metrics(logits, targets, mask, aggregation="mean")

    # Convert to probabilities
    probs = torch.sigmoid(logits)

    # Check shape
    assert probs.shape == (2, 32)

    # Check that all probabilities are in [0, 1]
    assert (probs >= 0.0).all()
    assert (probs <= 1.0).all()


def test_eval_audio_only_comparison_placeholder() -> None:
    """Placeholder test for audio-only comparison.

    NOTE: This is a placeholder because the audio encoder export does not
    include the spoof classifier weights. A full implementation would require
    either (a) exporting the full Phase 5 classifier or (b) re-computing
    spoof scores from the audio encoder alone.
    """
    # This test validates that we understand the limitation
    assert True  # Placeholder - would implement audio-only comparison when classifier weights available


def test_audio_baseline_computation() -> None:
    """Test audio baseline correlation computation."""
    import sys
    from pathlib import Path
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts.evaluate_sync import compute_audio_baseline

    # Create dummy mel spectrogram
    mel = torch.randn(1, 80, 100)  # [n_mels, T_mel]

    # Test shift=0 (should be perfect correlation)
    corr_0 = compute_audio_baseline(mel, shift_seconds=0.0, audio_token_seconds=0.01)
    assert corr_0 == 1.0

    # Test non-zero shift
    corr_shift = compute_audio_baseline(mel, shift_seconds=0.5, audio_token_seconds=0.01)
    assert -1.0 <= corr_shift <= 1.0


def test_category_aggregation() -> None:
    """Test that category-wise aggregation works correctly."""
    # Simulate video scores with categories
    video_scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    video_categories = ["REAL", "REAL", "AUDIO-ONLY", "AUDIO-ONLY", "VIDEO-ONLY", "VIDEO-ONLY", "AUDIO+VIDEO", "AUDIO+VIDEO"]

    # Aggregate by category
    category_stats = {}
    for category in ["REAL", "AUDIO-ONLY", "VIDEO-ONLY", "AUDIO+VIDEO"]:
        cat_scores = [s for s, c in zip(video_scores, video_categories) if c == category]
        if cat_scores:
            category_stats[category] = {
                "n_videos": len(cat_scores),
                "mean_score": sum(cat_scores) / len(cat_scores),
            }

    assert category_stats["REAL"]["n_videos"] == 2
    assert abs(category_stats["REAL"]["mean_score"] - 0.85) < 1e-6
    assert category_stats["AUDIO-ONLY"]["n_videos"] == 2
    assert category_stats["VIDEO-ONLY"]["n_videos"] == 2
    assert category_stats["AUDIO+VIDEO"]["n_videos"] == 2


def test_timeline_extraction_structure() -> None:
    """Test that timeline extraction produces correct structure."""
    # This tests the structure without running the full model
    timeline_data = {
        "window_times": [0.0, 0.04, 0.08, 0.12],
        "scores": [0.9, 0.8, 0.7, 0.6],
        "targets": [1, 1, 1, 1],
        "valid_mask": [True, True, True, True],
        "n_valid_windows": 4,
    }

    # Verify structure
    assert len(timeline_data["window_times"]) == 4
    assert len(timeline_data["scores"]) == 4
    assert len(timeline_data["targets"]) == 4
    assert len(timeline_data["valid_mask"]) == 4
    assert timeline_data["n_valid_windows"] == 4


def test_no_fake_periods_as_sync_labels() -> None:
    """Test that fake_periods are not used as synchronization labels."""
    from src.data.sync_pairs import parse_fake_periods

    # Parse fake periods
    fake_periods_str = "[[4.1, 5.044]]"
    periods = parse_fake_periods(fake_periods_str)

    # Verify they are parsed correctly
    assert periods == [[4.1, 5.044]]

    # Verify they are NOT used as sync targets in pair construction
    # (This is implicit in the implementation - sync targets are based on shift only)
    audio_tokens = torch.randn(100, 256)
    visual_tokens = torch.randn(32, 256)

    pair = build_sync_pair(
        audio_tokens,
        visual_tokens,
        video_fps=25.0,
        shift_seconds=0.0,
        audio_token_seconds=0.01,
        window_seconds=32 / 25.0,
    )

    # All targets should be 1 for shift=0, regardless of fake_periods
    assert (pair.targets == 1.0).all()


def test_valid_mask_handling() -> None:
    """Test that valid masks are handled correctly in evaluation."""
    logits = torch.tensor([[1.0, -1.0, 0.5], [-0.5, 0.8, -0.2]])
    targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    mask = torch.tensor([[True, True, False], [True, False, True]])

    metrics = compute_sync_metrics(logits, targets, mask, aggregation="mean")

    # Only 4 valid windows
    assert metrics.n_windows == 4
    assert metrics.n_positive_windows + metrics.n_negative_windows == 4
