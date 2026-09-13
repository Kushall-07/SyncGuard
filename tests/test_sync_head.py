"""Unit tests for Phase 11 sync head, loss, and temporal target construction."""

from __future__ import annotations

import pytest
import torch

from src.data.sync_pairs import (
    SyncPairConfig,
    build_sync_pair,
    compute_shifted_alignment,
    compute_shifted_targets,
    create_negative_pair,
    create_positive_pair,
    parse_fake_periods,
    sample_shift_seconds,
)
from src.models.fusion.temporal_align import align_audio_to_video
from src.evaluation.sync_metrics import SyncMetrics, compute_sync_metrics
from src.losses.sync_loss import SyncLoss, SyncLossConfig
from src.models.heads.sync_head import SyncHead, SyncHeadConfig


# --------------------------------------------------------------------- SyncHead tests


def test_sync_head_shape() -> None:
    """Test SyncHead produces correct output shape."""
    head = SyncHead(input_dim=256, hidden_dim=128)
    fused = torch.randn(4, 32, 256)  # [B, T, D]
    logits = head(fused)
    assert logits.shape == (4, 32)  # [B, T]


def test_sync_head_different_sequence_lengths() -> None:
    """Test SyncHead handles different sequence lengths."""
    head = SyncHead(input_dim=256)
    for T in [16, 32, 64, 128]:
        fused = torch.randn(2, T, 256)
        logits = head(fused)
        assert logits.shape == (2, T)


def test_sync_head_aggregation_mean() -> None:
    """Test mean aggregation."""
    head = SyncHead(input_dim=256, aggregation="mean")
    fused = torch.randn(2, 32, 256)
    logits = head(fused)
    mask = torch.ones(2, 32, dtype=torch.bool)
    aggregated = head.aggregate(logits, mask)
    assert aggregated.shape == (2,)


def test_sync_head_aggregation_max() -> None:
    """Test max aggregation."""
    head = SyncHead(input_dim=256, aggregation="max")
    fused = torch.randn(2, 32, 256)
    logits = head(fused)
    mask = torch.ones(2, 32, dtype=torch.bool)
    aggregated = head.aggregate(logits, mask)
    assert aggregated.shape == (2,)


def test_sync_head_aggregation_median() -> None:
    """Test median aggregation."""
    head = SyncHead(input_dim=256, aggregation="median")
    fused = torch.randn(2, 32, 256)
    logits = head(fused)
    mask = torch.ones(2, 32, dtype=torch.bool)
    aggregated = head.aggregate(logits, mask)
    assert aggregated.shape == (2,)


def test_sync_head_aggregation_with_mask() -> None:
    """Test aggregation respects mask."""
    head = SyncHead(input_dim=256, aggregation="mean")
    fused = torch.randn(2, 32, 256)
    logits = head(fused)
    mask = torch.ones(2, 32, dtype=torch.bool)
    mask[:, 16:] = False  # Mask out second half
    aggregated = head.aggregate(logits, mask)
    assert aggregated.shape == (2,)


def test_sync_head_from_config() -> None:
    """Test SyncHead creation from config."""
    cfg = SyncHeadConfig(hidden_dim=64, dropout=0.2, aggregation="max")
    head = SyncHead.from_config(cfg, input_dim=256)
    assert head.hidden_dim == 64
    assert head.aggregation == "max"


def test_sync_head_invalid_input_dim() -> None:
    """Test SyncHead rejects invalid input dimension."""
    head = SyncHead(input_dim=256)
    fused = torch.randn(2, 32, 128)  # Wrong dim
    with pytest.raises(ValueError, match="input last dim must be 256"):
        head(fused)


def test_sync_head_invalid_input_shape() -> None:
    """Test SyncHead rejects invalid input shape."""
    head = SyncHead(input_dim=256)
    fused = torch.randn(2, 256)  # Missing time dimension
    with pytest.raises(ValueError, match="expected \\[B, T, D\\]"):
        head(fused)


# --------------------------------------------------------------------- SyncLoss tests


def test_sync_loss_basic() -> None:
    """Test basic sync loss computation."""
    loss_fn = SyncLoss()
    logits = torch.randn(4, 32)
    targets = torch.randint(0, 2, (4, 32), dtype=torch.float32)
    loss = loss_fn(logits, targets)
    assert loss.item() >= 0
    assert not torch.isnan(loss)


def test_sync_loss_with_mask() -> None:
    """Test sync loss with valid mask."""
    loss_fn = SyncLoss()
    logits = torch.randn(4, 32)
    targets = torch.randint(0, 2, (4, 32), dtype=torch.float32)
    mask = torch.ones(4, 32, dtype=torch.bool)
    mask[:, 16:] = False
    loss = loss_fn(logits, targets, mask)
    assert loss.item() >= 0
    assert not torch.isnan(loss)


def test_sync_loss_fully_masked_no_nan() -> None:
    """Test sync loss doesn't produce NaN for fully masked samples."""
    loss_fn = SyncLoss()
    logits = torch.randn(4, 32)
    targets = torch.randint(0, 2, (4, 32), dtype=torch.float32)
    mask = torch.zeros(4, 32, dtype=torch.bool)  # Fully masked
    loss = loss_fn(logits, targets, mask)
    # Should not be NaN
    assert not torch.isnan(loss)


def test_sync_loss_reduction_none() -> None:
    """Test sync loss with reduction='none'."""
    loss_fn = SyncLoss(reduction="none")
    logits = torch.randn(4, 32)
    targets = torch.randint(0, 2, (4, 32), dtype=torch.float32)
    loss = loss_fn(logits, targets)
    assert loss.shape == (4,)  # Per-sample loss


def test_sync_loss_reduction_sum() -> None:
    """Test sync loss with reduction='sum'."""
    loss_fn = SyncLoss(reduction="sum")
    logits = torch.randn(4, 32)
    targets = torch.randint(0, 2, (4, 32), dtype=torch.float32)
    loss = loss_fn(logits, targets)
    assert loss.dim() == 0  # Scalar


def test_sync_loss_pos_weight() -> None:
    """Test sync loss with positive class weighting."""
    loss_fn = SyncLoss(pos_weight=2.0)
    logits = torch.randn(4, 32)
    targets = torch.randint(0, 2, (4, 32), dtype=torch.float32)
    loss = loss_fn(logits, targets)
    assert loss.item() >= 0


def test_sync_loss_from_config() -> None:
    """Test SyncLoss creation from config."""
    cfg = SyncLossConfig(pos_weight=1.5, reduction="sum")
    loss_fn = SyncLoss.from_config(cfg)
    assert loss_fn.pos_weight == 1.5
    assert loss_fn.reduction == "sum"


def test_sync_loss_shape_mismatch() -> None:
    """Test sync loss rejects shape mismatch."""
    loss_fn = SyncLoss()
    logits = torch.randn(4, 32)
    targets = torch.randn(4, 16)  # Wrong shape
    with pytest.raises(ValueError, match="must match"):
        loss_fn(logits, targets)


def test_sync_loss_mask_shape_mismatch() -> None:
    """Test sync loss rejects mask shape mismatch."""
    loss_fn = SyncLoss()
    logits = torch.randn(4, 32)
    targets = torch.randn(4, 32)
    mask = torch.ones(4, 16, dtype=torch.bool)  # Wrong shape
    with pytest.raises(ValueError, match="must match"):
        loss_fn(logits, targets, mask)


# --------------------------------------------------------- Temporal target tests


def test_positive_target_construction() -> None:
    """Test positive pair construction."""
    audio_aligned = torch.randn(32, 256)
    visual_tokens = torch.randn(32, 256)
    pair = create_positive_pair(audio_aligned, visual_tokens)
    assert pair.is_positive
    assert pair.shift_seconds == 0.0
    assert torch.all(pair.targets == 1.0)
    assert torch.all(pair.mask)


def test_positive_target_with_bucket_counts() -> None:
    """Test positive pair with bucket counts mask."""
    audio_aligned = torch.randn(32, 256)
    visual_tokens = torch.randn(32, 256)
    bucket_counts = torch.ones(32)
    bucket_counts[16:] = 0  # Invalid positions
    pair = create_positive_pair(audio_aligned, visual_tokens, bucket_counts)
    assert torch.all(pair.targets[:16] == 1.0)
    assert torch.all(pair.mask[:16])
    assert not torch.any(pair.mask[16:])


def test_negative_target_construction_small_shift() -> None:
    """Test negative pair with small non-zero shift (now always negative)."""
    audio_tokens = torch.randn(64, 256)
    visual_tokens = torch.randn(32, 256)
    # Shift of 0.03s (non-zero = negative)
    pair = create_negative_pair(
        audio_tokens, visual_tokens, shift_seconds=0.03, video_fps=25.0
    )
    assert not pair.is_positive
    assert pair.shift_seconds == 0.03
    # Any non-zero shift is now negative
    assert torch.all(pair.targets == 0.0)


def test_negative_target_construction_large_shift() -> None:
    """Test negative pair with large shift."""
    audio_tokens = torch.randn(64, 256)
    visual_tokens = torch.randn(32, 256)
    # Shift of 1.0s
    pair = create_negative_pair(
        audio_tokens, visual_tokens, shift_seconds=1.0, video_fps=25.0
    )
    assert not pair.is_positive
    assert pair.shift_seconds == 1.0
    # Large shift is negative
    assert torch.all(pair.targets == 0.0)


def test_negative_target_zero_shift() -> None:
    """Test negative pair with zero shift (should be positive target)."""
    audio_tokens = torch.randn(64, 256)
    visual_tokens = torch.randn(32, 256)
    pair = create_negative_pair(
        audio_tokens, visual_tokens, shift_seconds=0.0, video_fps=25.0
    )
    # Zero shift is treated as aligned (target=1)
    assert torch.all(pair.targets == 1.0)


def test_negative_target_edge_shift() -> None:
    """Test negative pair with edge case shift."""
    audio_tokens = torch.randn(64, 256)
    visual_tokens = torch.randn(32, 256)
    # Shift of 0.04s (non-zero = negative)
    pair = create_negative_pair(
        audio_tokens, visual_tokens, shift_seconds=0.04, video_fps=25.0
    )
    # Any non-zero shift is negative
    assert torch.all(pair.targets == 0.0)


def test_compute_shifted_targets_different_lengths() -> None:
    """Test target computation for different sequence lengths."""
    for T in [16, 32, 64]:
        targets, mask = compute_shifted_targets(
            n_video_tokens=T, video_fps=25.0, shift_seconds=1.0
        )
        assert targets.shape == (T,)
        assert mask.shape == (T,)


def test_compute_shifted_targets_invalid_fps() -> None:
    """Test target computation rejects invalid fps."""
    with pytest.raises(ValueError, match="video_fps must be positive"):
        compute_shifted_targets(n_video_tokens=32, video_fps=0.0, shift_seconds=1.0)


def test_compute_shifted_targets_invalid_tokens() -> None:
    """Test target computation rejects invalid token count."""
    with pytest.raises(ValueError, match="n_video_tokens must be positive"):
        compute_shifted_targets(n_video_tokens=0, video_fps=25.0, shift_seconds=1.0)


# ----------------------------------------------------------- fake_periods tests


def test_parse_fake_periods_single() -> None:
    """Test parsing single fake period."""
    periods = parse_fake_periods("[[4.1, 5.044]]")
    assert periods == [[4.1, 5.044]]


def test_parse_fake_periods_multiple() -> None:
    """Test parsing multiple fake periods."""
    periods = parse_fake_periods("[[2.8, 3.464], [4.964, 6.048]]")
    assert periods == [[2.8, 3.464], [4.964, 6.048]]


def test_parse_fake_periods_empty() -> None:
    """Test parsing empty fake periods."""
    periods = parse_fake_periods("[]")
    assert periods == []


def test_parse_fake_periods_none() -> None:
    """Test parsing None/empty string fake periods."""
    periods = parse_fake_periods("")
    assert periods == []


def test_parse_fake_periods_invalid_json() -> None:
    """Test parsing invalid JSON."""
    with pytest.raises(ValueError, match="Invalid JSON"):
        parse_fake_periods("not json")


def test_parse_fake_periods_invalid_format() -> None:
    """Test parsing invalid format (valid JSON but not a list)."""
    with pytest.raises(ValueError, match="fake_periods must be a list"):
        parse_fake_periods('{"not": "a list"}')


def test_parse_fake_periods_invalid_period() -> None:
    """Test parsing invalid period format."""
    with pytest.raises(ValueError, match="each fake_period must be"):
        parse_fake_periods("[[1.0]]")


def test_parse_fake_periods_not_sync_labels() -> None:
    """Test that fake_periods are not used as sync labels."""
    # This is a documentation test to ensure we don't conflate
    # manipulation with desynchronization
    periods = parse_fake_periods("[[4.1, 5.044]]")
    # The function only parses, doesn't assign sync labels
    assert isinstance(periods, list)
    # No sync labels are generated here
    assert all(isinstance(p, list) for p in periods)


# ----------------------------------------------------------- SyncMetrics tests


def test_sync_metrics_computation() -> None:
    """Test basic sync metrics computation."""
    logits = torch.randn(4, 32)
    targets = torch.randint(0, 2, (4, 32), dtype=torch.float32)
    metrics = compute_sync_metrics(logits, targets)
    assert isinstance(metrics, SyncMetrics)
    assert 0 <= metrics.accuracy <= 1
    assert 0 <= metrics.precision <= 1
    assert 0 <= metrics.recall <= 1
    assert 0 <= metrics.f1 <= 1
    assert metrics.n_samples == 4
    assert metrics.n_windows == 128


def test_sync_metrics_with_mask() -> None:
    """Test sync metrics with mask."""
    logits = torch.randn(4, 32)
    targets = torch.randint(0, 2, (4, 32), dtype=torch.float32)
    mask = torch.ones(4, 32, dtype=torch.bool)
    mask[:, 16:] = False
    metrics = compute_sync_metrics(logits, targets, mask)
    assert metrics.n_windows == 64  # Only half are valid


def test_sync_metrics_empty_mask() -> None:
    """Test sync metrics with fully masked input."""
    logits = torch.randn(4, 32)
    targets = torch.randint(0, 2, (4, 32), dtype=torch.float32)
    mask = torch.zeros(4, 32, dtype=torch.bool)
    metrics = compute_sync_metrics(logits, targets, mask)
    assert metrics.n_windows == 0
    assert metrics.accuracy == 0.0


def test_sync_metrics_video_aggregation_mean() -> None:
    """Test video-level metrics with mean aggregation."""
    logits = torch.randn(4, 32)
    targets = torch.randint(0, 2, (4, 32), dtype=torch.float32)
    metrics = compute_sync_metrics(logits, targets, aggregation="mean")
    assert 0 <= metrics.video_accuracy <= 1


def test_sync_metrics_video_aggregation_max() -> None:
    """Test video-level metrics with max aggregation."""
    logits = torch.randn(4, 32)
    targets = torch.randint(0, 2, (4, 32), dtype=torch.float32)
    metrics = compute_sync_metrics(logits, targets, aggregation="max")
    assert 0 <= metrics.video_accuracy <= 1


def test_sync_metrics_video_aggregation_median() -> None:
    """Test video-level metrics with median aggregation."""
    logits = torch.randn(4, 32)
    targets = torch.randint(0, 2, (4, 32), dtype=torch.float32)
    metrics = compute_sync_metrics(logits, targets, aggregation="median")
    assert 0 <= metrics.video_accuracy <= 1


def test_sync_metrics_shape_mismatch() -> None:
    """Test sync metrics rejects shape mismatch."""
    logits = torch.randn(4, 32)
    targets = torch.randn(4, 16)  # Wrong shape
    with pytest.raises(ValueError, match="must match"):
        compute_sync_metrics(logits, targets)


def test_sync_metrics_mask_shape_mismatch() -> None:
    """Test sync metrics rejects mask shape mismatch."""
    logits = torch.randn(4, 32)
    targets = torch.randn(4, 32)
    mask = torch.ones(4, 16, dtype=torch.bool)  # Wrong shape
    with pytest.raises(ValueError, match="must match"):
        compute_sync_metrics(logits, targets, mask)


# --------------------------------------------------------- Config validation tests


def test_sync_head_config_validation() -> None:
    """Test SyncHeadConfig validation."""
    with pytest.raises(ValueError, match="hidden_dim must be positive"):
        SyncHeadConfig(hidden_dim=0)
    with pytest.raises(ValueError, match="dropout must be between 0 and 1"):
        SyncHeadConfig(dropout=1.5)
    with pytest.raises(ValueError, match="aggregation must be one of"):
        SyncHeadConfig(aggregation="invalid")


def test_sync_loss_config_validation() -> None:
    """Test SyncLossConfig validation."""
    with pytest.raises(ValueError, match="pos_weight must be positive"):
        SyncLossConfig(pos_weight=-1.0)
    with pytest.raises(ValueError, match="reduction must be one of"):
        SyncLossConfig(reduction="invalid")


def test_sync_pair_config_validation() -> None:
    """Test SyncPairConfig validation."""
    with pytest.raises(ValueError, match="shift_seconds must not be empty"):
        SyncPairConfig(shift_seconds=[])
    with pytest.raises(ValueError, match="audio_token_seconds must be positive"):
        SyncPairConfig(audio_token_seconds=0.0)


# --------------------------------------------------------- Shift alignment tests


def test_shifted_alignment_zero_matches_phase9() -> None:
    """At shift=0, compute_shifted_alignment matches Phase 9 align_audio_to_video."""
    T_a, T_v, D = 64, 32, 256
    audio = torch.randn(1, T_a, D, dtype=torch.float64)
    fps = 25.0
    dt_a = 0.08

    p9_out, p9_counts = align_audio_to_video(
        audio, n_video_tokens=T_v, audio_token_seconds=dt_a, video_fps=fps
    )
    sp_out, sp_counts = compute_shifted_alignment(
        audio[0], n_video_tokens=T_v, video_fps=fps, shift_seconds=0.0, audio_token_seconds=dt_a
    )
    torch.testing.assert_close(p9_out[0], sp_out, atol=1e-9, rtol=0)
    torch.testing.assert_close(p9_counts[0], sp_counts)


def test_shifted_alignment_differs_for_positive_and_negative() -> None:
    """Shifts of +0.5 and -0.5 produce different alignment than shift=0."""
    T_a, T_v, D = 64, 32, 256
    audio_tokens = torch.randn(T_a, D)
    fps = 25.0

    aligned_zero, _ = compute_shifted_alignment(
        audio_tokens, n_video_tokens=T_v, video_fps=fps, shift_seconds=0.0
    )
    for shift in (0.5, -0.5):
        aligned_shift, counts = compute_shifted_alignment(
            audio_tokens, n_video_tokens=T_v, video_fps=fps, shift_seconds=shift
        )
        assert not torch.allclose(aligned_zero, aligned_shift, atol=1e-5)
        assert (counts > 0).any()


def test_build_sync_pair_positive_uses_provided_alignment() -> None:
    """build_sync_pair reuses Phase-9 alignment for positive pairs."""
    T_a, T_v, D = 64, 32, 256
    audio_tokens = torch.randn(T_a, D)
    visual_tokens = torch.randn(T_v, D)
    provided = torch.randn(T_v, D)
    counts = torch.ones(T_v, dtype=torch.int64)

    pair = build_sync_pair(
        audio_tokens,
        visual_tokens,
        video_fps=25.0,
        shift_seconds=0.0,
        positive_audio_aligned=provided,
        positive_bucket_counts=counts,
    )
    assert pair.is_positive
    assert torch.allclose(pair.audio_aligned, provided)
    assert torch.all(pair.targets == 1.0)


def test_build_sync_pair_negative_shifts_alignment() -> None:
    """build_sync_pair recomputes alignment for negative pairs."""
    T_a, T_v, D = 64, 32, 256
    audio_tokens = torch.randn(T_a, D)
    visual_tokens = torch.randn(T_v, D)
    provided = torch.randn(T_v, D)

    pair = build_sync_pair(
        audio_tokens,
        visual_tokens,
        video_fps=25.0,
        shift_seconds=0.5,
        positive_audio_aligned=provided,
    )
    assert not pair.is_positive
    assert not torch.allclose(pair.audio_aligned, provided, atol=1e-5)
    assert torch.all(pair.targets == 0.0)


def test_sample_shift_seconds_excludes_zero_by_default() -> None:
    """sample_shift_seconds never returns 0 unless include_positive=True."""
    cfg = SyncPairConfig(shift_seconds=[-0.5, 0.0, 0.5])
    rng = __import__("random").Random(0)
    for _ in range(20):
        shift = sample_shift_seconds(cfg, rng, include_positive=False)
        assert shift != 0.0
