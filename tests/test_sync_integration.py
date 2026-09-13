"""End-to-end integration tests for Phase 11 sync head."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.data.sync_pairs import SyncPairConfig, build_sync_pair, compute_shifted_alignment, create_negative_pair
from src.evaluation.sync_metrics import compute_sync_metrics
from src.losses.sync_loss import SyncLoss
from src.models.fusion.cross_attention import BidirectionalCrossAttention
from src.models.fusion.temporal_align import align_audio_to_video
from src.models.heads.sync_head import SyncHead


class MockAudioEncoder(nn.Module):
    """Mock audio encoder producing tokens."""

    def __init__(self, output_dim: int = 256, token_seconds: float = 0.08) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.token_seconds = token_seconds
        self.proj = nn.Linear(80, output_dim)

    def forward(self, mel: torch.Tensor):
        B, _n_mels, T_mel = mel.shape
        tokens = self.proj(mel.transpose(1, 2)[:, ::4, :])  # downsample ~4x
        from src.models.audio.cnn import AudioEncoderOutput
        return AudioEncoderOutput(tokens=tokens, time_downsample=4)


class MockVisualEncoder(nn.Module):
    """Mock visual encoder producing tokens."""

    def __init__(self, output_dim: int = 256) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.proj = nn.Linear(3, output_dim)

    def forward(self, landmarks: torch.Tensor):
        x = landmarks.mean(dim=2)
        from src.models.video.visual_encoder import VisualEncoderOutput
        return VisualEncoderOutput(tokens=self.proj(x))


def _run_sync_pipeline(
    mel: torch.Tensor,
    landmarks: torch.Tensor,
    *,
    video_fps: float,
    shift_seconds: float,
    audio_token_seconds: float = 0.08,
    window_seconds: float | None = None,
) -> dict[str, torch.Tensor]:
    """Run mock encoders -> alignment -> cross-attention -> sync head -> loss."""
    B, T = landmarks.shape[0], landmarks.shape[1]
    D = 256

    audio_enc = MockAudioEncoder(output_dim=D, token_seconds=audio_token_seconds)
    visual_enc = MockVisualEncoder(output_dim=D)
    cross_attn = BidirectionalCrossAttention(dim=D, num_heads=4, n_layers=1)
    sync_head = SyncHead(input_dim=D)
    loss_fn = SyncLoss()

    audio_out = audio_enc(mel)
    audio_tokens = audio_out.tokens
    visual_out = visual_enc(landmarks)
    visual_tokens = visual_out.tokens

    audio_aligned_list = []
    targets_list = []
    mask_list = []

    for b in range(B):
        pos_aligned, pos_counts = align_audio_to_video(
            audio_tokens[b : b + 1],
            n_video_tokens=T,
            audio_token_seconds=audio_token_seconds,
            video_fps=video_fps,
            window_seconds=window_seconds,
        )
        pair = build_sync_pair(
            audio_tokens[b],
            visual_tokens[b],
            video_fps=video_fps,
            shift_seconds=shift_seconds,
            audio_token_seconds=audio_token_seconds,
            window_seconds=window_seconds,
            positive_audio_aligned=pos_aligned[0],
            positive_bucket_counts=pos_counts[0],
        )
        audio_aligned_list.append(pair.audio_aligned)
        targets_list.append(pair.targets)
        mask_list.append(pair.mask)

    audio_aligned = torch.stack(audio_aligned_list)
    targets = torch.stack(targets_list)
    mask = torch.stack(mask_list)

    fused = cross_attn(audio_aligned, visual_tokens).fused
    logits = sync_head(fused)
    loss = loss_fn(logits, targets, mask)

    return {
        "audio_aligned": audio_aligned,
        "fused": fused,
        "logits": logits,
        "targets": targets,
        "mask": mask,
        "loss": loss,
        "audio_enc": audio_enc,
        "visual_enc": visual_enc,
        "cross_attn": cross_attn,
        "sync_head": sync_head,
    }


def test_end_to_end_positive_pair() -> None:
    """Full pipeline with positive (aligned) pair."""
    B, T = 2, 32
    mel = torch.randn(B, 80, 256, requires_grad=True)
    landmarks = torch.randn(B, T, 68, 3, requires_grad=True)

    out = _run_sync_pipeline(mel, landmarks, video_fps=30.0, shift_seconds=0.0)

    assert out["logits"].shape == (B, T)
    assert torch.all(out["targets"] == 1.0)
    assert out["loss"].item() >= 0
    assert torch.isfinite(out["loss"])

    out["loss"].backward()
    assert out["audio_enc"].proj.weight.grad is not None
    assert out["visual_enc"].proj.weight.grad is not None
    assert out["sync_head"].proj1.weight.grad is not None
    assert out["cross_attn"].fuse_proj.weight.grad is not None
    assert torch.isfinite(out["cross_attn"].fuse_proj.weight.grad).all()


def test_end_to_end_negative_pair() -> None:
    """Full pipeline with negative (shifted) pair."""
    B, T = 2, 32
    mel = torch.randn(B, 80, 256, requires_grad=True)
    landmarks = torch.randn(B, T, 68, 3, requires_grad=True)

    out = _run_sync_pipeline(mel, landmarks, video_fps=30.0, shift_seconds=0.5)

    assert out["logits"].shape == (B, T)
    assert torch.all(out["targets"] == 0.0)
    assert out["loss"].item() >= 0
    assert torch.isfinite(out["loss"])

    out["loss"].backward()
    assert out["cross_attn"].fuse_proj.weight.grad is not None
    assert torch.isfinite(out["cross_attn"].fuse_proj.weight.grad).all()


def test_positive_and_negative_produce_different_representations() -> None:
    """Shifted-negative pairs must reach the model as different aligned inputs."""
    torch.manual_seed(0)
    B, T = 1, 32
    mel = torch.randn(B, 80, 256)
    landmarks = torch.randn(B, T, 68, 3)
    fps = 30.0

    pos = _run_sync_pipeline(mel, landmarks, video_fps=fps, shift_seconds=0.0)
    neg = _run_sync_pipeline(mel, landmarks, video_fps=fps, shift_seconds=0.5)

    assert not torch.allclose(pos["audio_aligned"], neg["audio_aligned"], atol=1e-5)
    assert not torch.allclose(pos["fused"], neg["fused"], atol=1e-5)


def test_shifted_alignment_matches_phase9_at_zero_shift() -> None:
    """Unshifted compute_shifted_alignment must match Phase 9 align_audio_to_video."""
    T_a, T_v, D = 64, 32, 256
    audio_tokens = torch.randn(1, T_a, D, dtype=torch.float64)
    fps = 30.0
    dt_a = 0.08

    phase9_aligned, phase9_counts = align_audio_to_video(
        audio_tokens,
        n_video_tokens=T_v,
        audio_token_seconds=dt_a,
        video_fps=fps,
    )
    shifted_aligned, shifted_counts = compute_shifted_alignment(
        audio_tokens[0],
        n_video_tokens=T_v,
        video_fps=fps,
        shift_seconds=0.0,
        audio_token_seconds=dt_a,
    )

    torch.testing.assert_close(phase9_aligned[0], shifted_aligned, atol=1e-9, rtol=0)
    torch.testing.assert_close(phase9_counts[0], shifted_counts)


def test_shifted_alignment_differs_for_nonzero_shifts() -> None:
    """Verify shifted alignment differs from positive for ±0.5s shifts."""
    T_a, T_v, D = 64, 32, 256
    audio_tokens = torch.randn(T_a, D)
    video_fps = 30.0

    aligned_pos, _ = compute_shifted_alignment(
        audio_tokens, n_video_tokens=T_v, video_fps=video_fps, shift_seconds=0.0
    )
    for shift in (0.5, -0.5):
        aligned_shift, counts = compute_shifted_alignment(
            audio_tokens,
            n_video_tokens=T_v,
            video_fps=video_fps,
            shift_seconds=shift,
        )
        assert not torch.allclose(aligned_pos, aligned_shift, atol=1e-5)
        assert aligned_shift.shape == (T_v, D)
        assert counts.shape == (T_v,)


def test_negative_pair_masks_empty_buckets() -> None:
    """Large shifts should mask video tokens with no valid audio correspondence."""
    T_a, T_v, D = 16, 32, 256
    audio_tokens = torch.randn(T_a, D)
    visual_tokens = torch.randn(T_v, D)

    pair = create_negative_pair(
        audio_tokens=audio_tokens,
        visual_tokens=visual_tokens,
        shift_seconds=2.0,
        video_fps=25.0,
    )
    assert pair.mask.sum() < T_v
    assert torch.all(pair.targets[pair.mask] == 0.0)


def test_metrics_computation() -> None:
    """Test metrics computation on synthetic data."""
    B, T = 4, 32
    logits = torch.randn(B, T)
    targets = torch.randint(0, 2, (B, T), dtype=torch.float32)

    metrics = compute_sync_metrics(logits, targets)

    assert 0 <= metrics.accuracy <= 1
    assert metrics.n_samples == B
    assert metrics.n_windows == B * T


def test_mask_handling() -> None:
    """Test that masks are properly handled in loss and metrics."""
    B, T = 2, 32
    logits = torch.randn(B, T)
    targets = torch.randint(0, 2, (B, T), dtype=torch.float32)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, 16:] = False

    loss_fn = SyncLoss()
    loss = loss_fn(logits, targets, mask)
    assert torch.isfinite(loss)

    metrics = compute_sync_metrics(logits, targets, mask)
    assert metrics.n_windows == 32


def test_end_to_end_with_sync_trainer() -> None:
    """Test full pipeline with SyncTrainer using mock encoders."""
    from src.config import TrainingConfig
    from src.training.sync_trainer import SyncModel, SyncTrainer
    from src.training.utils import RunDirectory
    import tempfile
    import shutil

    B, T = 2, 32
    D = 256

    # Create model components
    audio_enc = MockAudioEncoder(output_dim=D, token_seconds=0.08)
    visual_enc = MockVisualEncoder(output_dim=D)
    cross_attn = BidirectionalCrossAttention(dim=D, num_heads=4, n_layers=1)
    sync_head = SyncHead(input_dim=D)

    model = SyncModel(audio_enc, visual_enc, cross_attn, sync_head)
    sync_loss = SyncLoss()
    sync_pair_config = SyncPairConfig()

    # Create temporary run directory
    tmpdir = tempfile.mkdtemp()
    try:
        run_dir = RunDirectory(tmpdir, "test-sync-trainer-integration", create=True)

        training_cfg = TrainingConfig(
            batch_size=8,
            learning_rate=1e-4,
            weight_decay=0.0,
            epochs=1,
            num_workers=0,
            amp=False,
            grad_accum_steps=1,
            grad_clip_norm=5.0,
            monitor="val_loss",
            monitor_mode="min",
            early_stopping_patience=5,
        )

        trainer = SyncTrainer(
            model=model,
            optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
            config=training_cfg,
            run_dir=run_dir,
            device="cpu",
            sync_loss=sync_loss,
            sync_pair_config=sync_pair_config,
            audio_token_seconds=0.08,
            aggregation="mean",
        )

        # Test with a batch
        batch = {
            "mel_window": torch.randn(B, 80, 256),
            "landmarks": torch.randn(B, T, 68, 3),
            "fps": 25.0,
            "window_seconds": 1.28,
            "shift_seconds": 0.0,
        }

        output = trainer.compute_loss(batch)

        assert output["loss"].item() >= 0
        assert torch.isfinite(output["loss"])
        assert output["logits"].shape == (B, T)
        assert output["targets"].shape == (B, T)
        assert output["mask"].shape == (B, T)

        # Test backward pass
        output["loss"].backward()

        # Verify encoders are frozen
        assert not model.audio_encoder.proj.weight.requires_grad
        assert not model.visual_encoder.proj.weight.requires_grad

        # Verify trainable components have gradients
        assert model.cross_attention.fuse_proj.weight.grad is not None
        assert model.sync_head.proj1.weight.grad is not None
    finally:
        # Close log handlers before cleanup
        import logging
        for handler in logging.getLogger("syncguard").handlers[:]:
            handler.close()
            logging.getLogger("syncguard").removeHandler(handler)
        shutil.rmtree(tmpdir, ignore_errors=True)
