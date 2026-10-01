"""Unit tests for PhysicalSyncModel / PhysicalSyncTrainer (physical-shift training)."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config import TrainingConfig
from src.losses.sync_loss import SyncLoss
from src.models.fusion.cross_attention import BidirectionalCrossAttention
from src.models.heads.sync_head import SyncHead
from src.training.physical_sync_trainer import PhysicalSyncModel, PhysicalSyncTrainer
from src.training.utils import RunDirectory


class MockAudioEncoder(nn.Module):
    def __init__(self, output_dim: int = 256) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.proj = nn.Linear(80, output_dim)

    def forward(self, mel: torch.Tensor):
        from src.models.audio.cnn import AudioEncoderOutput

        tokens = self.proj(mel.transpose(1, 2)[:, ::4, :])
        return AudioEncoderOutput(tokens=tokens, time_downsample=4)


class MockVisualEncoder(nn.Module):
    def __init__(self, output_dim: int = 256) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.proj = nn.Linear(3, output_dim)

    def forward(self, landmarks: torch.Tensor):
        from src.models.video.visual_encoder import VisualEncoderOutput

        return VisualEncoderOutput(tokens=self.proj(landmarks.mean(dim=2)))


def _build_model(D: int = 256) -> PhysicalSyncModel:
    return PhysicalSyncModel(
        MockAudioEncoder(D), MockVisualEncoder(D),
        BidirectionalCrossAttention(dim=D, num_heads=4, n_layers=1),
        SyncHead(input_dim=D),
    )


def test_forward_shapes() -> None:
    model = _build_model()
    B, T = 3, 32
    mel = torch.randn(B, 80, 256)
    landmarks = torch.randn(B, T, 68, 3)
    logits, mask = model(mel, landmarks, fps=25.0, window_seconds=1.28, audio_token_seconds=0.01)
    assert logits.shape == (B, T)
    assert mask.shape == (B, T)
    assert mask.dtype == torch.bool


def test_per_sample_fps_and_window_seconds() -> None:
    model = _build_model()
    B, T = 2, 32
    mel = torch.randn(B, 80, 256)
    landmarks = torch.randn(B, T, 68, 3)
    logits, mask = model(
        mel, landmarks, fps=[25.0, 30.0], window_seconds=[1.28, 32 / 30.0], audio_token_seconds=0.01
    )
    assert logits.shape == (B, T)
    assert mask.any()


def test_no_shift_seconds_argument_in_forward_signature() -> None:
    """The physical model's forward has no `shift_seconds` parameter at all -
    the temporal relationship is already baked into `mel` upstream by the
    physical-shift dataset, not recomputed via token-timeline reassignment.
    """
    import inspect

    params = inspect.signature(PhysicalSyncModel.forward).parameters
    assert "shift_seconds" not in params


def test_trainer_compute_loss_and_metrics(tmp_path) -> None:
    model = _build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    run_dir = RunDirectory(tmp_path, "physical-sync-test")
    cfg = TrainingConfig(batch_size=2, epochs=1)

    trainer = PhysicalSyncTrainer(
        model, SyncLoss(), optimizer=optimizer, config=cfg, run_dir=run_dir, device="cpu",
        audio_token_seconds=0.01, aggregation="mean",
    )

    B, T = 4, 32
    batch = {
        "mel_window": torch.randn(B, 80, 256),
        "landmarks": torch.randn(B, T, 68, 3),
        "fps": [25.0] * B,
        "window_seconds": [1.28] * B,
        "label": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        "negative_type": ["none", "physical_shift", "none", "cross_clip"],
        "shift_seconds": [0.0, 0.5, 0.0, 0.0],
    }
    out = trainer.compute_loss(batch)
    assert torch.isfinite(out["loss"])
    assert out["targets"].shape == (B, T)
    assert torch.all(out["targets"][0] == 1.0)
    assert torch.all(out["targets"][1] == 0.0)

    metrics = trainer.compute_metrics([{k: v for k, v in out.items() if k != "loss"}])
    assert "sync_accuracy" in metrics
    assert "video_sync_auc" in metrics


def test_encoders_frozen_after_trainer_init(tmp_path) -> None:
    model = _build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    run_dir = RunDirectory(tmp_path, "physical-sync-freeze-test")
    cfg = TrainingConfig(batch_size=2, epochs=1)
    trainer = PhysicalSyncTrainer(
        model, SyncLoss(), optimizer=optimizer, config=cfg, run_dir=run_dir, device="cpu",
    )
    assert all(not p.requires_grad for p in trainer.model.audio_encoder.parameters())
    assert all(not p.requires_grad for p in trainer.model.visual_encoder.parameters())
    assert any(p.requires_grad for p in trainer.model.cross_attention.parameters())
    assert any(p.requires_grad for p in trainer.model.sync_head.parameters())
