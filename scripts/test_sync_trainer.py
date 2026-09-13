"""Phase 11 smoke test: sync trainer with mock encoders.

Run from the repository root:

    python scripts/test_sync_trainer.py

Tests the SyncTrainer with mock encoders to verify:
- Encoder freezing
- Trainable parameter verification
- Forward/backward pass
- Loss computation
- Metrics computation
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.sync_pairs import SyncPairConfig
from src.losses.sync_loss import SyncLoss
from src.models.fusion.cross_attention import BidirectionalCrossAttention
from src.models.heads.sync_head import SyncHead
from src.training.sync_trainer import SyncModel, SyncTrainer
from src.training.utils import RunDirectory


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


class MockAudioEncoder(nn.Module):
    """Mock audio encoder for testing."""

    def __init__(self, output_dim: int = 256) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.proj = nn.Linear(80, output_dim)

    def forward(self, mel: torch.Tensor):
        B, n_mels, T_mel = mel.shape
        tokens = self.proj(mel.transpose(1, 2)[:, ::4, :])  # downsample ~4x
        from src.models.audio.cnn import AudioEncoderOutput
        return AudioEncoderOutput(tokens=tokens, time_downsample=4)


class MockVisualEncoder(nn.Module):
    """Mock visual encoder for testing."""

    def __init__(self, output_dim: int = 256) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.proj = nn.Linear(3, output_dim)

    def forward(self, landmarks: torch.Tensor):
        x = landmarks.mean(dim=2)
        from src.models.video.visual_encoder import VisualEncoderOutput
        return VisualEncoderOutput(tokens=self.proj(x))


class MockSyncDataset(torch.utils.data.Dataset):
    """Mock dataset for sync training."""

    def __init__(self, n_samples: int = 16, n_frames: int = 32) -> None:
        self.n_samples = n_samples
        self.n_frames = n_frames

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        return {
            "mel_window": torch.randn(80, 256),
            "landmarks": torch.randn(self.n_frames, 68, 3),
            "fps": 25.0,
            "window_seconds": 1.28,
            "shift_seconds": 0.0 if idx % 2 == 0 else 0.5,  # Alternate positive/negative
        }


def main() -> int:
    print("Phase 11 Sync Trainer smoke test")

    D = 256
    T = 32

    # Create mock encoders
    audio_enc = MockAudioEncoder(output_dim=D)
    visual_enc = MockVisualEncoder(output_dim=D)
    cross_attn = BidirectionalCrossAttention(dim=D, num_heads=4, n_layers=1)
    sync_head = SyncHead(input_dim=D)

    # Create sync model
    model = SyncModel(audio_enc, visual_enc, cross_attn, sync_head)

    # Create sync loss
    sync_loss = SyncLoss()
    sync_pair_config = SyncPairConfig()

    # Create temporary run directory
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = RunDirectory(tmpdir, "test-sync-trainer", create=True)

        # Create trainer
        from src.config import TrainingConfig
        training_cfg = TrainingConfig(
            batch_size=8,
            learning_rate=1e-4,
            weight_decay=0.0,
            epochs=2,
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

        print("✓ Trainer initialized")

        # Test encoder freezing
        for param in model.audio_encoder.parameters():
            _check(not param.requires_grad, "Audio encoder should be frozen")
        for param in model.visual_encoder.parameters():
            _check(not param.requires_grad, "Visual encoder should be frozen")
        for param in model.cross_attention.parameters():
            _check(param.requires_grad, "Cross-attention should be trainable")
        for param in model.sync_head.parameters():
            _check(param.requires_grad, "Sync head should be trainable")

        print("✓ Encoders frozen correctly")

        # Test forward pass
        batch = {
            "mel_window": torch.randn(2, 80, 256),
            "landmarks": torch.randn(2, T, 68, 3),
            "fps": 25.0,
            "window_seconds": 1.28,
            "shift_seconds": 0.0,
        }

        output = trainer.compute_loss(batch)
        _check("loss" in output, "Output should contain loss")
        _check("logits" in output, "Output should contain logits")
        _check("targets" in output, "Output should contain targets")
        _check("mask" in output, "Output should contain mask")
        _check(output["loss"].item() >= 0, "Loss should be non-negative")
        _check(torch.isfinite(output["loss"]), "Loss should be finite")

        print("✓ Forward pass successful")

        # Test backward pass
        output["loss"].backward()
        _check(model.cross_attention.fuse_proj.weight.grad is not None, "Cross-attention should have gradients")
        _check(model.sync_head.proj1.weight.grad is not None, "Sync head should have gradients")
        _check(torch.isfinite(model.cross_attention.fuse_proj.weight.grad).all(), "Cross-attention gradients should be finite")
        _check(torch.isfinite(model.sync_head.proj1.weight.grad).all(), "Sync head gradients should be finite")
        _check(model.audio_encoder.proj.weight.grad is None, "Audio encoder should not have gradients")
        _check(model.visual_encoder.proj.weight.grad is None, "Visual encoder should not have gradients")

        print("✓ Backward pass successful")

        # Test metrics computation
        step_outputs = [
            {
                "logits": torch.randn(4, T),
                "targets": torch.randint(0, 2, (4, T), dtype=torch.float32),
                "mask": torch.ones(4, T, dtype=torch.bool),
            }
            for _ in range(5)
        ]

        metrics = trainer.compute_metrics(step_outputs)
        _check("sync_accuracy" in metrics, "Metrics should contain sync_accuracy")
        _check("sync_precision" in metrics, "Metrics should contain sync_precision")
        _check("sync_recall" in metrics, "Metrics should contain sync_recall")
        _check("sync_f1" in metrics, "Metrics should contain sync_f1")
        _check(0 <= metrics["sync_accuracy"] <= 1, "Accuracy should be in [0, 1]")

        print("✓ Metrics computation successful")

        # Test with mock dataloaders
        train_dataset = MockSyncDataset(n_samples=16, n_frames=T)
        dev_dataset = MockSyncDataset(n_samples=8, n_frames=T)

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=4, shuffle=False)

        # Run a few training steps
        trainer.fit(train_loader, dev_loader, epochs=2)

        print("✓ Training loop successful")

        # Check artifacts
        _check(run_dir.config_path.is_file(), "Config file should exist")
        _check(run_dir.metrics_path.is_file(), "Metrics file should exist")
        _check(run_dir.log_path.is_file(), "Log file should exist")
        _check(run_dir.path / "summary.json".is_file(), "Summary file should exist")

        print("✓ Artifacts created successfully")

    print("\nPhase 11 Sync Trainer smoke test: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"Phase 11 Sync Trainer smoke test: FAIL\n  {exc}", file=sys.stderr)
        raise SystemExit(1)
