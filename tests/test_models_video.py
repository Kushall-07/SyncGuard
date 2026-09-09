"""Phase 7 unit tests: visual branch models + shared-encoder export round-trip."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.config import ModelConfig, TrainingConfig, VideoDataConfig
from src.data.audio_dataset import build_dataloader
from src.data.celebdf_dataset import build_celebdf_datasets, synthetic_landmark_manifest
from src.models.video import (
    DeepfakeClassifier,
    LandmarkEmbedding,
    LandmarkMLPBaseline,
    VisualEncoder,
    export_visual_encoder,
    load_visual_encoder,
)
from src.preprocessing.landmarks import region_point_count
from src.training.deepfake_trainer import DeepfakeTrainer
from src.training.utils import RunDirectory

DEVICE = "cpu"


def _cfg(**kw) -> ModelConfig:
    base = dict(visual_embedding_dim=32, num_heads=4, visual_embed_hidden=48,
                visual_tf_ff_dim=64, visual_tf_layers=2, visual_tf_dropout=0.0,
                spoof_head_hidden=32, dropout=0.0)
    base.update(kw)
    return ModelConfig(**base)


@pytest.mark.parametrize("regions", ["face", "mouth", "face_mouth"])
def test_landmark_embedding_shapes(regions: str) -> None:
    cfg = _cfg(visual_regions=regions, landmark_coords=3)
    emb = LandmarkEmbedding(cfg)
    n = region_point_count(regions)
    x = torch.randn(2, 8, n, 3)
    assert emb(x).shape == (2, 8, 32)
    assert emb(x.reshape(2, 8, n * 3)).shape == (2, 8, 32)


def test_landmark_embedding_rejects_wrong_point_count() -> None:
    emb = LandmarkEmbedding(_cfg(visual_regions="mouth", landmark_coords=3))
    with pytest.raises(ValueError):
        emb(torch.randn(2, 8, 999, 3))


@pytest.mark.parametrize("variant", ["transformer", "mlp_baseline"])
def test_deepfake_classifier_forward(variant: str) -> None:
    cfg = _cfg(visual_encoder=variant, visual_regions="face_mouth")
    model = DeepfakeClassifier(cfg)
    n = region_point_count("face_mouth")
    logits = model(torch.randn(3, 10, n, 3))
    assert logits.shape == (3, 2)
    assert model.variant == variant


def test_baseline_is_frame_order_invariant() -> None:
    model = LandmarkMLPBaseline(_cfg(visual_regions="face_mouth")).eval()
    n = region_point_count("face_mouth")
    x = torch.randn(2, 12, n, 3)
    a = model(x)
    b = model(x[:, torch.randperm(12)])
    assert torch.allclose(a, b, atol=1e-5)


def test_encode_only_on_transformer_variant() -> None:
    n = region_point_count("face_mouth")
    tf = DeepfakeClassifier(_cfg(visual_encoder="transformer"))
    assert tf.encode(torch.randn(1, 6, n, 3)).shape == (1, 6, 32)
    mlp = DeepfakeClassifier(_cfg(visual_encoder="mlp_baseline"))
    with pytest.raises(RuntimeError):
        mlp.encode(torch.randn(1, 6, n, 3))


def test_visual_encoder_export_round_trip(tmp_path) -> None:
    cfg = _cfg(visual_encoder="transformer")
    enc = VisualEncoder(cfg).eval()
    video_cfg = VideoDataConfig()
    path = export_visual_encoder(tmp_path / "visual_encoder.pt", encoder=enc,
                                 model_cfg=cfg, video_cfg=video_cfg)
    enc2, payload = load_visual_encoder(path)
    enc2.eval()
    n = region_point_count(cfg.visual_regions)
    x = torch.randn(2, 8, n, 3)
    assert torch.allclose(enc(x).tokens, enc2(x).tokens, atol=1e-6)
    assert payload["variant"] == "transformer"
    assert payload["regions"] == cfg.visual_regions


def test_deepfake_trainer_two_epoch_smoke(tmp_path) -> None:
    manifest = synthetic_landmark_manifest(
        tmp_path / "lm", n_per_split={"train": 16, "dev": 8}, n_ids_per_split=4,
        num_frames=8, seed=0,
    )
    video_cfg = VideoDataConfig(landmarks_dir=str(tmp_path / "lm"), num_frames=8)
    datasets = build_celebdf_datasets(manifest, video_cfg, splits=("train", "dev"),
                                      landmarks_dir=tmp_path / "lm", seed=0)
    train_cfg = TrainingConfig(batch_size=8, epochs=2, amp=False, num_workers=0,
                               monitor="val_eer", monitor_mode="min", early_stopping_patience=0)
    model = DeepfakeClassifier(_cfg(visual_encoder="transformer"))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    run_dir = RunDirectory(tmp_path / "runs", "video-smoke")
    trainer = DeepfakeTrainer(model, opt, config=train_cfg, run_dir=run_dir, device=DEVICE,
                              class_weights=datasets["train"].label_weights())
    summary = trainer.fit(
        build_dataloader(datasets["train"], train_cfg, shuffle=True),
        build_dataloader(datasets["dev"], train_cfg, shuffle=False),
    )
    assert summary["epochs_run"] == 2
    assert run_dir.checkpoint_path("best").is_file()
    assert run_dir.metrics_path.is_file()
