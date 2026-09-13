"""Tests for Phase 11 train_sync.py configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config import TrainingConfig


def test_av_align_yaml_structure() -> None:
    """Test that configs/av_align.yaml has the expected structure."""
    config_path = Path("configs/av_align.yaml")
    if not config_path.is_file():
        pytest.skip(f"Config file not found: {config_path}")

    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)

    # Check top-level keys
    assert "audio" in cfg
    assert "experiment" in cfg
    assert "training" in cfg
    assert "av_align" in cfg

    # Check av_align sub-keys
    av_align = cfg["av_align"]
    assert "audio_encoder_pt" in av_align
    assert "visual_encoder_pt" in av_align
    assert "alignment" in av_align
    assert "video" in av_align
    assert "cross_attention" in av_align
    assert "sync_head" in av_align
    assert "sync_loss" in av_align
    assert "sync_pairs" in av_align

    # Check sync_pairs does not have negative_pair_probability (that's in training)
    sync_pairs = av_align["sync_pairs"]
    assert "negative_pair_probability" not in sync_pairs
    assert "shift_seconds" in sync_pairs
    assert "shift_strategy" in sync_pairs
    assert "audio_token_seconds" in sync_pairs

    # Check training has negative_pair_probability
    training = cfg["training"]
    assert "negative_pair_probability" in training
    assert 0 <= training["negative_pair_probability"] <= 1

    # Check video section does not have landmark_coords (not in VideoDataConfig)
    video = av_align["video"]
    assert "landmark_coords" not in video
    assert "num_frames" in video
    assert "regions" in video


def test_training_config_with_negative_pair_probability() -> None:
    """Test that TrainingConfig accepts negative_pair_probability."""
    cfg_dict = {
        "batch_size": 32,
        "learning_rate": 1e-4,
        "weight_decay": 0.0001,
        "epochs": 10,
        "num_workers": 2,
        "amp": True,
        "grad_accum_steps": 1,
        "grad_clip_norm": 5.0,
        "monitor": "val_loss",
        "monitor_mode": "min",
        "early_stopping_patience": 5,
        "scheduler": "none",
        "warmup_epochs": 0,
        "class_weight": "balanced",
        "negative_pair_probability": 0.5,
    }

    cfg = TrainingConfig.from_dict(cfg_dict)
    assert cfg.negative_pair_probability == 0.5


def test_training_config_negative_pair_probability_validation() -> None:
    """Test that TrainingConfig validates negative_pair_probability range."""
    cfg_dict = {
        "batch_size": 32,
        "learning_rate": 1e-4,
        "epochs": 10,
        "negative_pair_probability": 1.5,  # Invalid: > 1
    }

    with pytest.raises(ValueError, match="negative_pair_probability must be in"):
        TrainingConfig.from_dict(cfg_dict)

    cfg_dict["negative_pair_probability"] = -0.1  # Invalid: < 0
    with pytest.raises(ValueError, match="negative_pair_probability must be in"):
        TrainingConfig.from_dict(cfg_dict)


def test_sync_pair_config_mutable() -> None:
    """Test that SyncPairConfig is mutable for audio_token_seconds replacement."""
    from src.data.sync_pairs import SyncPairConfig

    cfg = SyncPairConfig(shift_seconds=[-1.0, 1.0], audio_token_seconds=0.08)
    assert cfg.audio_token_seconds == 0.08

    # Replace should work now that it's not frozen
    from dataclasses import replace

    updated = replace(cfg, audio_token_seconds=0.01)
    assert updated.audio_token_seconds == 0.01
    assert cfg.audio_token_seconds == 0.08  # Original unchanged


def test_video_data_config_construction() -> None:
    """Test that VideoDataConfig can be constructed without landmark_coords."""
    from src.config import VideoDataConfig

    # VideoDataConfig does not have landmark_coords field
    cfg = VideoDataConfig(
        num_frames=32,
        regions="face_mouth",
    )
    assert cfg.num_frames == 32
    assert cfg.regions == "face_mouth"

    # Passing landmark_coords should fail
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        VideoDataConfig(
            num_frames=32,
            regions="face_mouth",
            landmark_coords=3,  # This should fail
        )


def test_audio_token_seconds_calculation() -> None:
    """Test that audio token seconds is calculated correctly from Phase 5 payload."""
    # Phase 5 audio encoder payload structure
    audio_payload = {
        "audio_cfg": {
            "mel": {
                "hop_length": 160,
            },
            "sample_rate": 16000,
        },
        "time_downsample": 1,  # Not present in actual payload, default to 1
    }

    # Calculate audio token seconds: hop_length * time_downsample / sample_rate
    hop_length = audio_payload["audio_cfg"]["mel"]["hop_length"]
    time_downsample = audio_payload.get("time_downsample", 1)
    sample_rate = audio_payload["audio_cfg"]["sample_rate"]
    audio_token_seconds = (hop_length * time_downsample) / sample_rate

    # Expected: 160 * 1 / 16000 = 0.01
    assert audio_token_seconds == 0.01


def test_monitor_metric_exists_in_phase11_metrics() -> None:
    """Test that the configured monitor metric exists in Phase 11 validation metrics."""
    from pathlib import Path

    # Load config
    config_path = Path("configs/av_align.yaml")
    if not config_path.is_file():
        pytest.skip("Config file not found")

    import yaml
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)

    # Get monitor from training config
    monitor = cfg["training"]["monitor"]
    monitor_mode = cfg["training"]["monitor_mode"]

    # Verify it's the Phase 11 video-level metric
    assert monitor == "val_video_sync_auc", f"Expected monitor 'val_video_sync_auc', got '{monitor}'"
    assert monitor_mode == "max", f"Expected monitor_mode 'max', got '{monitor_mode}'"

    # Verify the metric name matches what SyncTrainer.compute_metrics returns
    expected_metrics = [
        "sync_accuracy",
        "sync_precision",
        "sync_recall",
        "sync_f1",
        "sync_auc",
        "video_sync_accuracy",
        "video_sync_auc",
    ]

    # The trainer prefixes with "val_" for validation metrics
    expected_val_metrics = [f"val_{m}" for m in expected_metrics]

    assert monitor in expected_val_metrics, (
        f"Monitor '{monitor}' not in expected Phase 11 metrics: {expected_val_metrics}"
    )

    # Verify monitor_mode is compatible with the metric (higher is better for AUC)
    assert monitor_mode in ("min", "max"), f"monitor_mode must be 'min' or 'max', got '{monitor_mode}'"


def test_first_batch_loading() -> None:
    """Test that the first batch can be loaded without errors."""
    from pathlib import Path

    # Skip if required files don't exist
    if not Path("data/lavdf/manifest_train.csv").is_file():
        pytest.skip("LAV-DF manifest not found")
    if not Path("data/lavdf/processed/audio").exists():
        pytest.skip("LAV-DF audio directory not found")
    if not Path("data/lavdf/processed/landmarks").exists():
        pytest.skip("LAV-DF landmarks directory not found")
    if not Path("outputs/runs/spoof-transformer-20260906-123646/checkpoints/audio_encoder.pt").is_file():
        pytest.skip("Audio encoder checkpoint not found")
    if not Path("outputs/runs/deepfake-transformer-final-20260908-210034/checkpoints/visual_encoder.pt").is_file():
        pytest.skip("Visual encoder checkpoint not found")

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    import torch
    import yaml
    from dataclasses import replace

    from src.config import AudioConfig, TrainingConfig, VideoDataConfig
    from src.data.lavdf_dataset import build_lavdf_datasets
    from src.data.sync_pairs import SyncPairConfig
    from src.models.audio.encoder import load_audio_encoder
    from src.models.video.visual_encoder import load_visual_encoder

    # Load config
    config_path = Path("configs/av_align.yaml")
    with config_path.open("r") as f:
        av_align_cfg = yaml.safe_load(f)

    # Load encoders
    audio_encoder, audio_payload = load_audio_encoder(
        "outputs/runs/spoof-transformer-20260906-123646/checkpoints/audio_encoder.pt",
        map_location="cpu"
    )
    visual_encoder, visual_payload = load_visual_encoder(
        "outputs/runs/deepfake-transformer-final-20260908-210034/checkpoints/visual_encoder.pt",
        map_location="cpu"
    )

    # Get audio token seconds
    hop_length = audio_payload["audio_cfg"]["mel"]["hop_length"]
    time_downsample = audio_payload.get("time_downsample", 1)
    sample_rate = audio_payload["audio_cfg"]["sample_rate"]
    audio_token_seconds = (hop_length * time_downsample) / sample_rate

    # Load configs
    video_dict = av_align_cfg.get("av_align", {}).get("video", {})
    video_cfg = VideoDataConfig(
        num_frames=video_dict.get("num_frames", 32),
        regions=video_dict.get("regions", "face_mouth"),
    )
    audio_dict = av_align_cfg.get("audio", {})
    audio_cfg = AudioConfig.from_dict(audio_dict)
    training_cfg = TrainingConfig.from_dict(av_align_cfg.get("training", {}))

    # Load sync pair config
    sp_cfg_dict = av_align_cfg.get("av_align", {}).get("sync_pairs", {})
    sp_cfg = SyncPairConfig(**sp_cfg_dict)
    sp_cfg = replace(sp_cfg, audio_token_seconds=audio_token_seconds)

    # Build train dataset (only train split)
    train_dataset = build_lavdf_datasets(
        manifest_path="data/lavdf/manifest_train.csv",
        video_dir="data/lavdf/extracted/train",
        audio_dir="data/lavdf/processed/audio",
        landmarks_dir="data/lavdf/processed/landmarks",
        audio_cfg=audio_cfg,
        video_cfg=video_cfg,
        splits=("train",),
        n_video_tokens=video_dict.get("num_frames", 32),
        sync_pair_config=sp_cfg,
        use_negative_pairs=True,
        negative_pair_probability=training_cfg.negative_pair_probability,
        random_sample=True,
        seed=42,
    )["train"]

    # Load first batch
    def collate_fn(batch):
        max_mel_len = max(item["mel_window"].shape[1] for item in batch)
        n_mels = batch[0]["mel_window"].shape[0]
        mel_windows = []
        for item in batch:
            mel = item["mel_window"]
            if mel.shape[1] < max_mel_len:
                padding = max_mel_len - mel.shape[1]
                mel = torch.nn.functional.pad(mel, (0, padding), mode='constant', value=0)
            mel_windows.append(mel)
        return {
            "mel_window": torch.stack(mel_windows),
            "landmarks": torch.stack([item["landmarks"] for item in batch]),
            "fps": [item["fps"] for item in batch],
            "window_seconds": [item["window_seconds"] for item in batch],
            "shift_seconds": [item["shift_seconds"] for item in batch],
            "is_positive": [item["is_positive"] for item in batch],
        }

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Get first batch
    batch = next(iter(train_loader))

    # Verify batch structure
    assert "mel_window" in batch
    assert "landmarks" in batch
    assert "fps" in batch
    assert "window_seconds" in batch
    assert "shift_seconds" in batch
    assert "is_positive" in batch

    # Verify shapes
    assert batch["mel_window"].shape[0] == 2  # batch size
    assert batch["landmarks"].shape[0] == 2
    assert len(batch["fps"]) == 2
    assert len(batch["window_seconds"]) == 2
    assert len(batch["shift_seconds"]) == 2
    assert len(batch["is_positive"]) == 2
