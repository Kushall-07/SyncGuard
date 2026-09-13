"""Tests for visual encoder export from Phase 8 checkpoint."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.config import ModelConfig, VideoDataConfig
from src.models.video.visual_encoder import VisualEncoder, export_visual_encoder, load_visual_encoder


def test_export_load_roundtrip() -> None:
    """Test that export_visual_encoder and load_visual_encoder are compatible."""
    # Create a simple config matching Phase 8
    model_cfg = ModelConfig(
        visual_encoder="transformer",
        visual_embedding_dim=256,
        num_heads=4,
        visual_tf_layers=2,
        visual_tf_ff_dim=512,
        visual_tf_dropout=0.1,
        visual_regions="face_mouth",  # Phase 8 used dense landmarks (142 points)
        landmark_coords=3,
    )
    video_cfg = VideoDataConfig(
        num_frames=32,
        regions="face_mouth",
        normalize="interocular",
    )

    # Create encoder
    encoder = VisualEncoder(model_cfg)

    # Export
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "visual_encoder.pt"
        export_visual_encoder(
            export_path,
            encoder=encoder,
            model_cfg=model_cfg,
            video_cfg=video_cfg,
        )

        # Load
        loaded_encoder, payload = load_visual_encoder(export_path)

        # Verify structure
        assert loaded_encoder.variant == "transformer"
        assert loaded_encoder.output_dim == encoder.output_dim
        assert payload["format"] == 1
        assert "model_cfg" in payload
        assert "video_cfg" in payload
        assert "variant" in payload
        assert payload["variant"] == "transformer"

        # Verify weights match
        for (name1, p1), (name2, p2) in zip(
            encoder.named_parameters(), loaded_encoder.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(p1, p2, atol=1e-6)

        # Verify forward pass works with dense landmarks
        B, T = 2, 32
        landmarks = torch.randn(B, T, 142, 3)
        output = loaded_encoder(landmarks)
        assert output.tokens.shape == (B, T, 256)


def test_load_phase8_exported_encoder() -> None:
    """Test loading the actual Phase 8 exported encoder."""
    export_path = Path(
        "outputs/runs/deepfake-transformer-final-20260908-210034/checkpoints/visual_encoder.pt"
    )

    if not export_path.is_file():
        pytest.skip(f"Phase 8 export not found at {export_path}")

    # Load
    encoder, payload = load_visual_encoder(export_path)

    # Verify basic properties
    assert encoder.variant == "transformer"
    assert encoder.output_dim == 256
    assert payload["variant"] == "transformer"
    assert payload["format"] == 1

    # Verify config reconstruction
    model_cfg = ModelConfig.from_dict(payload["model_cfg"])
    assert model_cfg.visual_encoder == "transformer"
    assert model_cfg.visual_embedding_dim == 256
    assert model_cfg.visual_tf_layers == 3
    assert model_cfg.num_heads == 4

    video_cfg = VideoDataConfig.from_dict(payload["video_cfg"])
    assert video_cfg.num_frames == 32
    assert video_cfg.regions == "face_mouth"

    # Verify forward pass works
    # Phase 8 was trained on dense landmarks (142 points per frame)
    B, T = 2, 32
    landmarks = torch.randn(B, T, 142, 3)
    output = encoder(landmarks)
    assert output.tokens.shape == (B, T, 256)
