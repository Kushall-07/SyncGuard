#!/usr/bin/env python3
"""Export Phase 8 visual encoder from a full DeepfakeClassifier checkpoint.

This script loads a Phase 8 training checkpoint (which contains the full
DeepfakeClassifier with encoder + head), extracts only the encoder weights,
and saves a standalone visual_encoder.pt checkpoint compatible with Phase 9/11.

Usage:
    python scripts/export_visual_encoder_from_phase8.py \
        --checkpoint outputs/runs/deepfake-transformer-final-20260908-210034/checkpoints/best.pt \
        --config outputs/runs/deepfake-transformer-final-20260908-210034/config.yaml \
        --output outputs/runs/deepfake-transformer-final-20260908-210034/checkpoints/visual_encoder.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ModelConfig, VideoDataConfig
from src.models.video.visual_encoder import VisualEncoder, export_visual_encoder


def extract_encoder_weights(full_checkpoint: dict[str, any]) -> dict[str, torch.Tensor]:
    """Extract encoder weights from a full DeepfakeClassifier checkpoint.

    Args:
        full_checkpoint: Dict with 'model' key containing full state_dict

    Returns:
        Encoder state_dict with 'encoder.' prefix stripped
    """
    full_state_dict = full_checkpoint["model"]

    # Filter keys that start with "encoder."
    encoder_keys = {k: v for k, v in full_state_dict.items() if k.startswith("encoder.")}

    if not encoder_keys:
        raise ValueError(
            "No keys starting with 'encoder.' found in checkpoint. "
            "This checkpoint may not be from a transformer variant DeepfakeClassifier."
        )

    # Strip the "encoder." prefix
    encoder_state_dict = {k[len("encoder.") :]: v for k, v in encoder_keys.items()}

    return encoder_state_dict


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export Phase 8 visual encoder from full checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to Phase 8 best.pt checkpoint",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to Phase 8 config.yaml",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output path for visual_encoder.pt",
    )
    args = parser.parse_args()

    # Validate inputs
    if not args.checkpoint.is_file():
        print(f"Error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1
    if not args.config.is_file():
        print(f"Error: config not found: {args.config}", file=sys.stderr)
        return 1

    print(f"Loading checkpoint from {args.checkpoint}")
    full_checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    print(f"Loading config from {args.config}")
    with args.config.open("r") as f:
        config_dict = yaml.safe_load(f)

    # Extract encoder weights
    print("Extracting encoder weights...")
    encoder_state_dict = extract_encoder_weights(full_checkpoint)
    print(f"Extracted {len(encoder_state_dict)} encoder tensors")

    # Reconstruct ModelConfig and VideoDataConfig
    model_cfg = ModelConfig.from_dict(config_dict["model"])
    video_cfg = VideoDataConfig.from_dict(config_dict["video"])

    # Verify variant is transformer
    if model_cfg.visual_encoder != "transformer":
        print(
            f"Error: expected visual_encoder='transformer', got '{model_cfg.visual_encoder}'",
            file=sys.stderr,
        )
        return 1

    # Instantiate encoder
    print(f"Instantiating VisualEncoder with variant={model_cfg.visual_encoder}")
    encoder = VisualEncoder(model_cfg)

    # Load extracted weights with strict=True
    print("Loading extracted weights into encoder...")
    missing_keys, unexpected_keys = encoder.load_state_dict(encoder_state_dict, strict=True)

    if missing_keys:
        print(f"Error: missing keys after stripping 'encoder.' prefix: {missing_keys}", file=sys.stderr)
        return 1
    if unexpected_keys:
        print(f"Error: unexpected keys after stripping 'encoder.' prefix: {unexpected_keys}", file=sys.stderr)
        return 1

    print("Strict loading successful: no missing or unexpected keys")

    # Export using the standard export_visual_encoder function
    print(f"Exporting to {args.output}")
    export_visual_encoder(
        args.output,
        encoder=encoder,
        model_cfg=model_cfg,
        video_cfg=video_cfg,
        extra={
            "source_checkpoint": str(args.checkpoint),
            "source_config": str(args.config),
            "exported_from": "phase8_deepfake_classifier",
        },
    )

    print(f"Successfully exported visual encoder to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
