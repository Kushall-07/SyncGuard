"""Phase 11: train the audio-visual synchronization model.

Examples
--------
Basic training run::

    python scripts/train_sync.py \\
        --config configs/av_align.yaml \\
        --train-manifest data/lavdf/manifest_train.csv \\
        --dev-manifest data/lavdf/manifest_dev.csv \\
        --video-dir data/lavdf/videos \\
        --audio-dir data/lavdf/audio \\
        --landmarks-dir data/lavdf/landmarks \\
        --audio-encoder checkpoints/audio_encoder.pt \\
        --visual-encoder checkpoints/visual_encoder.pt

Custom hyperparameters::

    python scripts/train_sync.py \\
        --config configs/av_align.yaml \\
        --train-manifest data/lavdf/manifest_train.csv \\
        --dev-manifest data/lavdf/manifest_dev.csv \\
        --video-dir data/lavdf/videos \\
        --audio-dir data/lavdf/audio \\
        --landmarks-dir data/lavdf/landmarks \\
        --audio-encoder checkpoints/audio_encoder.pt \\
        --visual-encoder checkpoints/visual_encoder.pt \\
        --epochs 50 \\
        --batch-size 16 \\
        --learning-rate 1e-4

Writes checkpoints, ``metrics.csv``, ``run.log`` and a final ``summary.json``
under ``outputs/runs/<name>-<timestamp>/``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import AudioConfig, ExperimentConfig, TrainingConfig, VideoDataConfig
from src.data.lavdf_dataset import build_lavdf_datasets, lavdf_collate_fn
from src.data.sync_pairs import SyncPairConfig
from src.losses.sync_loss import SyncLoss, SyncLossConfig
from src.models.audio.encoder import load_audio_encoder
from src.models.fusion.cross_attention import (
    BidirectionalCrossAttention,
    CrossAttentionConfig,
)
from src.models.heads.sync_head import SyncHead, SyncHeadConfig
from src.models.video.visual_encoder import load_visual_encoder
from src.training.sync_trainer import SyncModel, SyncTrainer
from src.training.utils import RunDirectory, get_device, set_seed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "av_align.yaml")
    parser.add_argument("--train-manifest", type=Path, required=True,
                        help="LAV-DF training manifest CSV")
    parser.add_argument("--dev-manifest", type=Path, required=True,
                        help="LAV-DF dev manifest CSV")
    parser.add_argument("--video-dir", type=Path, required=True,
                        help="Directory containing LAV-DF video files")
    parser.add_argument("--audio-dir", type=Path, required=True,
                        help="Directory containing LAV-DF audio files")
    parser.add_argument("--landmarks-dir", type=Path, required=True,
                        help="Directory containing LAV-DF landmark NPZ files")
    parser.add_argument("--audio-encoder", type=Path, required=True,
                        help="Path to Phase 5 audio encoder checkpoint")
    parser.add_argument("--visual-encoder", type=Path, required=True,
                        help="Path to Phase 8 visual encoder checkpoint")
    parser.add_argument("--output-dir", type=Path, default="outputs/runs",
                        help="Output directory for training runs")
    parser.add_argument("--run-name", type=str, default="sync-training",
                        help="Name for this training run")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate")
    parser.add_argument("--seed", type=int, help="Override random seed")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "auto"], default="auto",
                        help="Device to use for training")
    parser.add_argument("--num-workers", type=int, help="Override number of data loading workers")
    parser.add_argument("--help-config", action="store_true",
                        help="Show configuration and exit without training")

    args = parser.parse_args()

    # Load configuration YAML directly (av_align.yaml has custom structure)
    av_align_cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    # Set seed from experiment config if present, otherwise use default
    experiment_cfg = av_align_cfg.get("experiment", {})
    seed = experiment_cfg.get("seed", 42)
    deterministic = experiment_cfg.get("deterministic", False)
    set_seed(seed, deterministic=deterministic)

    # Load training config if present, otherwise use defaults
    training_cfg_dict = av_align_cfg.get("training", {})
    training_cfg = TrainingConfig.from_dict(training_cfg_dict)

    # Get device
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)

    # Load encoders
    print(f"Loading audio encoder from {args.audio_encoder}")
    audio_encoder, audio_payload = load_audio_encoder(args.audio_encoder, map_location=device)
    print(f"Loading visual encoder from {args.visual_encoder}")
    visual_encoder, visual_payload = load_visual_encoder(args.visual_encoder, map_location=device)

    # Get audio token seconds from payload
    audio_token_seconds = 0.08  # Default from Phase 5
    if "audio_cfg" in audio_payload:
        hop_length = audio_payload["audio_cfg"]["mel"]["hop_length"]
        time_downsample = int(audio_payload.get("time_downsample", 1))
        sample_rate = audio_payload["audio_cfg"]["sample_rate"]
        audio_token_seconds = (hop_length * time_downsample) / sample_rate
        print(f"Audio token seconds: {audio_token_seconds:.4f}")

    # Load cross-attention config from YAML
    ca_cfg_dict = av_align_cfg.get("av_align", {}).get("cross_attention", {})
    ca_cfg = CrossAttentionConfig(**ca_cfg_dict)

    # Load sync head config from YAML
    sh_cfg_dict = av_align_cfg.get("av_align", {}).get("sync_head", {})
    sh_cfg = SyncHeadConfig(**sh_cfg_dict)

    # Load sync loss config from YAML
    sl_cfg_dict = av_align_cfg.get("av_align", {}).get("sync_loss", {})
    sl_cfg = SyncLossConfig(**sl_cfg_dict)

    # Load sync pair config from YAML
    sp_cfg_dict = av_align_cfg.get("av_align", {}).get("sync_pairs", {})
    sp_cfg = SyncPairConfig(**sp_cfg_dict)
    sp_cfg = replace(sp_cfg, audio_token_seconds=audio_token_seconds)
    sp_cfg_dict = {}  # Clear after use

    # Initialize cross-attention
    cross_attention = BidirectionalCrossAttention.from_config(ca_cfg)

    # Initialize sync head
    sync_head = SyncHead.from_config(sh_cfg, input_dim=ca_cfg.dim)

    # Initialize sync loss
    sync_loss = SyncLoss.from_config(sl_cfg)

    # Create sync model
    model = SyncModel(
        audio_encoder=audio_encoder,
        visual_encoder=visual_encoder,
        cross_attention=cross_attention,
        sync_head=sync_head,
    )

    # Show configuration and exit if requested
    if args.help_config:
        print("Configuration:")
        print(f"  Audio encoder: {args.audio_encoder}")
        print(f"  Visual encoder: {args.visual_encoder}")
        print(f"  Audio token seconds: {audio_token_seconds:.4f}")
        print(f"  Cross-attention: {ca_cfg}")
        print(f"  Sync head: {sh_cfg}")
        print(f"  Sync loss: {sl_cfg}")
        print(f"  Sync pairs: {sp_cfg}")
        print(f"  Training: {training_cfg}")
        print(f"  Device: {device}")
        return 0

    # Build datasets
    print(f"Building datasets from {args.train_manifest} and {args.dev_manifest}")

    # Extract video config from av_align section
    video_dict = av_align_cfg.get("av_align", {}).get("video", {})
    video_cfg = VideoDataConfig(
        num_frames=video_dict.get("num_frames", 32),
        regions=video_dict.get("regions", "face_mouth"),
    )

    # Extract audio config from YAML or use defaults
    audio_dict = av_align_cfg.get("audio", {})
    audio_cfg = AudioConfig.from_dict(audio_dict)

    # Build train dataset
    train_dataset = build_lavdf_datasets(
        manifest_path=args.train_manifest,
        video_dir=args.video_dir,
        audio_dir=args.audio_dir,
        landmarks_dir=args.landmarks_dir,
        audio_cfg=audio_cfg,
        video_cfg=video_cfg,
        splits=("train",),
        n_video_tokens=video_dict.get("num_frames", 32),
        sync_pair_config=sp_cfg,
        use_negative_pairs=True,
        negative_pair_probability=training_cfg.negative_pair_probability,
        random_sample=True,
        seed=seed,
    )["train"]

    # Build dev dataset
    dev_dataset = build_lavdf_datasets(
        manifest_path=args.dev_manifest,
        video_dir=args.video_dir,
        audio_dir=args.audio_dir,
        landmarks_dir=args.landmarks_dir,
        audio_cfg=audio_cfg,
        video_cfg=video_cfg,
        splits=("dev",),
        n_video_tokens=video_dict.get("num_frames", 32),
        sync_pair_config=sp_cfg,
        use_negative_pairs=True,
        negative_pair_probability=training_cfg.negative_pair_probability,
        random_sample=True,
        seed=seed,
    )["dev"]

    datasets = {"train": train_dataset, "dev": dev_dataset}

    print(f"Train samples: {len(datasets['train'])}")
    print(f"Dev samples: {len(datasets['dev'])}")

    # Create data loaders
    batch_size = args.batch_size if args.batch_size is not None else training_cfg.batch_size
    num_workers = args.num_workers if args.num_workers is not None else training_cfg.num_workers

    train_loader = torch.utils.data.DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        collate_fn=lavdf_collate_fn,
    )
    dev_loader = torch.utils.data.DataLoader(
        datasets["dev"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=lavdf_collate_fn,
    )

    # Create optimizer
    learning_rate = args.learning_rate if args.learning_rate is not None else training_cfg.learning_rate
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=training_cfg.weight_decay,
    )

    # Create run directory
    run_dir = RunDirectory(args.output_dir, args.run_name)
    run_dir.save_config(av_align_cfg)

    # Create trainer
    epochs = args.epochs if args.epochs is not None else training_cfg.epochs
    trainer = SyncTrainer(
        model=model,
        optimizer=optimizer,
        config=training_cfg,
        run_dir=run_dir,
        device=device,
        sync_loss=sync_loss,
        sync_pair_config=sp_cfg,
        audio_token_seconds=audio_token_seconds,
        aggregation=sh_cfg.aggregation,
    )

    # Train
    print(f"Starting training for {epochs} epochs")
    summary = trainer.fit(train_loader, dev_loader, epochs=epochs)

    # Print summary
    print(f"\nRun directory: {run_dir.path}")
    if summary['best_metric'] is not None:
        print(f"Best {summary['monitor']} = {summary['best_metric']:.4f} @ epoch {summary['best_epoch']}")
    else:
        print(f"Monitor {summary['monitor']} not found in metrics")
    print(f"Epochs run: {summary['epochs_run']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
