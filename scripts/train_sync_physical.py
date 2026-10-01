"""Train the audio-visual SyncHead on PHYSICALLY-shifted audio (sync-detection fix).

See `configs/av_align_physical.yaml` and
`docs/decisions/0006-physical-sync-training.md` for the full design. Unlike
`scripts/train_sync.py` (the original Phase 11/12 token-reassignment baseline,
left unmodified for comparison), this script trains on
`src.data.lavdf_physical_sync_dataset.LAVDFPhysicalSyncDataset`: negatives are
built by physically shifting the raw audio waveform (or substituting a
different clip's native audio) before the mel spectrogram / audio encoder,
using the same consecutive 32-frame production windows
`SyncGuardPredictor` evaluates at inference time.

Example
-------
    .venv/Scripts/python.exe scripts/train_sync_physical.py \\
        --config configs/av_align_physical.yaml \\
        --train-manifest data/lavdf/manifest_train.csv \\
        --dev-manifest data/lavdf/manifest_dev.csv \\
        --video-dir data/lavdf/extracted/train \\
        --audio-dir data/lavdf/processed/audio \\
        --landmarks-dir data/lavdf/processed/landmarks \\
        --audio-encoder outputs/runs/spoof-transformer-20260906-123646/checkpoints/audio_encoder.pt \\
        --visual-encoder outputs/runs/deepfake-transformer-final-20260908-210034/checkpoints/visual_encoder.pt

Writes checkpoints, `metrics.csv`, `run.log` and `summary.json` under
`outputs/runs/<run-name>-<timestamp>/`.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import AudioConfig, TrainingConfig, VideoDataConfig
from src.data.lavdf_physical_sync_dataset import (
    PhysicalSyncPairConfig,
    build_lavdf_physical_datasets,
    physical_sync_collate_fn,
)
from src.losses.sync_loss import SyncLoss, SyncLossConfig
from src.models.audio.encoder import audio_token_seconds_from_encoder, load_audio_encoder
from src.models.fusion.cross_attention import BidirectionalCrossAttention, CrossAttentionConfig
from src.models.heads.sync_head import SyncHead, SyncHeadConfig
from src.models.video.visual_encoder import load_visual_encoder
from src.training.physical_sync_trainer import PhysicalSyncModel, PhysicalSyncTrainer
from src.training.utils import RunDirectory, build_scheduler, get_device, set_seed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "av_align_physical.yaml")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True, help="Video dir for the train split (not read for training - see module docstring)")
    parser.add_argument("--dev-video-dir", type=Path, help="Video dir for the dev split, if different from --video-dir")
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--landmarks-dir", type=Path, required=True)
    parser.add_argument("--audio-encoder", type=Path, required=True)
    parser.add_argument("--visual-encoder", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default="outputs/runs")
    parser.add_argument("--run-name", type=str, default="sync-physical")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--num-workers", type=int, help="Override number of data loading workers")
    parser.add_argument(
        "--max-windows-per-clip", type=int,
        help="Speed: cap windows per clip at dataset-build time (evenly spaced, deterministic, every clip "
             "still represented at least once) - the main lever for per-epoch wall-clock time, since it "
             "directly cuts the number of full-clip audio decodes.",
    )
    parser.add_argument("--limit-train-samples", type=int, help="Debug/speed: cap number of train windows used")
    parser.add_argument(
        "--limit-dev-samples", type=int,
        help="Speed: cap number of dev windows used for PER-EPOCH validation (a random, seeded, fixed subset - "
             "not the final held-out number; run scripts/evaluate_physical_sync.py on the full dev set after "
             "training for that).",
    )
    parser.add_argument("--help-config", action="store_true")
    args = parser.parse_args()

    av_cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    experiment_cfg = av_cfg.get("experiment", {})
    seed = experiment_cfg.get("seed", 42)
    set_seed(seed, deterministic=experiment_cfg.get("deterministic", False))

    training_cfg = TrainingConfig.from_dict(av_cfg.get("training", {}))
    device = get_device() if args.device == "auto" else torch.device(args.device)

    print(f"Loading audio encoder from {args.audio_encoder}")
    audio_encoder, audio_payload = load_audio_encoder(args.audio_encoder, map_location=device)
    print(f"Loading visual encoder from {args.visual_encoder}")
    visual_encoder, _ = load_visual_encoder(args.visual_encoder, map_location=device)

    audio_token_seconds = audio_token_seconds_from_encoder(audio_encoder, audio_payload["audio_cfg"])
    print(f"Audio token seconds: {audio_token_seconds:.4f}")

    ca_cfg = CrossAttentionConfig(**av_cfg.get("av_align", {}).get("cross_attention", {}))
    sh_cfg = SyncHeadConfig(**av_cfg.get("av_align", {}).get("sync_head", {}))
    sl_cfg = SyncLossConfig(**av_cfg.get("av_align", {}).get("sync_loss", {}))
    ps_cfg_dict = dict(av_cfg.get("av_align", {}).get("physical_sync_pairs", {}))
    pair_cfg = PhysicalSyncPairConfig(**{k: tuple(v) if k == "shift_magnitudes_seconds" else v for k, v in ps_cfg_dict.items()})

    cross_attention = BidirectionalCrossAttention.from_config(ca_cfg)
    sync_head = SyncHead.from_config(sh_cfg, input_dim=ca_cfg.dim)
    sync_loss = SyncLoss.from_config(sl_cfg)

    model = PhysicalSyncModel(audio_encoder, visual_encoder, cross_attention, sync_head)

    if args.help_config:
        print("Configuration:")
        print(f"  Audio token seconds: {audio_token_seconds:.4f}")
        print(f"  Cross-attention: {ca_cfg}")
        print(f"  Sync head: {sh_cfg}")
        print(f"  Sync loss: {sl_cfg}")
        print(f"  Physical sync pairs: {pair_cfg}")
        print(f"  Training: {training_cfg}")
        print(f"  Device: {device}")
        return 0

    video_dict = av_cfg.get("av_align", {}).get("video", {})
    video_cfg = VideoDataConfig(num_frames=video_dict.get("num_frames", 32), regions=video_dict.get("regions", "face_mouth"))
    audio_cfg = AudioConfig.from_dict(av_cfg.get("audio", {}))

    print(f"Building datasets from {args.train_manifest} and {args.dev_manifest}")
    train_dataset = build_lavdf_physical_datasets(
        args.train_manifest, args.video_dir, args.audio_dir, args.landmarks_dir,
        audio_cfg, video_cfg, splits=("train",), pair_config=pair_cfg, seed=seed,
        max_windows_per_clip=args.max_windows_per_clip,
    )["train"]
    dev_dataset = build_lavdf_physical_datasets(
        args.dev_manifest, args.dev_video_dir or args.video_dir, args.audio_dir, args.landmarks_dir,
        audio_cfg, video_cfg, splits=("dev",), pair_config=pair_cfg, seed=seed,
        max_windows_per_clip=args.max_windows_per_clip,
    )["dev"]

    if args.limit_train_samples:
        train_dataset._entries = train_dataset._entries[: args.limit_train_samples]
    if args.limit_dev_samples and len(dev_dataset) > args.limit_dev_samples:
        subset_rng = random.Random(seed)
        keep = sorted(subset_rng.sample(range(len(dev_dataset)), args.limit_dev_samples))
        dev_dataset._entries = [dev_dataset._entries[i] for i in keep]

    print(f"Train windows: {len(train_dataset)}")
    print(f"Dev windows: {len(dev_dataset)}")

    batch_size = args.batch_size if args.batch_size is not None else training_cfg.batch_size
    num_workers = args.num_workers if args.num_workers is not None else training_cfg.num_workers

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=(device.type == "cuda"), drop_last=True, collate_fn=physical_sync_collate_fn,
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=(device.type == "cuda"), collate_fn=physical_sync_collate_fn,
    )

    # Resolve the EFFECTIVE epoch count before building the scheduler - the cosine
    # schedule's decay curve spans `training_cfg.epochs`, so a --epochs override
    # must be folded into the config passed to build_scheduler/the trainer,
    # otherwise LR would decay to ~0 at the YAML's epoch count and then sit at ~0
    # for the rest of an overridden, longer run.
    epochs = args.epochs if args.epochs is not None else training_cfg.epochs
    if epochs != training_cfg.epochs:
        from dataclasses import replace as _replace
        training_cfg = _replace(training_cfg, epochs=epochs)

    learning_rate = args.learning_rate if args.learning_rate is not None else training_cfg.learning_rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=training_cfg.weight_decay)
    scheduler = build_scheduler(optimizer, training_cfg)

    run_dir = RunDirectory(args.output_dir, args.run_name)
    run_dir.save_config(av_cfg)

    trainer = PhysicalSyncTrainer(
        model, sync_loss, optimizer=optimizer, config=training_cfg, run_dir=run_dir, device=device,
        scheduler=scheduler, audio_token_seconds=audio_token_seconds, aggregation=sh_cfg.aggregation,
    )

    print(f"Starting training for {epochs} epochs")
    summary = trainer.fit(train_loader, dev_loader, epochs=epochs)

    print(f"\nRun directory: {run_dir.path}")
    if summary["best_metric"] is not None:
        print(f"Best {summary['monitor']} = {summary['best_metric']:.4f} @ epoch {summary['best_epoch']}")
    print(f"Epochs run: {summary['epochs_run']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
