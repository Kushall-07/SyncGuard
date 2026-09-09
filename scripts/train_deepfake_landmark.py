"""Phase 7: train a landmark-based video deepfake detector on Celeb-DF-v2.

Examples
--------
Plumbing run on random landmarks (no dataset / MediaPipe needed)::

    python scripts/train_deepfake_landmark.py --synthetic --epochs 2
    python scripts/train_deepfake_landmark.py --synthetic --epochs 2 \
        --config configs/deepfake_landmark_baseline.yaml

Real run once landmarks are cached and the manifest built::

    python scripts/train_deepfake_landmark.py --manifest data/celebdf/manifest.csv --export-encoder

Writes checkpoints, ``metrics.csv``, ``run.log`` and a final ``report.json``
(accuracy / precision / recall / F1 / ROC-AUC / EER / confusion on the eval
split) under ``outputs/runs/<name>-<timestamp>/``.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config  # noqa: E402
from src.data.audio_dataset import build_dataloader  # noqa: E402
from src.data.celebdf_dataset import (  # noqa: E402
    build_celebdf_datasets,
    build_celebdf_manifest,
    synthetic_landmark_manifest,
)
from src.data.manifests import Manifest  # noqa: E402
from src.evaluation.metrics import binary_classification_report  # noqa: E402
from src.models.video import DeepfakeClassifier, export_visual_encoder  # noqa: E402
from src.preprocessing.video_augment import FrameAugment  # noqa: E402
from src.training.checkpoint import load_checkpoint  # noqa: E402
from src.training.deepfake_trainer import DeepfakeTrainer  # noqa: E402
from src.training.utils import (  # noqa: E402
    RunDirectory,
    build_scheduler,
    count_parameters,
    get_device,
    set_seed,
)

SYNTHETIC_DIR = REPO_ROOT / "outputs" / "data" / "deepfake_synthetic_landmarks"


def _apply_overrides(cfg, args):
    """Return cfg with --regions / --num-frames / --landmarks-dir folded into the
    model + video sections. ``--landmarks-dir`` + ``--num-frames 32`` is the
    Phase 8 dense-cache plumbing: point the dataset at data/celebdf/landmarks_dense
    and consume the full 32-frame contiguous window (num_frames == window length
    makes frame selection the identity, so the window stays contiguous)."""

    model, video, experiment = cfg.model, cfg.video, cfg.experiment
    if args.regions:
        model = dataclasses.replace(model, visual_regions=args.regions)
        video = dataclasses.replace(video, regions=args.regions)
    if args.num_frames:
        video = dataclasses.replace(video, num_frames=args.num_frames)
    if getattr(args, "landmarks_dir", None):
        video = dataclasses.replace(video, landmarks_dir=str(args.landmarks_dir))
    if getattr(args, "seed", None) is not None:
        experiment = dataclasses.replace(experiment, seed=args.seed)
    return dataclasses.replace(cfg, model=model, video=video, experiment=experiment)


def _resolve_manifest(args, cfg) -> tuple[Manifest, Path, bool]:
    """Return (manifest, landmarks_dir, is_synthetic)."""

    if args.synthetic or (not args.manifest and not args.celebdf_root
                          and not Path(cfg.video.manifest_csv).is_file()):
        manifest = synthetic_landmark_manifest(
            SYNTHETIC_DIR,
            n_per_split={"train": 96, "dev": 32, "eval": 48},
            n_ids_per_split=8,
            num_frames=max(8, cfg.video.num_frames),
            seed=cfg.experiment.seed,
        )
        return manifest, SYNTHETIC_DIR, True
    if args.manifest:
        return Manifest.read_csv(args.manifest), Path(cfg.video.landmarks_dir), False
    if args.celebdf_root:
        m = build_celebdf_manifest(args.celebdf_root, out_csv=args.out_manifest,
                                   dev_identity_frac=cfg.video.dev_identity_frac,
                                   seed=cfg.experiment.seed,
                                   landmarks_dir=cfg.video.landmarks_dir,
                                   min_valid_frames=cfg.video.min_valid_frames)
        return m, Path(cfg.video.landmarks_dir), False
    return Manifest.read_csv(cfg.video.manifest_csv), Path(cfg.video.landmarks_dir), False


@torch.no_grad()
def _evaluate(model, loader, device) -> dict:
    model.eval()
    all_logits, all_targets = [], []
    for features, target in loader:
        all_logits.append(model(features.to(device)).float().cpu())
        all_targets.append(target)
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    probs = torch.softmax(logits, dim=1)[:, 1]
    return binary_classification_report(targets, logits.argmax(dim=1), probs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "configs" / "deepfake_transformer.yaml")
    parser.add_argument("--manifest", type=Path, help="pre-built manifest CSV")
    parser.add_argument("--celebdf-root", type=Path, help="data/celebdf; builds the manifest")
    parser.add_argument("--out-manifest", type=Path,
                        default=REPO_ROOT / "data" / "celebdf" / "manifest.csv")
    parser.add_argument("--synthetic", action="store_true",
                        help="force a random-landmark plumbing run")
    parser.add_argument("--epochs", type=int, help="override training.epochs")
    parser.add_argument("--eval-split", default="eval", choices=["train", "dev", "eval"])
    parser.add_argument("--regions", choices=["face", "mouth", "face_mouth"],
                        help="ablation override for model + video regions")
    parser.add_argument("--num-frames", type=int,
                        help="override video.num_frames (use 32 with the dense cache)")
    parser.add_argument("--landmarks-dir", type=Path,
                        help="override video.landmarks_dir (e.g. data/celebdf/landmarks_dense)")
    parser.add_argument("--seed", type=int, help="override experiment.seed")
    parser.add_argument("--export-encoder", action="store_true",
                        help="write <run>/checkpoints/visual_encoder.pt (transformer only)")
    args = parser.parse_args()

    cfg = _apply_overrides(load_config(args.config), args)
    set_seed(cfg.experiment.seed, deterministic=cfg.experiment.deterministic)
    device = get_device()

    manifest, landmarks_dir, is_synthetic = _resolve_manifest(args, cfg)
    splits = tuple(s for s in ("train", "dev", args.eval_split) if s in manifest.split_sizes())
    datasets = build_celebdf_datasets(
        manifest, cfg.video, splits=splits,
        coords=cfg.model.landmark_coords, seed=cfg.experiment.seed,
        landmarks_dir=landmarks_dir,
    )
    if "train" not in datasets or "dev" not in datasets:
        raise SystemExit("manifest must contain 'train' and 'dev' splits")

    if cfg.video_augment is not None and cfg.video_augment.enabled:
        datasets["train"].frame_transform = FrameAugment(cfg.video_augment, seed=cfg.experiment.seed)

    train_loader = build_dataloader(datasets["train"], cfg.training, shuffle=True)
    val_loader = build_dataloader(datasets["dev"], cfg.training, shuffle=False)

    model = DeepfakeClassifier(cfg.model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay
    )
    scheduler = build_scheduler(optimizer, cfg.training)
    class_weights = (
        datasets["train"].label_weights()
        if cfg.training.class_weight == "balanced"
        else None                       # "none" -> ordinary cross-entropy
    )

    run_dir = RunDirectory(cfg.experiment.output_root, cfg.experiment.name)
    run_dir.save_config(cfg)
    trainer = DeepfakeTrainer(
        model, optimizer, config=cfg.training, run_dir=run_dir, device=device,
        scheduler=scheduler, class_weights=class_weights,
    )
    if is_synthetic:
        trainer.logger.warning(
            "SYNTHETIC LANDMARKS - this run validates the pipeline only, not detection skill"
        )
    trainer.logger.info("model params: %s  variant: %s",
                        f"{count_parameters(model):,}", model.variant)

    summary = trainer.fit(train_loader, val_loader, epochs=args.epochs or cfg.training.epochs)

    best = run_dir.checkpoint_path("best")
    if best.is_file():
        load_checkpoint(best, model=model, map_location=device)

    eval_split = args.eval_split if args.eval_split in datasets else "dev"
    eval_loader = build_dataloader(datasets[eval_split], cfg.training, shuffle=False)
    report = _evaluate(model, eval_loader, device)
    report.update({
        "eval_split": eval_split, "synthetic": is_synthetic,
        "visual_encoder": cfg.model.visual_encoder, "regions": cfg.model.visual_regions,
        "num_frames": cfg.video.num_frames, "best_epoch": summary["best_epoch"],
        "seed": cfg.experiment.seed,
        "class_weight": cfg.training.class_weight,
        "scheduler": cfg.training.scheduler,
        "warmup_epochs": cfg.training.warmup_epochs,
        "selected_by": summary["monitor"],                 # e.g. "val_roc_auc"
        "selected_metric_value": summary["best_metric"],   # dev metric of best.pt
    })
    run_dir.write_json("report.json", report)

    if args.export_encoder:
        if model.variant != "transformer":
            trainer.logger.warning("--export-encoder ignored: variant is %s", model.variant)
        else:
            path = export_visual_encoder(
                run_dir.checkpoint_path("visual_encoder"),
                encoder=model.encoder, model_cfg=cfg.model, video_cfg=cfg.video,
                extra={"run": run_dir.path.name, "eval_report": report},
            )
            trainer.logger.info("exported shared visual encoder -> %s", path)

    print(f"\nrun dir   : {run_dir.path}")
    print(f"best {summary['monitor']} = {summary['best_metric']:.4f} @ epoch {summary['best_epoch']}")
    print(f"{eval_split} split: acc={report['accuracy']:.3f}  f1={report['f1']:.3f}  "
          f"auc={report['roc_auc']:.3f}  eer={report['eer']:.3f}")
    print(f"confusion [[TN, FP], [FN, TP]] = {report['confusion_matrix']}")
    if is_synthetic:
        print("\n(NOTE: synthetic landmarks - numbers reflect plumbing, not real performance)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
