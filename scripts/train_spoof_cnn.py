"""Phase 4: train the CNN audio spoof-detection baseline.

Examples
--------
Plumbing run on generated audio (no dataset needed)::

    python scripts/train_spoof_cnn.py --synthetic --epochs 3

Real run once ASVspoof 2019 LA is downloaded and its manifest built::

    python scripts/train_spoof_cnn.py --manifest data/asvspoof/manifest.csv

Writes checkpoints, ``metrics.csv``, ``run.log`` and a final ``report.json``
(accuracy / precision / recall / F1 / ROC-AUC / EER / confusion matrix on the
evaluation split) under ``outputs/runs/<name>-<timestamp>/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.data.asvspoof_dataset import build_asvspoof_manifest, make_asvspoof_datasets
from src.data.audio_dataset import build_dataloader
from src.data.manifests import Manifest, synthetic_manifest
from src.evaluation.metrics import binary_classification_report
from src.models.audio import SpoofClassifier, export_audio_encoder
from src.preprocessing.augment import SpecAugment, WaveformAugment
from src.training.checkpoint import load_checkpoint
from src.training.spoof_trainer import SpoofTrainer
from src.training.utils import RunDirectory, count_parameters, get_device, set_seed

SYNTHETIC_DIR = REPO_ROOT / "outputs" / "data" / "spoof_cnn_synthetic"


def _resolve_manifest(args, cfg) -> tuple[Manifest, bool]:
    if args.manifest:
        return Manifest.read_csv(args.manifest), False
    if args.asvspoof_root:
        return build_asvspoof_manifest(args.asvspoof_root, out_csv=args.out_manifest), False
    if args.synthetic or not Path(cfg.data.manifest_csv).is_file():
        manifest = synthetic_manifest(
            SYNTHETIC_DIR,
            n_per_split={"train": 160, "dev": 48, "eval": 48},
            n_speakers_per_split=8,
            seed=cfg.experiment.seed,
            sample_rate=cfg.audio.sample_rate,
            duration_s=cfg.data.fixed_seconds,
        )
        return manifest, True
    return Manifest.read_csv(cfg.data.manifest_csv), False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "spoof_cnn.yaml")
    parser.add_argument("--manifest", type=Path, help="pre-built manifest CSV")
    parser.add_argument("--asvspoof-root", type=Path, help="LA/ directory; builds the manifest")
    parser.add_argument("--out-manifest", type=Path,
                        default=REPO_ROOT / "data" / "asvspoof" / "manifest.csv")
    parser.add_argument("--synthetic", action="store_true",
                        help="force a synthetic-audio plumbing run")
    parser.add_argument("--epochs", type=int, help="override training.epochs")
    parser.add_argument("--eval-split", default="eval", choices=["train", "dev", "eval"])
    parser.add_argument("--export-encoder", action="store_true",
                        help="after training, write the shared audio encoder to "
                             "<run>/checkpoints/audio_encoder.pt (reloads best.pt first)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.experiment.seed, deterministic=cfg.experiment.deterministic)
    device = get_device()

    manifest, is_synthetic = _resolve_manifest(args, cfg)
    datasets = make_asvspoof_datasets(
        manifest, cfg.audio,
        feature=cfg.data.feature,
        fixed_seconds=cfg.data.fixed_seconds,
        splits=tuple(s for s in ("train", "dev", args.eval_split) if s in manifest.split_sizes()),
        seed=cfg.experiment.seed,
    )
    if "train" not in datasets or "dev" not in datasets:
        raise SystemExit("manifest must contain 'train' and 'dev' splits")

    if cfg.augment.enabled:
        datasets["train"].waveform_transform = WaveformAugment(cfg.augment, seed=cfg.experiment.seed)
    spec_aug = SpecAugment(cfg.augment.specaugment) if cfg.augment.enabled else None

    train_loader = build_dataloader(datasets["train"], cfg.training, shuffle=True)
    val_loader = build_dataloader(datasets["dev"], cfg.training, shuffle=False)

    model = SpoofClassifier(cfg.model, n_mels=cfg.audio.mel.n_mels)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay
    )

    run_dir = RunDirectory(cfg.experiment.output_root, cfg.experiment.name)
    run_dir.save_config(cfg)
    trainer = SpoofTrainer(
        model, optimizer, config=cfg.training, run_dir=run_dir, device=device,
        spec_augment=spec_aug, class_weights=datasets["train"].label_weights(),
    )
    if is_synthetic:
        trainer.logger.warning(
            "SYNTHETIC AUDIO - this run validates the pipeline only, not spoof-detection skill"
        )
    trainer.logger.info("model params: %s", f"{count_parameters(model):,}")

    summary = trainer.fit(train_loader, val_loader, epochs=args.epochs or cfg.training.epochs)

    # Evaluate (and export) the best checkpoint, not the last-epoch weights.
    best = run_dir.checkpoint_path("best")
    if best.is_file():
        load_checkpoint(best, model=model, map_location=device)

    # ---- final report on the evaluation split --------------------------
    eval_split = args.eval_split if args.eval_split in datasets else "dev"
    eval_loader = build_dataloader(datasets[eval_split], cfg.training, shuffle=False)
    report = _evaluate(model, eval_loader, device)
    report["eval_split"] = eval_split
    report["synthetic"] = is_synthetic
    report["audio_encoder"] = cfg.model.audio_encoder
    report["best_epoch"] = summary["best_epoch"]
    run_dir.write_json("report.json", report)

    if args.export_encoder:
        export_path = export_audio_encoder(
            run_dir.checkpoint_path("audio_encoder"),
            encoder=model.encoder, model_cfg=cfg.model, audio_cfg=cfg.audio,
            n_mels=cfg.audio.mel.n_mels,
            extra={"run": run_dir.path.name, "eval_report": report},
        )
        trainer.logger.info("exported shared audio encoder -> %s", export_path)

    print(f"\nrun dir   : {run_dir.path}")
    print(f"best {summary['monitor']} = {summary['best_metric']:.4f} @ epoch {summary['best_epoch']}")
    print(f"{eval_split} split: acc={report['accuracy']:.3f}  f1={report['f1']:.3f}  "
          f"auc={report['roc_auc']:.3f}  eer={report['eer']:.3f}")
    print(f"confusion [[TN, FP], [FN, TP]] = {report['confusion_matrix']}")
    if is_synthetic:
        print("\n(NOTE: synthetic audio - numbers reflect plumbing, not real performance)")
    return 0


@torch.no_grad()
def _evaluate(model: torch.nn.Module, loader, device) -> dict:
    model.eval()
    all_logits, all_targets = [], []
    for mel, target in loader:
        logits = model(mel.to(device))
        all_logits.append(logits.float().cpu())
        all_targets.append(target)
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    probs = torch.softmax(logits, dim=1)[:, 1]
    return binary_classification_report(targets, logits.argmax(dim=1), probs)


if __name__ == "__main__":
    raise SystemExit(main())
