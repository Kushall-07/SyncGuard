"""Phase 5: head-to-head of the CNN baseline vs the CNN + Transformer encoder.

Trains ``model.audio_encoder = "cnn"`` and ``"cnn_transformer"`` on the *same*
manifest, split, seed and augmentation, then writes a comparison table. This is
the Phase 4-vs-Phase 5 experiment the project spec (section 24) asks for.

    python scripts/compare_audio_encoders.py --synthetic --epochs 5
    python scripts/compare_audio_encoders.py --manifest data/asvspoof/manifest.csv

Output: ``outputs/runs/compare-audio-encoders-<ts>/comparison.json`` plus a
per-variant run directory each.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
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
from src.models.audio import SpoofClassifier
from src.preprocessing.augment import SpecAugment, WaveformAugment
from src.training.checkpoint import load_checkpoint
from src.training.spoof_trainer import SpoofTrainer
from src.training.utils import RunDirectory, count_parameters, get_device, set_seed

SYNTHETIC_DIR = REPO_ROOT / "outputs" / "data" / "compare_encoders_synthetic"
VARIANTS = ("cnn", "cnn_transformer")


@torch.no_grad()
def _evaluate(model, loader, device) -> dict:
    model.eval()
    logits = torch.cat([model(mel.to(device)).float().cpu() for mel, _ in loader])
    targets = torch.cat([y for _, y in loader])
    probs = torch.softmax(logits, dim=1)[:, 1]
    return binary_classification_report(targets, logits.argmax(dim=1), probs)


def _run_variant(variant, cfg, datasets, device, epochs) -> dict:
    set_seed(cfg.experiment.seed, deterministic=cfg.experiment.deterministic)
    model_cfg = replace(cfg.model, audio_encoder=variant)
    model = SpoofClassifier(model_cfg, n_mels=cfg.audio.mel.n_mels)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate,
                            weight_decay=cfg.training.weight_decay)
    run_dir = RunDirectory(cfg.experiment.output_root, f"compare-{variant}")
    run_dir.save_config(replace(cfg, model=model_cfg))
    trainer = SpoofTrainer(
        model, opt, config=cfg.training, run_dir=run_dir, device=device,
        spec_augment=SpecAugment(cfg.augment.specaugment) if cfg.augment.enabled else None,
        class_weights=datasets["train"].label_weights(),
    )
    summary = trainer.fit(
        build_dataloader(datasets["train"], cfg.training, shuffle=True),
        build_dataloader(datasets["dev"], cfg.training, shuffle=False),
        epochs=epochs,
    )
    best = run_dir.checkpoint_path("best")
    if best.is_file():
        load_checkpoint(best, model=model, map_location=device)
    eval_split = "eval" if "eval" in datasets else "dev"
    report = _evaluate(model, build_dataloader(datasets[eval_split], cfg.training, shuffle=False),
                       device)
    return {
        "variant": variant,
        "params": count_parameters(model),
        "epochs_run": summary["epochs_run"],
        "best_monitor": summary["monitor"],
        "best_value": summary["best_metric"],
        "eval_split": eval_split,
        "eval": {k: report[k] for k in ("accuracy", "f1", "roc_auc", "eer")},
        "run_dir": run_dir.path.name,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "spoof_transformer.yaml")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--asvspoof-root", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--epochs", type=int, help="override training.epochs for both runs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    epochs = args.epochs or cfg.training.epochs

    if args.manifest:
        manifest, synthetic = Manifest.read_csv(args.manifest), False
    elif args.asvspoof_root:
        manifest, synthetic = build_asvspoof_manifest(args.asvspoof_root), False
    else:
        manifest = synthetic_manifest(
            SYNTHETIC_DIR, n_per_split={"train": 160, "dev": 48, "eval": 48},
            n_speakers_per_split=8, seed=cfg.experiment.seed,
            sample_rate=cfg.audio.sample_rate, duration_s=cfg.data.fixed_seconds,
        )
        synthetic = True

    splits = tuple(s for s in ("train", "dev", "eval") if s in manifest.split_sizes())
    datasets = make_asvspoof_datasets(manifest, cfg.audio, feature=cfg.data.feature,
                                      fixed_seconds=cfg.data.fixed_seconds, splits=splits,
                                      seed=cfg.experiment.seed)
    if cfg.augment.enabled:
        datasets["train"].waveform_transform = WaveformAugment(cfg.augment, seed=cfg.experiment.seed)

    results = [_run_variant(v, cfg, datasets, device, epochs) for v in VARIANTS]

    compare_dir = RunDirectory(cfg.experiment.output_root, "compare-audio-encoders")
    payload = {"synthetic": synthetic, "epochs": epochs, "seed": cfg.experiment.seed,
               "eval_split": results[0]["eval_split"], "results": results}
    compare_dir.write_json("comparison.json", payload)

    print(f"\n{'variant':<18}{'params':>12}{'acc':>8}{'f1':>8}{'auc':>8}{'eer':>8}")
    print("-" * 62)
    for r in results:
        e = r["eval"]
        print(f"{r['variant']:<18}{r['params']:>12,}{e['accuracy']:>8.3f}{e['f1']:>8.3f}"
              f"{e['roc_auc']:>8.3f}{e['eer']:>8.3f}")
    print(f"\ncomparison.json -> {compare_dir.path}")
    if synthetic:
        print("(synthetic audio - this compares plumbing + capacity, not real skill)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
