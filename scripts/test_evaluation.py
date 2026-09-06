"""Phase 6 smoke test: train briefly, then run the full evaluation + figures.

    python scripts/test_evaluation.py

Trains a small spoof classifier for a few epochs, evaluates the best checkpoint
with ``evaluate_spoof_model`` (scalar report + per-attack EER), renders every
figure, and plots training curves from the run's ``metrics.csv``. Exit code is
non-zero on any failure.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.data.asvspoof_dataset import make_asvspoof_datasets
from src.data.audio_dataset import build_dataloader
from src.data.manifests import synthetic_manifest
from src.evaluation.spoof_eval import evaluate_spoof_model
from src.evaluation.visualization import plot_training_curves, save_spoof_evaluation_figures
from src.models.audio import SpoofClassifier
from src.preprocessing.augment import SpecAugment
from src.training.checkpoint import load_checkpoint
from src.training.spoof_trainer import SpoofTrainer
from src.training.utils import RunDirectory, get_device, set_seed

DATA_DIR = REPO_ROOT / "outputs" / "data" / "evaluation_smoke"
FIXED_SECONDS = 1.0


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> int:
    cfg = load_config(REPO_ROOT / "configs" / "spoof_cnn.yaml")
    cfg = replace(
        cfg,
        model=replace(cfg.model, audio_cnn_channels=(16, 32), audio_embedding_dim=64,
                      spoof_head_hidden=32),
        data=replace(cfg.data, fixed_seconds=FIXED_SECONDS),
        training=replace(cfg.training, batch_size=8, num_workers=0, epochs=4,
                         early_stopping_patience=0),
    )
    set_seed(cfg.experiment.seed)
    device = get_device()
    n_mels = cfg.audio.mel.n_mels

    manifest = synthetic_manifest(
        DATA_DIR, n_per_split={"train": 64, "dev": 32, "eval": 96}, n_speakers_per_split=8,
        seed=cfg.experiment.seed, sample_rate=cfg.audio.sample_rate, duration_s=FIXED_SECONDS,
    )
    datasets = make_asvspoof_datasets(manifest, cfg.audio, feature="logmel",
                                      fixed_seconds=FIXED_SECONDS, splits=("train", "dev", "eval"))

    run_dir = RunDirectory(cfg.experiment.output_root, "phase6-smoke")
    run_dir.save_config(cfg)
    model = SpoofClassifier(cfg.model, n_mels=n_mels)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = SpoofTrainer(model, opt, config=cfg.training, run_dir=run_dir, device=device,
                           spec_augment=SpecAugment(cfg.augment.specaugment),
                           class_weights=datasets["train"].label_weights())
    trainer.fit(
        build_dataloader(datasets["train"], cfg.training, shuffle=True),
        build_dataloader(datasets["dev"], cfg.training, shuffle=False),
        epochs=cfg.training.epochs,
    )

    load_checkpoint(run_dir.checkpoint_path("best"), model=model, map_location=device)
    eval_loader = build_dataloader(datasets["eval"], cfg.training, shuffle=False)
    result = evaluate_spoof_model(model, eval_loader, device)

    _check(result.n_samples == 96, f"expected 96 eval samples, got {result.n_samples}")
    _check(result.y_score.min() >= 0 and result.y_score.max() <= 1, "scores out of [0, 1]")
    _check({"accuracy", "f1", "roc_auc", "eer", "eer_threshold"} <= set(result.report),
           "report missing keys")
    _check(len(result.per_attack) >= 1, "per-attack EER breakdown is empty")
    for name, d in result.per_attack.items():
        _check({"n_spoof", "eer", "threshold"} <= set(d), f"per-attack entry {name} malformed")

    eval_dir = run_dir.path / "eval"
    figures = save_spoof_evaluation_figures(
        eval_dir, y_true=result.y_true, y_score=result.y_score,
        confusion=result.report["confusion_matrix"], prefix="eval",
    )
    curves = plot_training_curves(run_dir.metrics_path, eval_dir / "training_curves.png")
    for p in [*figures.values(), curves]:
        _check(p.is_file() and p.stat().st_size > 0, f"figure not written: {p}")

    print(f"\n{result.summary_line()}")
    print(f"per-attack EER: " + ", ".join(
        f"{k}={v['eer'] * 100:.1f}%" for k, v in sorted(result.per_attack.items())))
    print(f"figures       : {len(figures) + 1} PNGs -> {eval_dir}")
    print("\nPhase 6 smoke test: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"Phase 6 smoke test: FAIL\n  {exc}", file=sys.stderr)
        raise SystemExit(1)
