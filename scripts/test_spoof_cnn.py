"""Phase 4 smoke test: CNN spoof baseline end-to-end on synthetic audio.

    python scripts/test_spoof_cnn.py

Builds a small speaker-disjoint synthetic manifest, trains
``SpoofClassifier`` for a few epochs through ``SpoofTrainer`` (with SpecAugment +
waveform augmentation on the train split), and checks encoder/head shapes, that
the loss decreases, that all run artifacts including ``report.json`` are written,
and that a CUDA forward works. Exit code is non-zero on any failure.
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
from src.data.audio_dataset import build_dataloader
from src.data.manifests import synthetic_manifest
from src.data.asvspoof_dataset import make_asvspoof_datasets
from src.evaluation.metrics import binary_classification_report
from src.models.audio import AudioCNNEncoder, SpoofClassifier
from src.models.heads.spoof_head import SpoofHead
from src.preprocessing.augment import SpecAugment, WaveformAugment
from src.training.spoof_trainer import SpoofTrainer
from src.training.utils import RunDirectory, count_parameters, get_device, set_seed

DATA_DIR = REPO_ROOT / "outputs" / "data" / "spoof_cnn_smoke"
FIXED_SECONDS = 1.0


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> int:
    cfg = load_config(REPO_ROOT / "configs" / "spoof_cnn.yaml")
    cfg = replace(
        cfg,
        data=replace(cfg.data, fixed_seconds=FIXED_SECONDS),
        training=replace(cfg.training, batch_size=8, num_workers=0, epochs=4,
                         early_stopping_patience=0),
    )
    set_seed(cfg.experiment.seed)
    device = get_device()
    n_mels = cfg.audio.mel.n_mels

    # --- unit-ish shape checks -------------------------------------------
    encoder = AudioCNNEncoder(cfg.model, n_mels)
    dummy_mel = torch.randn(3, n_mels, 101)
    out = encoder(dummy_mel)
    t_expected = -(-101 // encoder.time_downsample)  # ceil
    _check(out.tokens.shape == (3, t_expected, cfg.model.audio_embedding_dim),
           f"encoder tokens {tuple(out.tokens.shape)} unexpected (T'={t_expected})")
    head = SpoofHead.from_config(cfg.model, cfg.model.audio_embedding_dim)
    _check(head(out.tokens).shape == (3, 2), "spoof head should emit [B, 2]")

    model = SpoofClassifier(cfg.model, n_mels=n_mels)
    _check(model(dummy_mel).shape == (3, 2), "SpoofClassifier forward shape wrong")
    n_params = count_parameters(model)
    _check(n_params < 5_000_000, f"baseline unexpectedly large: {n_params:,} params")

    # --- SpecAugment behaviour ------------------------------------------
    spec = SpecAugment(cfg.augment.specaugment, mask_value=0.0)
    masked = spec(torch.ones(4, n_mels, 60))
    _check(masked.shape == (4, n_mels, 60), "SpecAugment changed shape")
    _check((masked == 0.0).any().item(), "SpecAugment masked nothing")
    _check((masked == 1.0).any().item(), "SpecAugment masked everything")

    # --- data --------------------------------------------------------------
    manifest = synthetic_manifest(
        DATA_DIR, n_per_split={"train": 48, "dev": 24}, n_speakers_per_split=6,
        seed=cfg.experiment.seed, sample_rate=cfg.audio.sample_rate, duration_s=FIXED_SECONDS,
    )
    datasets = make_asvspoof_datasets(manifest, cfg.audio, feature="logmel",
                                      fixed_seconds=FIXED_SECONDS, splits=("train", "dev"))
    datasets["train"].waveform_transform = WaveformAugment(cfg.augment, seed=1)

    train_loader = build_dataloader(datasets["train"], cfg.training, shuffle=True)
    val_loader = build_dataloader(datasets["dev"], cfg.training, shuffle=False)

    xb, yb = next(iter(train_loader))
    _check(xb.shape[1:] == (n_mels, 1 + int(FIXED_SECONDS * cfg.audio.sample_rate) //
                            cfg.audio.mel.hop_length), f"train batch mel shape {tuple(xb.shape)}")

    # --- train -----------------------------------------------------------
    run_dir = RunDirectory(cfg.experiment.output_root, "phase4-smoke")
    run_dir.save_config(cfg)
    model = SpoofClassifier(cfg.model, n_mels=n_mels)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = SpoofTrainer(model, opt, config=cfg.training, run_dir=run_dir, device=device,
                           spec_augment=SpecAugment(cfg.augment.specaugment),
                           class_weights=datasets["train"].label_weights())
    summary = trainer.fit(train_loader, val_loader, epochs=cfg.training.epochs)

    losses = [row["train_loss"] for row in trainer.history]
    _check(losses[-1] < losses[0], f"train loss did not decrease: {losses}")
    for name in ("config.yaml", "metrics.csv", "run.log", "summary.json"):
        _check((run_dir.path / name).is_file(), f"missing {name}")
    _check(run_dir.checkpoint_path("best").is_file(), "no best checkpoint")

    # --- report + CUDA forward ----------------------------------------
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(mel.to(device)).float().cpu() for mel, _ in val_loader])
    targets = torch.cat([y for _, y in val_loader])
    report = binary_classification_report(
        targets, logits.argmax(1), torch.softmax(logits, 1)[:, 1]
    )
    _check({"accuracy", "f1", "roc_auc", "eer", "confusion_matrix"} <= set(report),
           "report missing keys")

    print(f"\nparams        : {n_params:,}")
    print(f"encoder tokens: {tuple(out.tokens.shape)}  (downsample x{encoder.time_downsample})")
    print(f"train loss    : {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"best {summary['monitor']:<8}: {summary['best_metric']:.4f} @ epoch {summary['best_epoch']}")
    print(f"dev report    : acc={report['accuracy']:.3f} f1={report['f1']:.3f} "
          f"auc={report['roc_auc']:.3f} eer={report['eer']:.3f}")
    print("\nPhase 4 smoke test: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"Phase 4 smoke test: FAIL\n  {exc}", file=sys.stderr)
        raise SystemExit(1)
