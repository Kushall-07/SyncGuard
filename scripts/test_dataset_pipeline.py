"""Phase 3 smoke test: synthetic manifest -> Dataset -> DataLoader -> Trainer.

Run from the repository root:

    python scripts/test_dataset_pipeline.py

Builds a speaker-disjoint synthetic manifest of generated WAVs, round-trips it
through CSV, wraps the splits in ``ManifestAudioDataset``, checks batch shapes for
both feature modes, verifies no speaker leaks across splits (and that the leakage
guard actually fires), and trains a tiny classifier for one epoch through the
real Phase 2B ``Trainer``. Exit code is non-zero on any failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.data.asvspoof_dataset import make_asvspoof_datasets
from src.data.audio_dataset import ManifestAudioDataset, build_dataloader
from src.data.manifests import LeakageError, Manifest, assert_speaker_disjoint, synthetic_manifest
from src.evaluation.metrics import accuracy
from src.training.trainer import Trainer
from src.training.utils import RunDirectory, get_device, set_seed

DATA_DIR = REPO_ROOT / "outputs" / "data" / "synthetic_asvspoof"
FIXED_SECONDS = 1.0


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


class _AudioClsTrainer(Trainer):
    def compute_loss(self, batch):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        if x.dim() == 3:  # [B, n_mels, T] -> add channel
            x = x.unsqueeze(1)
        logits = self.model(x)
        return {"loss": nn.functional.cross_entropy(logits, y),
                "logits": logits.detach(), "targets": y.detach()}

    def compute_metrics(self, step_outputs):
        logits = torch.cat([o["logits"] for o in step_outputs])
        targets = torch.cat([o["targets"] for o in step_outputs])
        return {"accuracy": accuracy(targets, logits.argmax(dim=1))}


def _tiny_cnn(n_mels: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(), nn.Linear(8 * 4 * 4, 2),
    )


def main() -> int:
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    set_seed(cfg.experiment.seed)
    device = get_device()

    # --- 1. synthetic manifest + CSV round-trip ----------------------------
    manifest = synthetic_manifest(
        DATA_DIR, n_per_split={"train": 40, "dev": 16}, n_speakers_per_split=4,
        seed=cfg.experiment.seed, sample_rate=cfg.audio.sample_rate, duration_s=FIXED_SECONDS,
    )
    csv_path = DATA_DIR / "manifest.csv"
    manifest.write_csv(csv_path)
    reloaded = Manifest.read_csv(csv_path)
    _check(len(reloaded) == len(manifest), "CSV round-trip changed row count")
    _check(reloaded.rows[0] == manifest.rows[0], "CSV round-trip changed the first row")

    # --- 2. leakage guards ----------------------------------------------
    reloaded.assert_splits_speaker_disjoint()
    try:
        assert_speaker_disjoint(a=reloaded.split("train"), b=reloaded.split("train"))
    except LeakageError:
        pass
    else:
        raise AssertionError("assert_speaker_disjoint failed to detect an obvious overlap")

    # --- 3. datasets via the ASVspoof factory (works on any manifest) -----
    datasets = make_asvspoof_datasets(
        reloaded, cfg.audio, feature="logmel", fixed_seconds=FIXED_SECONDS, splits=("train", "dev"),
    )
    _check(datasets["train"].random_crop and not datasets["dev"].random_crop,
           "random_crop should be on for train only")

    expected_t = 1 + int(round(FIXED_SECONDS * cfg.audio.sample_rate)) // cfg.audio.mel.hop_length

    # --- 4. batch shapes for both feature modes -------------------------
    mel_loader = build_dataloader(datasets["train"], cfg.training, shuffle=True)
    xb, yb = next(iter(mel_loader))
    _check(tuple(xb.shape) == (cfg.training.batch_size, cfg.audio.mel.n_mels, expected_t),
           f"logmel batch shape {tuple(xb.shape)} unexpected")
    _check(yb.dtype == torch.int64 and set(yb.tolist()) <= {0, 1}, "labels malformed")

    wav_ds = ManifestAudioDataset(reloaded.split("dev"), cfg.audio, feature="waveform",
                                  fixed_seconds=FIXED_SECONDS)
    wb, _ = next(iter(build_dataloader(wav_ds, cfg.training, shuffle=False)))
    _check(tuple(wb.shape) == (cfg.training.batch_size, 1,
                               int(round(FIXED_SECONDS * cfg.audio.sample_rate))),
           f"waveform batch shape {tuple(wb.shape)} unexpected")

    # --- 5. stratified subset ----------------------------------------------
    sub = reloaded.split("train").subset(10, seed=1, stratify_by="label")
    _check(len(sub) == 10, f"subset returned {len(sub)} rows, expected 10")
    _check(set(r.label for r in sub) == {0, 1}, "stratified subset lost a class")

    # --- 6. one real training epoch --------------------------------------
    run_dir = RunDirectory(cfg.experiment.output_root, "phase3-smoke")
    run_dir.save_config(cfg)
    model = _tiny_cnn(cfg.audio.mel.n_mels)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = _AudioClsTrainer(model, opt, config=cfg.training, run_dir=run_dir, device=device)
    summary = trainer.fit(
        build_dataloader(datasets["train"], cfg.training, shuffle=True),
        build_dataloader(datasets["dev"], cfg.training, shuffle=False),
        epochs=2,
    )
    _check(run_dir.checkpoint_path("last").is_file(), "trainer did not checkpoint")

    print(f"\nmanifest      : {manifest!r}")
    print(f"train batch   : logmel {tuple(xb.shape)}  |  waveform {tuple(wb.shape)}")
    print(f"epochs run    : {summary['epochs_run']}  best {summary['monitor']}={summary['best_metric']:.4f}")
    print(f"artifacts     : {run_dir.path}")
    print("\nPhase 3 smoke test: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"Phase 3 smoke test: FAIL\n  {exc}", file=sys.stderr)
        raise SystemExit(1)
