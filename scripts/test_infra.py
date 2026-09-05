"""Phase 2B smoke test: config -> seed -> run dir -> Trainer -> checkpoints -> metrics.

Run from the repository root:

    python scripts/test_infra.py

Trains a tiny MLP on a synthetic linearly-separable-ish binary dataset for a few
epochs through the real :class:`~src.training.trainer.Trainer`, then checks that
the run directory, ``config.yaml``, ``metrics.csv``, ``run.log``, ``summary.json``
and both checkpoints were produced, that a checkpoint round-trips, and that the
metrics module returns a sane report. Exit code is non-zero on any failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.evaluation.metrics import binary_classification_report
from src.training.checkpoint import load_checkpoint
from src.training.trainer import Trainer
from src.training.utils import RunDirectory, get_device, set_seed

CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _make_dataset(n: int, dim: int, seed: int) -> TensorDataset:
    # The decision boundary `w` is fixed (seed 0) so train and val share it;
    # only the sampled points differ between splits.
    w = torch.randn(dim, generator=torch.Generator().manual_seed(0))
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, dim, generator=g)
    logits = x @ w + 0.3 * torch.randn(n, generator=g)
    y = (logits > 0).long()
    return TensorDataset(x, y)


class _MLPTrainer(Trainer):
    def compute_loss(self, batch):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        logits = self.model(x)
        loss = nn.functional.cross_entropy(logits, y)
        return {"loss": loss, "logits": logits.detach(), "targets": y.detach()}

    def compute_metrics(self, step_outputs):
        logits = torch.cat([o["logits"] for o in step_outputs])
        targets = torch.cat([o["targets"] for o in step_outputs])
        preds = logits.argmax(dim=1)
        report = binary_classification_report(
            targets, preds, torch.softmax(logits, dim=1)[:, 1]
        )
        return {"accuracy": report["accuracy"], "f1": report["f1"], "auc": report["roc_auc"]}


def main() -> int:
    cfg = load_config(CONFIG_PATH)
    _check(cfg.audio.sample_rate == 16000, "audio section not resolved from configs/audio.yaml")
    _check(cfg.training.monitor == "val_loss", "training.monitor default mismatch")

    set_seed(cfg.experiment.seed, deterministic=cfg.experiment.deterministic)
    device = get_device()

    dim = 16
    train_ds = _make_dataset(512, dim, seed=cfg.experiment.seed)
    val_ds = _make_dataset(128, dim, seed=cfg.experiment.seed + 1)
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size)

    model = nn.Sequential(nn.Linear(dim, 32), nn.ReLU(), nn.Linear(32, 2))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay
    )

    run_dir = RunDirectory(cfg.experiment.output_root, "phase2b-smoke")
    run_dir.save_config(cfg)

    trainer = _MLPTrainer(model, optimizer, config=cfg.training, run_dir=run_dir, device=device)
    summary = trainer.fit(train_loader, val_loader, epochs=4)

    # --- artifact checks -----------------------------------------------------
    for artifact in (run_dir.config_path, run_dir.metrics_path, run_dir.log_path,
                     run_dir.path / "summary.json",
                     run_dir.checkpoint_path("best"), run_dir.checkpoint_path("last")):
        _check(artifact.is_file(), f"missing artifact: {artifact}")

    metrics_rows = run_dir.metrics_path.read_text(encoding="utf-8").strip().splitlines()
    _check(len(metrics_rows) == 1 + summary["epochs_run"],
           f"metrics.csv has {len(metrics_rows)} lines, expected {1 + summary['epochs_run']}")
    _check("val_accuracy" in metrics_rows[0], "metrics.csv missing val_accuracy column")

    # --- checkpoint round-trip --------------------------------------------
    fresh = nn.Sequential(nn.Linear(dim, 32), nn.ReLU(), nn.Linear(32, 2))
    ckpt = load_checkpoint(run_dir.checkpoint_path("best"), model=fresh, map_location="cpu")
    _check(ckpt["epoch"] == summary["best_epoch"], "best checkpoint epoch mismatch")
    xb, _ = next(iter(val_loader))
    with torch.no_grad():
        ref = model.cpu()(xb)
        got = fresh(xb)
    _check(torch.allclose(ref, got, atol=1e-5), "restored model produces different outputs")

    print(f"\nrun dir      : {run_dir.path}")
    print(f"epochs run   : {summary['epochs_run']}")
    print(f"best {summary['monitor']:<9}: {summary['best_metric']:.4f} @ epoch {summary['best_epoch']}")
    print(f"final val    : acc={trainer.history[-1].get('val_accuracy'):.3f} "
          f"f1={trainer.history[-1].get('val_f1'):.3f} auc={trainer.history[-1].get('val_auc'):.3f}")
    print("\nPhase 2B smoke test: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"Phase 2B smoke test: FAIL\n  {exc}", file=sys.stderr)
        raise SystemExit(1)
