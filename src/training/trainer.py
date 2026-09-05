"""Framework-light training loop base class (Phase 2B).

:class:`Trainer` owns the mechanics every training phase repeats - epoch loop,
mixed precision, gradient accumulation and clipping, per-epoch metric logging to
``metrics.csv``, best/last checkpointing, and early stopping. Task specifics live
in a subclass that implements :meth:`Trainer.compute_loss` (and optionally
:meth:`Trainer.compute_metrics`).

Example
-------
::

    class SpoofTrainer(Trainer):
        def compute_loss(self, batch):
            x, y = batch
            logits = self.model(x.to(self.device))
            loss = F.cross_entropy(logits, y.to(self.device))
            return {"loss": loss, "logits": logits.detach(), "targets": y}

    trainer = SpoofTrainer(model, optimizer, config=cfg.training,
                           run_dir=run_dir, device=device)
    summary = trainer.fit(train_loader, val_loader)
"""

from __future__ import annotations

import csv
import logging
import math
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import TrainingConfig
from src.training.checkpoint import save_checkpoint
from src.training.utils import RunDirectory, count_parameters

__all__ = ["Trainer"]


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        config: TrainingConfig,
        run_dir: RunDirectory,
        device: str | torch.device,
        scheduler: Any | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.run_dir = run_dir
        self.logger = logger or run_dir.get_logger()

        self.amp_enabled = bool(config.amp) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.amp_enabled)

        self._csv_columns: list[str] | None = None
        self.history: list[dict[str, float]] = []
        self.best_metric: float | None = None
        self.best_epoch: int = -1

        self.logger.info(
            "Trainer ready | device=%s | params=%s | amp=%s | grad_accum=%d",
            self.device, f"{count_parameters(self.model):,}", self.amp_enabled,
            config.grad_accum_steps,
        )

    # ----------------------------------------------------------------- overrides

    def compute_loss(self, batch: Any) -> Mapping[str, Any]:
        """Return a mapping with at least ``{"loss": <scalar tensor>}``.

        Optionally also return ``logits``/``targets`` (or any keys your
        :meth:`compute_metrics` expects); non-loss values should be detached.
        """

        raise NotImplementedError

    def compute_metrics(self, step_outputs: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        """Aggregate per-step outputs from one validation epoch into scalar metrics.

        Default: no extra metrics (the loop always logs ``train_loss`` /
        ``val_loss``). Override to add accuracy, EER, etc.
        """

        return {}

    # --------------------------------------------------------------------- loop

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        *,
        epochs: int | None = None,
    ) -> dict[str, Any]:
        epochs = epochs or self.config.epochs
        monitor = self.config.monitor
        mode = self.config.monitor_mode
        patience = self.config.early_stopping_patience
        stale_epochs = 0
        warned_missing_monitor = False

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader, epoch)
            row: dict[str, float] = {"epoch": epoch, "train_loss": train_loss}

            if val_loader is not None:
                val_loss, val_metrics = self._eval_epoch(val_loader)
                row["val_loss"] = val_loss
                row.update({f"val_{k}": v for k, v in val_metrics.items()})

            if self.scheduler is not None:
                self.scheduler.step()
                row["lr"] = self.optimizer.param_groups[0]["lr"]

            self._append_metrics_row(row)
            self.history.append(row)
            self.logger.info("epoch %d | %s", epoch, self._format_row(row))

            improved = False
            if monitor in row:
                improved = self._is_improvement(row[monitor], mode)
                if improved:
                    self.best_metric = row[monitor]
                    self.best_epoch = epoch
                    stale_epochs = 0
                else:
                    stale_epochs += 1
            elif not warned_missing_monitor:
                self.logger.warning(
                    "monitor %r not in logged metrics %s - best/early-stop disabled",
                    monitor, sorted(row),
                )
                warned_missing_monitor = True

            self._save(epoch, tag="last")
            if improved:
                self._save(epoch, tag="best")

            if patience and monitor in row and stale_epochs >= patience:
                self.logger.info(
                    "early stopping at epoch %d (no %s improvement for %d epochs)",
                    epoch, monitor, patience,
                )
                break

        summary = {
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "monitor": monitor,
            "epochs_run": len(self.history),
            "history": self.history,
        }
        self.run_dir.write_json("summary.json", summary)
        return summary

    # ------------------------------------------------------------------ epochs

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        accum = max(1, self.config.grad_accum_steps)
        clip = self.config.grad_clip_norm
        n_batches = len(loader) if _has_len(loader) else None

        total_loss = 0.0
        seen = 0
        self.optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(loader):
            with torch.amp.autocast(self.device.type, enabled=self.amp_enabled):
                out = self.compute_loss(batch)
                loss = out["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch} batch {i}: {loss.item()}")

            self.scaler.scale(loss / accum).backward()
            is_step = ((i + 1) % accum == 0) or (n_batches is not None and i + 1 == n_batches)
            if is_step:
                if clip and clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()
            seen += 1

        return total_loss / max(1, seen)

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> tuple[float, dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        seen = 0
        step_outputs: list[Mapping[str, Any]] = []

        for batch in loader:
            with torch.amp.autocast(self.device.type, enabled=self.amp_enabled):
                out = self.compute_loss(batch)
            total_loss += out["loss"].item()
            seen += 1
            step_outputs.append({k: v for k, v in out.items() if k != "loss"})

        metrics = self.compute_metrics(step_outputs) if step_outputs else {}
        return total_loss / max(1, seen), metrics

    # ------------------------------------------------------------------ helpers

    def _is_improvement(self, value: float, mode: str) -> bool:
        if self.best_metric is None:
            return True
        if mode == "min":
            return value < self.best_metric
        return value > self.best_metric

    def _save(self, epoch: int, *, tag: str) -> None:
        save_checkpoint(
            self.run_dir.checkpoint_path(tag),
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            best_metric=self.best_metric,
            extra={"monitor": self.config.monitor, "tag": tag},
        )

    def _append_metrics_row(self, row: Mapping[str, float]) -> None:
        path = self.run_dir.metrics_path
        new_file = not path.exists()
        if self._csv_columns is None:
            self._csv_columns = list(row.keys())
        for key in row:
            if key not in self._csv_columns:
                self._csv_columns.append(key)
        with path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._csv_columns)
            if new_file:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in self._csv_columns})

    @staticmethod
    def _format_row(row: Mapping[str, float]) -> str:
        parts = []
        for k, v in row.items():
            if k == "epoch":
                continue
            parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
        return " | ".join(parts)


def _has_len(obj: Iterable[Any]) -> bool:
    try:
        len(obj)  # type: ignore[arg-type]
        return True
    except TypeError:
        return False
