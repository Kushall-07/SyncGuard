"""Trainer subclass for the video deepfake classifier (Phase 7).

Adds optional class-weighted cross-entropy for the Celeb-DF real/fake imbalance
and the anti-spoofing metric set (accuracy / F1 / ROC-AUC / EER) via
:func:`src.evaluation.metrics.binary_classification_report`. Landmark-space
augmentation is handled per-sample in the dataset's ``frame_transform`` hook
(:class:`src.preprocessing.video_augment.FrameAugment`), so - unlike
:class:`~src.training.spoof_trainer.SpoofTrainer` - there is no on-device
augmentation step here.
"""

from __future__ import annotations

import torch
from torch import nn

from src.evaluation.metrics import binary_classification_report
from src.training.trainer import Trainer

__all__ = ["DeepfakeTrainer"]


class DeepfakeTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = (
            class_weights.to(self.device) if class_weights is not None else None
        )
        if self.class_weights is not None:
            self.logger.info("class weights: %s", self.class_weights.tolist())

    def compute_loss(self, batch):
        features, target = batch
        features = features.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)

        logits = self.model(features)
        loss = nn.functional.cross_entropy(logits, target, weight=self.class_weights)
        return {"loss": loss, "logits": logits.detach().float(), "targets": target.detach()}

    def compute_metrics(self, step_outputs):
        logits = torch.cat([o["logits"] for o in step_outputs]).cpu()
        targets = torch.cat([o["targets"] for o in step_outputs]).cpu()
        probs = torch.softmax(logits, dim=1)[:, 1]          # P(real)
        preds = logits.argmax(dim=1)
        report = binary_classification_report(targets, preds, probs)
        return {
            "accuracy": report["accuracy"],
            "f1": report["f1"],
            "roc_auc": report["roc_auc"],
            "eer": report["eer"],
        }
