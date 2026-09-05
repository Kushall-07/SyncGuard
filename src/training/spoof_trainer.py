"""Trainer subclass for the audio-only spoof classifier (Phase 4).

Adds on-device SpecAugment (training batches only), optional class-weighted
cross-entropy for the ASVspoof bonafide/spoof imbalance, and the anti-spoofing
metric set (accuracy / F1 / ROC-AUC / EER) via
:func:`src.evaluation.metrics.binary_classification_report`.
"""

from __future__ import annotations

import torch
from torch import nn

from src.evaluation.metrics import binary_classification_report
from src.preprocessing.augment import SpecAugment
from src.training.trainer import Trainer

__all__ = ["SpoofTrainer"]


class SpoofTrainer(Trainer):
    def __init__(
        self,
        *args,
        spec_augment: SpecAugment | None = None,
        class_weights: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.spec_augment = spec_augment
        self.class_weights = (
            class_weights.to(self.device) if class_weights is not None else None
        )
        if self.class_weights is not None:
            self.logger.info("class weights: %s", self.class_weights.tolist())

    def compute_loss(self, batch):
        mel, target = batch
        mel = mel.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)

        if self.model.training and self.spec_augment is not None:
            mel = self.spec_augment(mel)

        logits = self.model(mel)
        loss = nn.functional.cross_entropy(logits, target, weight=self.class_weights)
        return {"loss": loss, "logits": logits.detach().float(), "targets": target.detach()}

    def compute_metrics(self, step_outputs):
        logits = torch.cat([o["logits"] for o in step_outputs]).cpu()
        targets = torch.cat([o["targets"] for o in step_outputs]).cpu()
        probs = torch.softmax(logits, dim=1)[:, 1]          # P(bonafide)
        preds = logits.argmax(dim=1)
        report = binary_classification_report(targets, preds, probs)
        return {
            "accuracy": report["accuracy"],
            "f1": report["f1"],
            "roc_auc": report["roc_auc"],
            "eer": report["eer"],
        }
