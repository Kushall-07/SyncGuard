"""Run a trained deepfake classifier over a dataset and assemble the report (Phase 7).

The visual analogue of :mod:`src.evaluation.spoof_eval`. ``evaluate_deepfake_model``
returns per-video scores/targets plus the scalar report and, when the dataset
carries an ``attack`` column, a per-synthesis-method EER breakdown (Celeb-DF has
one method, ``celebdf-fs``, so this is a single row vs the pooled real set - the
hook is kept for future datasets / per-identity slicing). It relies on the eval
loader being un-shuffled and non-dropping so scores line up with
``dataset.manifest``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.celebdf_dataset import CelebDFLandmarkDataset
from src.evaluation.metrics import binary_classification_report, per_attack_eer
from src.evaluation.spoof_eval import loader_preserves_order

__all__ = ["DeepfakeEvalResult", "evaluate_deepfake_model"]


@dataclass
class DeepfakeEvalResult:
    y_true: np.ndarray
    y_pred: np.ndarray
    y_score: np.ndarray                       # P(real)
    report: dict[str, Any]
    per_attack: dict[str, dict[str, float]] = field(default_factory=dict)
    n_samples: int = 0

    def summary_line(self) -> str:
        r = self.report
        return (f"n={self.n_samples}  acc={r['accuracy']:.4f}  f1={r['f1']:.4f}  "
                f"auc={r.get('roc_auc', float('nan')):.4f}  "
                f"eer={r.get('eer', float('nan')) * 100:.2f}%")


@torch.no_grad()
def evaluate_deepfake_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str | torch.device,
    *,
    groups: list[str] | None = None,
) -> DeepfakeEvalResult:
    """Forward every batch, collect ``P(real)`` scores, build the report.

    ``groups`` (per-video labels aligned with the loader order) enables the
    per-group EER breakdown; if omitted it is taken from the loader's
    :class:`CelebDFLandmarkDataset` ``attack`` column when order is preserved.
    """

    model.eval()
    logits_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    for features, target in loader:
        logits_chunks.append(model(features.to(device)).float().cpu())
        target_chunks.append(target.reshape(-1))

    logits = torch.cat(logits_chunks)
    y_true = torch.cat(target_chunks).numpy().astype(int)
    y_score = torch.softmax(logits, dim=1)[:, 1].numpy()
    y_pred = logits.argmax(dim=1).numpy()

    report = binary_classification_report(y_true, y_pred, y_score)

    ds = getattr(loader, "dataset", None)
    if groups is None and isinstance(ds, CelebDFLandmarkDataset) and loader_preserves_order(loader):
        if len(ds) == len(y_true):
            groups = [row.attack for row in ds.manifest]

    per_attack: dict[str, dict[str, float]] = {}
    if groups is not None and len(groups) == len(y_true) and (y_true == 1).any():
        per_attack = per_attack_eer(groups, y_true, y_score)

    return DeepfakeEvalResult(
        y_true=y_true, y_pred=y_pred, y_score=y_score,
        report=report, per_attack=per_attack, n_samples=int(y_true.size),
    )
