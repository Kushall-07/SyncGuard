"""Run a trained spoof classifier over a dataset and assemble the full report (Phase 6).

:func:`evaluate_spoof_model` returns per-sample scores/targets plus the scalar
report and, when the dataset carries an ``attack`` column, an ASVspoof-style
per-spoofing-system EER breakdown. It relies on the eval loader being
un-shuffled and non-dropping so scores line up with ``dataset.manifest``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.audio_dataset import ManifestAudioDataset
from src.evaluation.metrics import binary_classification_report, per_attack_eer

__all__ = ["SpoofEvalResult", "evaluate_spoof_model"]


@dataclass
class SpoofEvalResult:
    y_true: np.ndarray
    y_pred: np.ndarray
    y_score: np.ndarray                       # P(bonafide)
    report: dict[str, Any]
    per_attack: dict[str, dict[str, float]] = field(default_factory=dict)
    n_samples: int = 0

    def summary_line(self) -> str:
        r = self.report
        return (f"n={self.n_samples}  acc={r['accuracy']:.4f}  f1={r['f1']:.4f}  "
                f"auc={r.get('roc_auc', float('nan')):.4f}  "
                f"eer={r.get('eer', float('nan')) * 100:.2f}%")


@torch.no_grad()
def evaluate_spoof_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str | torch.device,
    *,
    attacks: list[str] | None = None,
) -> SpoofEvalResult:
    """Forward every batch, collect scores, build the report.

    ``attacks`` (per-sample system ids aligned with the loader order) enables the
    per-attack EER breakdown. If omitted, it is taken from the loader's
    :class:`ManifestAudioDataset` when available.
    """

    model.eval()
    logits_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    for mel, target in loader:
        logits_chunks.append(model(mel.to(device)).float().cpu())
        target_chunks.append(target.reshape(-1))

    logits = torch.cat(logits_chunks)
    y_true = torch.cat(target_chunks).numpy().astype(int)
    y_score = torch.softmax(logits, dim=1)[:, 1].numpy()
    y_pred = logits.argmax(dim=1).numpy()

    report = binary_classification_report(y_true, y_pred, y_score)

    if attacks is None and isinstance(getattr(loader, "dataset", None), ManifestAudioDataset):
        ds = loader.dataset
        if not loader_preserves_order(loader):
            attacks = None
        elif len(ds) == len(y_true):
            attacks = [row.attack for row in ds.manifest]

    per_attack: dict[str, dict[str, float]] = {}
    if attacks is not None and len(attacks) == len(y_true) and (y_true == 1).any():
        per_attack = per_attack_eer(attacks, y_true, y_score)

    return SpoofEvalResult(
        y_true=y_true, y_pred=y_pred, y_score=y_score,
        report=report, per_attack=per_attack, n_samples=int(y_true.size),
    )


def loader_preserves_order(loader: DataLoader) -> bool:
    """True when the loader yields samples in dataset order (no shuffle, no drop_last)."""

    sampler = getattr(loader, "sampler", None)
    is_sequential = sampler.__class__.__name__ == "SequentialSampler"
    return is_sequential and not getattr(loader, "drop_last", False)
