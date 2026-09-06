"""Classification metrics for SyncGuard.

Shared by :class:`~src.training.trainer.Trainer` and the Phase 6 evaluation
scripts so there is one implementation of every number the project reports:

* :func:`accuracy`
* :func:`precision_recall_f1`
* :func:`roc_auc`
* :func:`equal_error_rate` - the standard anti-spoofing metric (EER)
* :func:`confusion_matrix`
* :func:`roc_points` / :func:`det_points` - curve data for plotting
* :func:`per_attack_eer` - ASVspoof per-spoofing-system EER breakdown
* :func:`evaluate_logits` - report straight from model logits
* :func:`binary_classification_report` - aggregates the scalar metrics

Inputs may be Python lists, NumPy arrays or 1-D ``torch.Tensor``s. Score inputs
are the probability / score of the positive (bonafide, label ``1``) class.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn import metrics as _skm

__all__ = [
    "accuracy",
    "precision_recall_f1",
    "roc_auc",
    "equal_error_rate",
    "confusion_matrix",
    "roc_points",
    "det_points",
    "per_attack_eer",
    "evaluate_logits",
    "binary_classification_report",
]


def _to_1d_numpy(values: Any) -> np.ndarray:
    if hasattr(values, "detach"):  # torch.Tensor
        values = values.detach().cpu().numpy()
    arr = np.asarray(values)
    return arr.reshape(-1)


def accuracy(y_true: Any, y_pred: Any) -> float:
    y_true, y_pred = _to_1d_numpy(y_true), _to_1d_numpy(y_pred)
    if y_true.size == 0:
        return float("nan")
    return float((y_true == y_pred).mean())


def precision_recall_f1(
    y_true: Any,
    y_pred: Any,
    *,
    average: str = "binary",
    pos_label: int = 1,
    zero_division: int = 0,
) -> dict[str, float]:
    y_true, y_pred = _to_1d_numpy(y_true), _to_1d_numpy(y_pred)
    precision, recall, f1, _ = _skm.precision_recall_fscore_support(
        y_true, y_pred, average=average, pos_label=pos_label, zero_division=zero_division,
    )
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def roc_auc(y_true: Any, y_score: Any) -> float:
    """Area under the ROC curve. Returns NaN if only one class is present."""

    y_true, y_score = _to_1d_numpy(y_true), _to_1d_numpy(y_score)
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(_skm.roc_auc_score(y_true, y_score))


def equal_error_rate(y_true: Any, y_score: Any) -> tuple[float, float]:
    """Equal Error Rate and the score threshold at which it occurs.

    EER is the operating point where the false-acceptance rate equals the
    false-rejection rate. Convention here: label ``1`` is the positive
    (bonafide) class, so FAR is computed on the negative (spoof) class.
    Returns ``(eer, threshold)``; ``(nan, nan)`` if only one class is present.
    """

    y_true, y_score = _to_1d_numpy(y_true), _to_1d_numpy(y_score)
    if np.unique(y_true).size < 2:
        return float("nan"), float("nan")

    fpr, tpr, thresholds = _skm.roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, float(thresholds[idx])


def confusion_matrix(y_true: Any, y_pred: Any, *, labels: list[int] | None = None) -> np.ndarray:
    y_true, y_pred = _to_1d_numpy(y_true), _to_1d_numpy(y_pred)
    return _skm.confusion_matrix(y_true, y_pred, labels=labels)


def roc_points(y_true: Any, y_score: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(fpr, tpr, thresholds)`` from :func:`sklearn.metrics.roc_curve`."""

    y_true, y_score = _to_1d_numpy(y_true), _to_1d_numpy(y_score)
    return _skm.roc_curve(y_true, y_score)


def det_points(y_true: Any, y_score: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(fpr, fnr, thresholds)`` for a Detection Error Tradeoff (DET) curve.

    ``fnr = 1 - tpr``; the EER is where ``fpr`` and ``fnr`` cross.
    """

    fpr, tpr, thresholds = roc_points(y_true, y_score)
    return fpr, 1.0 - tpr, thresholds


def per_attack_eer(
    attacks: Sequence[str],
    y_true: Any,
    y_score: Any,
    *,
    bonafide_tag: str = "-",
) -> dict[str, dict[str, float]]:
    """EER of each spoofing system vs the shared bonafide pool (ASVspoof convention).

    ``attacks`` is the per-sample system id (``"-"`` for bonafide, ``"A01"``... for
    spoof), aligned with ``y_true`` / ``y_score``. Each spoof system is scored
    against *all* bonafide trials, matching how ASVspoof reports pooled-vs-attack
    EER. Returns ``{attack_id: {"n_spoof": int, "eer": float, "threshold": float}}``.
    """

    attacks = np.asarray(list(attacks), dtype=object)
    y_true = _to_1d_numpy(y_true).astype(int)
    y_score = _to_1d_numpy(y_score)

    bonafide_mask = y_true == 1
    bona_scores = y_score[bonafide_mask]

    out: dict[str, dict[str, float]] = {}
    for attack in sorted(set(attacks[~bonafide_mask])):
        if attack == bonafide_tag:
            continue
        spoof_scores = y_score[(~bonafide_mask) & (attacks == attack)]
        sub_true = np.concatenate([np.ones_like(bona_scores), np.zeros_like(spoof_scores)])
        sub_score = np.concatenate([bona_scores, spoof_scores])
        eer, threshold = equal_error_rate(sub_true, sub_score)
        out[str(attack)] = {
            "n_spoof": int(spoof_scores.size),
            "eer": eer,
            "threshold": threshold,
        }
    return out


def evaluate_logits(logits: Any, y_true: Any) -> dict[str, Any]:
    """Full report from raw 2-class logits: softmax -> P(bonafide), argmax -> pred."""

    if hasattr(logits, "detach"):
        logits = logits.detach().cpu().numpy()
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(f"expected [N, 2] logits, got shape {logits.shape}")
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    y_score = probs[:, 1]
    y_pred = logits.argmax(axis=1)
    return binary_classification_report(y_true, y_pred, y_score)


def binary_classification_report(
    y_true: Any,
    y_pred: Any,
    y_score: Any | None = None,
) -> dict[str, Any]:
    """One dict with accuracy, precision/recall/F1, and (given scores) ROC-AUC + EER."""

    report: dict[str, Any] = {"accuracy": accuracy(y_true, y_pred)}
    report.update(precision_recall_f1(y_true, y_pred))
    report["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    if y_score is not None:
        report["roc_auc"] = roc_auc(y_true, y_score)
        eer, threshold = equal_error_rate(y_true, y_score)
        report["eer"] = eer
        report["eer_threshold"] = threshold
    return report
