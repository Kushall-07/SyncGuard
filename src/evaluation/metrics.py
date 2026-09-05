"""Classification metrics for SyncGuard (Phase 2B skeleton).

Phase 6 fills in the full audio-spoof evaluation report; this module already
provides the pieces it needs so the :class:`~src.training.trainer.Trainer` and
later evaluation scripts share one implementation:

* :func:`accuracy`
* :func:`precision_recall_f1`
* :func:`roc_auc`
* :func:`equal_error_rate` - the standard anti-spoofing metric (EER)
* :func:`confusion_matrix`
* :func:`binary_classification_report` - aggregates the above

Inputs may be Python lists, NumPy arrays or 1-D ``torch.Tensor``s. Score inputs
are the probability / logit of the positive class (class ``1``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn import metrics as _skm

__all__ = [
    "accuracy",
    "precision_recall_f1",
    "roc_auc",
    "equal_error_rate",
    "confusion_matrix",
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
