"""Evaluation figures for SyncGuard (Phase 6).

Matplotlib (``Agg`` backend) helpers that turn a set of per-sample scores into
the plots the project reports: confusion matrix, ROC curve, DET curve with the
EER point marked, score-distribution histograms, and training curves read back
from a run's ``metrics.csv``. Every function writes a PNG and returns its path;
:func:`save_spoof_evaluation_figures` writes the whole set.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.metrics import (
    det_points,
    equal_error_rate,
    roc_auc,
    roc_points,
)

__all__ = [
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_det_curve",
    "plot_score_distributions",
    "plot_training_curves",
    "save_spoof_evaluation_figures",
]

_CLASS_NAMES = ("spoof", "bonafide")


def _finish(fig: "plt.Figure", path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion_matrix(
    cm: Any,
    path: str | Path,
    *,
    class_names: Sequence[str] = _CLASS_NAMES,
    normalize: bool = False,
    title: str = "Confusion matrix",
) -> Path:
    cm = np.asarray(cm, dtype=float)
    display = cm / cm.sum(axis=1, keepdims=True).clip(min=1) if normalize else cm

    fig, ax = plt.subplots(figsize=(4.2, 3.8), constrained_layout=True)
    im = ax.imshow(display, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set(
        xticks=range(len(class_names)), yticks=range(len(class_names)),
        xticklabels=class_names, yticklabels=class_names,
        xlabel="predicted", ylabel="true", title=title,
    )
    fmt = ".2f" if normalize else ".0f"
    thresh = display.max() / 2
    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            ax.text(j, i, format(display[i, j], fmt), ha="center", va="center",
                    color="white" if display[i, j] > thresh else "black")
    return _finish(fig, path)


def plot_roc_curve(y_true: Any, y_score: Any, path: str | Path, *, title: str = "ROC curve") -> Path:
    fpr, tpr, _ = roc_points(y_true, y_score)
    auc = roc_auc(y_true, y_score)

    fig, ax = plt.subplots(figsize=(4.4, 4.0), constrained_layout=True)
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=0.8)
    ax.set(xlabel="false positive rate", ylabel="true positive rate", title=title,
           xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="lower right")
    return _finish(fig, path)


def plot_det_curve(y_true: Any, y_score: Any, path: str | Path, *, title: str = "DET curve") -> Path:
    fpr, fnr, _ = det_points(y_true, y_score)
    eer, _ = equal_error_rate(y_true, y_score)

    fig, ax = plt.subplots(figsize=(4.4, 4.0), constrained_layout=True)
    ax.plot(fpr * 100, fnr * 100, label="DET")
    if np.isfinite(eer):
        ax.scatter([eer * 100], [eer * 100], color="red", zorder=3,
                   label=f"EER = {eer * 100:.2f}%")
    ax.set(xlabel="false acceptance rate (%)", ylabel="false rejection rate (%)", title=title)
    ax.legend(loc="upper right")
    return _finish(fig, path)


def plot_score_distributions(
    y_true: Any,
    y_score: Any,
    path: str | Path,
    *,
    bins: int = 40,
    title: str = "Score distribution: P(bonafide)",
) -> Path:
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_score = np.asarray(y_score).reshape(-1)
    eer, threshold = equal_error_rate(y_true, y_score)

    fig, ax = plt.subplots(figsize=(5.0, 3.6), constrained_layout=True)
    ax.hist(y_score[y_true == 0], bins=bins, alpha=0.6, label="spoof", color="#d1495b")
    ax.hist(y_score[y_true == 1], bins=bins, alpha=0.6, label="bonafide", color="#00798c")
    if np.isfinite(threshold):
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1,
                   label=f"EER thr = {threshold:.3f} (EER {eer * 100:.2f}%)")
    ax.set(xlabel="P(bonafide)", ylabel="count", title=title)
    ax.legend()
    return _finish(fig, path)


def plot_training_curves(metrics_csv: str | Path, path: str | Path) -> Path:
    """Plot loss and val EER/AUC over epochs from a run's ``metrics.csv``."""

    rows: list[dict[str, str]] = []
    with Path(metrics_csv).open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{metrics_csv}: no rows")

    def col(name: str) -> list[float]:
        return [float(r[name]) for r in rows if r.get(name) not in (None, "")]

    epochs = col("epoch")
    fig, (ax_loss, ax_meta) = plt.subplots(1, 2, figsize=(9.0, 3.6), constrained_layout=True)

    ax_loss.plot(epochs, col("train_loss"), label="train")
    if any(r.get("val_loss") for r in rows):
        ax_loss.plot(epochs, col("val_loss"), label="val")
    ax_loss.set(xlabel="epoch", ylabel="loss", title="Loss")
    ax_loss.legend()

    plotted = False
    for name, axis_label in (("val_eer", "EER"), ("val_roc_auc", "ROC-AUC"), ("val_accuracy", "acc")):
        if any(r.get(name) for r in rows):
            ax_meta.plot(epochs, col(name), label=axis_label)
            plotted = True
    ax_meta.set(xlabel="epoch", title="Validation metrics")
    if plotted:
        ax_meta.legend()
    return _finish(fig, path)


def save_spoof_evaluation_figures(
    out_dir: str | Path,
    *,
    y_true: Any,
    y_score: Any,
    confusion: Any,
    prefix: str = "",
) -> dict[str, Path]:
    """Write the confusion-matrix, ROC, DET and score-distribution PNGs."""

    out_dir = Path(out_dir)
    p = f"{prefix}_" if prefix else ""
    return {
        "confusion_matrix": plot_confusion_matrix(confusion, out_dir / f"{p}confusion_matrix.png"),
        "roc_curve": plot_roc_curve(y_true, y_score, out_dir / f"{p}roc_curve.png"),
        "det_curve": plot_det_curve(y_true, y_score, out_dir / f"{p}det_curve.png"),
        "score_distributions": plot_score_distributions(
            y_true, y_score, out_dir / f"{p}score_distributions.png"
        ),
    }
