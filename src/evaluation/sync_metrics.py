"""Metrics for audio-visual synchronization evaluation (Phase 11).

Implements per-window and video-level sync metrics, clearly distinguished from
deepfake/manipulation metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

__all__ = ["SyncMetrics", "compute_sync_metrics"]


@dataclass
class SyncMetrics:
    """Container for sync evaluation metrics."""

    # Per-window metrics
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float | None

    # Video-level metrics
    video_accuracy: float
    video_auc: float | None

    # Counts
    n_samples: int
    n_windows: int
    n_positive_windows: int
    n_negative_windows: int


def compute_sync_metrics(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor | None = None,
    aggregation: Literal["mean", "max", "median"] = "mean",
) -> SyncMetrics:
    """Compute sync metrics from logits and targets.

    Args:
        logits: Per-window sync logits [B, T]
        targets: Per-window sync targets [B, T] (0 or 1)
        mask: Optional mask [B, T] where True indicates valid positions
        aggregation: Method for aggregating to video-level scores

    Returns:
        SyncMetrics object with all computed metrics
    """
    if logits.shape != targets.shape:
        raise ValueError(
            f"logits shape {tuple(logits.shape)} must match targets shape {tuple(targets.shape)}"
        )

    if mask is None:
        mask = torch.ones_like(logits, dtype=torch.bool)

    if mask.shape != logits.shape:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} must match logits shape {tuple(logits.shape)}"
        )

    # Flatten and mask
    flat_logits = logits[mask]
    flat_targets = targets[mask]

    if flat_logits.numel() == 0:
        # No valid windows
        return SyncMetrics(
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            auc=None,
            video_accuracy=0.0,
            video_auc=None,
            n_samples=logits.shape[0],
            n_windows=0,
            n_positive_windows=0,
            n_negative_windows=0,
        )

    # Compute probabilities
    probs = torch.sigmoid(flat_logits)
    preds = (probs >= 0.5).float()

    # Count classes
    n_positive = (flat_targets == 1).sum().item()
    n_negative = (flat_targets == 0).sum().item()

    # Per-window metrics
    correct = (preds == flat_targets).sum().item()
    accuracy = correct / flat_logits.numel()

    # Precision, recall, F1
    true_positives = ((preds == 1) & (flat_targets == 1)).sum().item()
    false_positives = ((preds == 1) & (flat_targets == 0)).sum().item()
    false_negatives = ((preds == 0) & (flat_targets == 1)).sum().item()

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # AUC (only if both classes present)
    if n_positive > 0 and n_negative > 0:
        try:
            from sklearn.metrics import roc_auc_score

            auc = roc_auc_score(flat_targets.cpu().numpy(), flat_logits.cpu().numpy())
        except ImportError:
            auc = None
        except Exception:
            auc = None
    else:
        auc = None

    # Video-level metrics
    video_scores = _aggregate_to_video_level(logits, mask, aggregation)
    video_targets = _aggregate_to_video_level(targets, mask, aggregation)

    # Video-level predictions
    video_probs = torch.sigmoid(video_scores)
    video_preds = (video_probs >= 0.5).float()

    video_correct = (video_preds == video_targets).sum().item()
    video_accuracy = video_correct / video_scores.numel()

    # Video-level AUC
    video_n_positive = (video_targets == 1).sum().item()
    video_n_negative = (video_targets == 0).sum().item()

    if video_n_positive > 0 and video_n_negative > 0:
        try:
            from sklearn.metrics import roc_auc_score

            video_auc = roc_auc_score(video_targets.cpu().numpy(), video_scores.cpu().numpy())
        except ImportError:
            video_auc = None
        except Exception:
            video_auc = None
    else:
        video_auc = None

    return SyncMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        auc=auc,
        video_accuracy=video_accuracy,
        video_auc=video_auc,
        n_samples=logits.shape[0],
        n_windows=flat_logits.numel(),
        n_positive_windows=n_positive,
        n_negative_windows=n_negative,
    )


def _aggregate_to_video_level(
    values: Tensor,
    mask: Tensor,
    aggregation: Literal["mean", "max", "median"],
) -> Tensor:
    """Aggregate per-window values to video-level.

    Args:
        values: Per-window values [B, T]
        mask: Valid positions mask [B, T]
        aggregation: Aggregation method

    Returns:
        Video-level values [B]
    """
    # Ensure 2D tensors
    if values.dim() == 1:
        values = values.unsqueeze(0)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)

    # Apply mask: set invalid positions to 0
    masked_values = values.masked_fill(~mask, 0.0)

    if aggregation == "mean":
        # Mean over valid positions
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        video_values = masked_values.sum(dim=1) / counts.squeeze(-1)
    elif aggregation == "max":
        # Max over valid positions (set invalid to -inf)
        masked_values_inf = values.masked_fill(~mask, float("-inf"))
        video_values = masked_values_inf.max(dim=1).values
    elif aggregation == "median":
        # Median over valid positions
        video_values = []
        for i in range(values.shape[0]):
            valid = masked_values[i][mask[i]]
            if len(valid) > 0:
                video_values.append(valid.median())
            else:
                video_values.append(torch.tensor(0.0, device=values.device))
        video_values = torch.stack(video_values)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    return video_values
