"""Binary sync classification loss (Phase 11).

Implements binary cross-entropy loss for per-window sync prediction with support
for optional valid/window masks to handle padding and variable-length sequences.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

__all__ = ["SyncLossConfig", "SyncLoss"]


@dataclass(frozen=True)
class SyncLossConfig:
    """Configuration for sync loss."""

    pos_weight: float | None = None
    reduction: str = "mean"

    def __post_init__(self) -> None:
        if self.pos_weight is not None and self.pos_weight <= 0:
            raise ValueError("pos_weight must be positive")
        if self.reduction not in ("none", "mean", "sum"):
            raise ValueError(f"reduction must be one of ('none', 'mean', 'sum'), got {self.reduction!r}")


class SyncLoss(nn.Module):
    """Binary cross-entropy loss for per-window sync prediction.

    Supports:
    - Optional valid/window mask to handle padding
    - Positive/negative per-window targets
    - NaN avoidance for fully masked samples
    - Independent of LAV-DF-specific labels
    """

    def __init__(
        self,
        *,
        pos_weight: float | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"reduction must be one of ('none', 'mean', 'sum'), got {reduction!r}")

        self.pos_weight = pos_weight
        self.reduction = reduction

        # Convert pos_weight to tensor for BCEWithLogitsLoss
        if pos_weight is not None:
            self.register_buffer(
                "pos_weight_tensor", torch.tensor(pos_weight, dtype=torch.float32)
            )
        else:
            self.pos_weight_tensor = None

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Compute binary cross-entropy loss.

        Args:
            logits: Per-window sync logits [B, T]
            targets: Per-window sync targets [B, T] (0 or 1)
            mask: Optional mask [B, T] where True indicates valid positions

        Returns:
            Loss scalar (or per-sample if reduction="none")
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

        # Apply mask: set invalid positions to 0 in both logits and targets
        # This ensures they don't contribute to the loss
        masked_logits = logits.masked_fill(~mask, 0.0)
        masked_targets = targets.masked_fill(~mask, 0.0)

        # Compute BCE loss
        if self.pos_weight_tensor is not None:
            loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=self.pos_weight_tensor, reduction="none"
            )
        else:
            loss_fn = nn.BCEWithLogitsLoss(reduction="none")

        loss = loss_fn(masked_logits, masked_targets)  # [B, T]

        # Apply mask to loss (invalid positions get 0 loss)
        loss = loss.masked_fill(~mask, 0.0)

        # Handle reduction
        if self.reduction == "none":
            # Sum over time dimension, keep batch dimension
            loss = loss.sum(dim=1)  # [B]
        elif self.reduction == "sum":
            loss = loss.sum()
        else:  # "mean"
            # Mean over valid positions only
            valid_count = mask.sum().clamp(min=1).float()
            loss = loss.sum() / valid_count

        return loss

    @classmethod
    def from_config(cls, cfg: SyncLossConfig) -> "SyncLoss":
        return cls(pos_weight=cfg.pos_weight, reduction=cfg.reduction)
