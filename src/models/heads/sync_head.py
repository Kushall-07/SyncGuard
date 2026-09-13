"""Audio-Visual Sync Head (Phase 11).

Lightweight temporal prediction head that consumes the fused AV representation
from Phase 10 cross-attention and produces per-window synchronization logits.

The head operates on the per-timestep fused representation [B, T, D] and outputs
sync logits [B, T] (or [B, T, 1]) for each video token position. No global pooling
is applied to the primary output, enabling frame-level synchronization assessment.

An aggregation method is provided to compute video-level sync scores from the
per-window logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

__all__ = ["SyncHeadConfig", "SyncHead"]


@dataclass(frozen=True)
class SyncHeadConfig:
    """Configuration for the sync head."""

    hidden_dim: int = 128
    dropout: float = 0.1
    aggregation: Literal["mean", "max", "median"] = "mean"

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not 0 <= self.dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")
        if self.aggregation not in ("mean", "max", "median"):
            raise ValueError(f"aggregation must be one of ('mean', 'max', 'median'), got {self.aggregation!r}")


class SyncHead(nn.Module):
    """Lightweight temporal prediction head for audio-visual synchronization.

    Consumes fused AV representation [B, T, D] from Phase 10 cross-attention
    and produces per-window sync logits [B, T].

    Architecture:
    - Linear projection: [B, T, D] -> [B, T, hidden_dim]
    - Layer normalization
    - ReLU activation
    - Dropout
    - Linear projection: [B, T, hidden_dim] -> [B, T, 1]
    - Squeeze to [B, T]

    The head is designed to be lightweight and fast, with no global pooling
    on the primary output to preserve temporal resolution.
    """

    def __init__(
        self,
        input_dim: int = 256,
        *,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        aggregation: Literal["mean", "max", "median"] = "mean",
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation

        self.proj1 = nn.Linear(input_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj2 = nn.Linear(hidden_dim, 1)

    def forward(self, fused: Tensor) -> Tensor:
        """Forward pass.

        Args:
            fused: Fused AV representation [B, T, D]

        Returns:
            Per-window sync logits [B, T]
        """
        if fused.dim() != 3:
            raise ValueError(f"expected [B, T, D] input, got {tuple(fused.shape)}")
        if fused.shape[-1] != self.input_dim:
            raise ValueError(
                f"input last dim must be {self.input_dim}, got {fused.shape[-1]}"
            )

        # [B, T, D] -> [B, T, hidden_dim]
        x = self.proj1(fused)
        x = self.ln(x)
        x = torch.relu(x)
        x = self.dropout(x)

        # [B, T, hidden_dim] -> [B, T, 1]
        logits = self.proj2(x)

        # [B, T, 1] -> [B, T]
        return logits.squeeze(-1)

    def aggregate(self, logits: Tensor, mask: Tensor | None = None) -> Tensor:
        """Aggregate per-window logits to video-level score.

        Args:
            logits: Per-window sync logits [B, T]
            mask: Optional mask [B, T] where True indicates valid positions

        Returns:
            Video-level sync score [B]
        """
        if logits.dim() != 2:
            raise ValueError(f"expected [B, T] logits, got {tuple(logits.shape)}")

        if mask is None:
            mask = torch.ones_like(logits, dtype=torch.bool)

        if mask.shape != logits.shape:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} must match logits shape {tuple(logits.shape)}"
            )

        # Apply mask: set invalid positions to -inf so they don't affect aggregation
        masked_logits = logits.masked_fill(~mask, float("-inf"))

        if self.aggregation == "mean":
            # Mean over valid positions
            score = masked_logits.sum(dim=1) / mask.sum(dim=1).clamp(min=1).float()
        elif self.aggregation == "max":
            # Max over valid positions
            score = masked_logits.max(dim=1).values
        elif self.aggregation == "median":
            # Median over valid positions
            # For each sample, gather valid values and compute median
            scores = []
            for i in range(logits.shape[0]):
                valid = masked_logits[i][mask[i]]
                if len(valid) > 0:
                    scores.append(valid.median())
                else:
                    scores.append(torch.tensor(0.0, device=logits.device))
            score = torch.stack(scores)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return score

    @classmethod
    def from_config(cls, cfg: SyncHeadConfig, input_dim: int = 256) -> "SyncHead":
        return cls(
            input_dim=input_dim,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
            aggregation=cfg.aggregation,
        )
