"""Spoof classification head (Phase 4).

Pools a sequence of audio tokens ``[B, T, D]`` to a single vector and maps it to
two logits ``[spoof, bonafide]`` (index 1 = bonafide, matching the label
convention in :mod:`src.data.manifests` and :mod:`src.evaluation.metrics`).

Pooling options:

* ``mean``     - temporal mean
* ``meanmax``  - concat of temporal mean and max (2*D)
* ``attentive``- softmax-weighted sum with a learned per-frame score
"""

from __future__ import annotations

import torch
from torch import nn

from src.config import ModelConfig

__all__ = ["SpoofHead", "temporal_pool"]


def temporal_pool(tokens: torch.Tensor, mode: str) -> torch.Tensor:
    if tokens.dim() != 3:
        raise ValueError(f"expected [B, T, D], got {tuple(tokens.shape)}")
    if mode == "mean":
        return tokens.mean(dim=1)
    if mode == "meanmax":
        return torch.cat([tokens.mean(dim=1), tokens.amax(dim=1)], dim=-1)
    raise ValueError(f"temporal_pool does not handle mode {mode!r} (attentive is stateful)")


class SpoofHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        *,
        hidden: int = 128,
        n_classes: int = 2,
        dropout: float = 0.1,
        pooling: str = "attentive",
    ) -> None:
        super().__init__()
        self.pooling = pooling

        if pooling == "attentive":
            self.attn = nn.Linear(in_dim, 1)
            pooled_dim = in_dim
        elif pooling == "mean":
            pooled_dim = in_dim
        elif pooling == "meanmax":
            pooled_dim = 2 * in_dim
        else:
            raise ValueError(f"unknown pooling {pooling!r}")

        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    @classmethod
    def from_config(cls, model_cfg: ModelConfig, in_dim: int, *, n_classes: int = 2) -> "SpoofHead":
        return cls(
            in_dim,
            hidden=model_cfg.spoof_head_hidden,
            n_classes=n_classes,
            dropout=model_cfg.dropout,
            pooling=model_cfg.spoof_head_pooling,
        )

    def pool(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.pooling == "attentive":
            weights = torch.softmax(self.attn(tokens), dim=1)   # [B, T, 1]
            return (weights * tokens).sum(dim=1)                # [B, D]
        return temporal_pool(tokens, self.pooling)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(tokens))
