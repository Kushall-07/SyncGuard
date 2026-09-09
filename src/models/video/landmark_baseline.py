"""Non-temporal landmark baseline for the visual branch (Phase 7).

`LandmarkMLPBaseline` embeds each frame, then pools over time with
**order-free** statistics (mean and standard deviation) before a small MLP head.
Because the pooling discards frame order, any gap between this baseline and the
temporal Transformer (:class:`~src.models.video.visual_encoder.VisualEncoder` +
head) measures what temporal modelling contributes - the visual analogue of the
audio branch's CNN-vs-Transformer comparison.
"""

from __future__ import annotations

import torch
from torch import nn

from src.config import ModelConfig
from src.models.video.landmark_embedding import LandmarkEmbedding

__all__ = ["LandmarkMLPBaseline"]


class LandmarkMLPBaseline(nn.Module):
    def __init__(self, model_cfg: ModelConfig, *, n_classes: int = 2) -> None:
        super().__init__()
        self.embed = LandmarkEmbedding(model_cfg)
        d = self.embed.output_dim
        self.classifier = nn.Sequential(
            nn.Linear(2 * d, model_cfg.spoof_head_hidden),
            nn.GELU(),
            nn.Dropout(model_cfg.dropout),
            nn.Linear(model_cfg.spoof_head_hidden, n_classes),
        )

    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        """``[B, T, N, C]`` -> class logits ``[B, n_classes]`` (index 1 = real)."""

        tokens = self.embed(landmarks)                 # [B, T, D]
        mean = tokens.mean(dim=1)
        std = tokens.std(dim=1, unbiased=False)
        return self.classifier(torch.cat([mean, std], dim=-1))
