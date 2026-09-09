"""Landmark embedding for the visual branch (Phase 7).

Maps one frame's selected face/mouth landmarks to a ``visual_embedding_dim``
token. A frame is ``[N, C]`` (``N`` points from
:data:`src.preprocessing.landmarks.REGIONS`, ``C`` = 2 or 3 coords); the whole
clip is ``[B, T, N, C]`` -> ``[B, T, D]``. This is the visual analogue of the
audio CNN front-end's per-frame projection, and it feeds both the temporal
Transformer (:class:`~src.models.video.visual_encoder.VisualEncoder`) and the
non-temporal baseline (:class:`~src.models.video.landmark_baseline.LandmarkMLPBaseline`).
"""

from __future__ import annotations

import torch
from torch import nn

from src.config import ModelConfig
from src.preprocessing.landmarks import region_point_count

__all__ = ["LandmarkEmbedding"]


class LandmarkEmbedding(nn.Module):
    def __init__(self, model_cfg: ModelConfig) -> None:
        super().__init__()
        self.n_points = region_point_count(model_cfg.visual_regions)
        self.coords = model_cfg.landmark_coords
        self.in_dim = self.n_points * self.coords
        self.output_dim = model_cfg.visual_embedding_dim

        self.net = nn.Sequential(
            nn.Linear(self.in_dim, model_cfg.visual_embed_hidden),
            nn.GELU(),
            nn.Dropout(model_cfg.dropout),
            nn.Linear(model_cfg.visual_embed_hidden, self.output_dim),
        )

    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        """``[B, T, N, C]`` (or ``[B, T, N*C]``) -> ``[B, T, D]``."""

        if landmarks.dim() == 4:
            b, t, n, c = landmarks.shape
            if n * c != self.in_dim:
                raise ValueError(
                    f"expected N*C = {self.in_dim} (N={self.n_points}, C={self.coords}), "
                    f"got N={n}, C={c}"
                )
            landmarks = landmarks.reshape(b, t, n * c)
        elif landmarks.dim() != 3 or landmarks.shape[-1] != self.in_dim:
            raise ValueError(
                f"expected [B, T, {self.n_points}, {self.coords}] or [B, T, {self.in_dim}], "
                f"got {tuple(landmarks.shape)}"
            )
        return self.net(landmarks)
