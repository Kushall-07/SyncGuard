"""Video deepfake classifier: shared visual encoder + deepfake head (Phase 7).

``model.visual_encoder`` selects the variant with no change to the training code:

* ``"transformer"``  - :class:`~src.models.video.visual_encoder.VisualEncoder`
  (landmark embedding + positional encoding + temporal Transformer) followed by
  the reused :class:`~src.models.heads.spoof_head.SpoofHead` (attentive temporal
  pooling -> 2 logits).
* ``"mlp_baseline"`` -
  :class:`~src.models.video.landmark_baseline.LandmarkMLPBaseline` (order-free
  temporal pooling + MLP), the non-temporal reference.

Logits are ordered ``[fake, real]`` (index 1 = real), matching the label
convention in :mod:`src.data.celebdf_dataset` and :mod:`src.evaluation.metrics`.
"""

from __future__ import annotations

import torch
from torch import nn

from src.config import ModelConfig
from src.models.heads.spoof_head import SpoofHead
from src.models.video.landmark_baseline import LandmarkMLPBaseline
from src.models.video.visual_encoder import VisualEncoder

__all__ = ["DeepfakeClassifier"]


class DeepfakeClassifier(nn.Module):
    def __init__(self, model_cfg: ModelConfig, *, n_classes: int = 2) -> None:
        super().__init__()
        self.variant = model_cfg.visual_encoder
        if self.variant == "transformer":
            self.encoder = VisualEncoder(model_cfg)
            self.head = SpoofHead.from_config(model_cfg, self.encoder.output_dim, n_classes=n_classes)
        elif self.variant == "mlp_baseline":
            self.baseline = LandmarkMLPBaseline(model_cfg, n_classes=n_classes)
        else:  # pragma: no cover - ModelConfig already validates this
            raise ValueError(f"unknown visual_encoder {self.variant!r}")

    def encode(self, landmarks: torch.Tensor) -> torch.Tensor:
        """``[B, T, N, C]`` -> temporal visual tokens ``[B, T, D]`` (transformer only)."""

        if self.variant != "transformer":
            raise RuntimeError("encode() is only available for the 'transformer' variant")
        return self.encoder(landmarks).tokens

    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        """``[B, T, N, C]`` -> class logits ``[B, n_classes]`` (index 1 = real)."""

        if self.variant == "transformer":
            return self.head(self.encode(landmarks))
        return self.baseline(landmarks)
