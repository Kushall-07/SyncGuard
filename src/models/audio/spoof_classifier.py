"""Audio-only spoof classifier: shared audio encoder + spoof head.

Phase 4 introduced this as CNN-encoder + head. Phase 5 swaps the encoder for
:class:`~src.models.audio.encoder.AudioEncoder`, so ``model.audio_encoder`` in the
config selects the CNN baseline (``"cnn"``) or the CNN + Transformer variant
(``"cnn_transformer"``) with no change to this class or the training code.
"""

from __future__ import annotations

import torch
from torch import nn

from src.config import ModelConfig
from src.models.audio.encoder import AudioEncoder
from src.models.heads.spoof_head import SpoofHead

__all__ = ["SpoofClassifier"]


class SpoofClassifier(nn.Module):
    def __init__(self, model_cfg: ModelConfig, n_mels: int, *, n_classes: int = 2) -> None:
        super().__init__()
        self.encoder = AudioEncoder(model_cfg, n_mels)
        self.head = SpoofHead.from_config(model_cfg, self.encoder.output_dim, n_classes=n_classes)

    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        """log-mel ``[B, n_mels, T]`` -> temporal tokens ``[B, T', D]``."""

        return self.encoder(mel).tokens

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """log-mel ``[B, n_mels, T]`` -> class logits ``[B, n_classes]`` (index 1 = bonafide)."""

        return self.head(self.encode(mel))
