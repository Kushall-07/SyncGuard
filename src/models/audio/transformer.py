"""Transformer encoder over audio temporal tokens (Phase 5).

Sits on top of the Phase 4 CNN front-end: it takes the token sequence
``[B, T', D]`` the CNN emits, adds sinusoidal positional information, and refines
it with a small ``nn.TransformerEncoder`` stack, returning ``[B, T', D]``
unchanged in shape so the same :class:`~src.models.heads.spoof_head.SpoofHead`
(and, later, the cross-attention fusion) consumes it.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from src.config import ModelConfig

__all__ = ["SinusoidalPositionalEncoding", "AudioTransformerEncoder"]


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal position encoding (Vaswani et al., 2017), added to the input."""

    def __init__(self, d_model: int, *, max_len: int = 4096, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(
                f"sequence length {seq_len} exceeds positional-encoding max_len {self.pe.size(1)}"
            )
        return self.dropout(x + self.pe[:, :seq_len])


class AudioTransformerEncoder(nn.Module):
    """``[B, T, D]`` -> ``[B, T, D]`` via positional encoding + Transformer layers."""

    def __init__(self, model_cfg: ModelConfig) -> None:
        super().__init__()
        d_model = model_cfg.audio_embedding_dim
        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model, dropout=model_cfg.audio_tf_dropout
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=model_cfg.num_heads,
            dim_feedforward=model_cfg.audio_tf_ff_dim,
            dropout=model_cfg.audio_tf_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=model_cfg.audio_tf_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,  # no-op with norm_first=True; silences a warning
        )

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(f"expected [B, T, D], got {tuple(tokens.shape)}")
        x = self.pos_encoding(tokens)
        return self.encoder(x, src_key_padding_mask=key_padding_mask)
