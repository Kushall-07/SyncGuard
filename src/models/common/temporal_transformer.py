"""Temporal Transformer encoder over a token sequence ``[B, T, D]`` (Phase 7).

Lifted out of the Phase 5 audio Transformer so the audio branch and the Phase 7
visual branch share one implementation. It adds sinusoidal positional information
to a token sequence and refines it with a small ``nn.TransformerEncoder`` stack,
returning ``[B, T, D]`` unchanged in shape so the same
:class:`~src.models.heads.spoof_head.SpoofHead` (and, later, cross-attention
fusion) can consume it.

:class:`~src.models.audio.transformer.AudioTransformerEncoder` and
:class:`~src.models.video.visual_encoder.VisualEncoder` are thin config adapters
over this class; keeping the submodule attribute names (``pos_encoding``,
``encoder``) stable means existing Phase 5 checkpoints still load.
"""

from __future__ import annotations

import math

import torch
from torch import nn

__all__ = ["SinusoidalPositionalEncoding", "TemporalTransformerEncoder"]


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


class TemporalTransformerEncoder(nn.Module):
    """``[B, T, D]`` -> ``[B, T, D]`` via positional encoding + Transformer layers."""

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        ff_dim: int,
        dropout: float,
        num_layers: int,
        max_len: int = 4096,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
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
