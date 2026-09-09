"""Modality-agnostic cross-attention block (Phase 10).

One *directional* cross-attention stack: a query token sequence ``[B, T, D]``
attends into a context token sequence ``[B, S, D]`` and is refined by a
per-position feed-forward block. Pre-norm, query-stream residual connections, a
trailing :class:`~torch.nn.LayerNorm`, and PyTorch's standard
:class:`~torch.nn.MultiheadAttention` (``batch_first=True``) - no custom kernel.

Mirrors the conventions of
:class:`~src.models.common.temporal_transformer.TemporalTransformerEncoder`
(GELU FFN, ``norm_first``-style pre-norm, ``d_model % n_heads == 0`` guard) so
the audio, visual and fusion stacks all look the same.

This block does **not** classify, pool, or produce a score.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["CrossAttentionBlock"]


class _CrossAttnLayer(nn.Module):
    """Pre-norm cross-attention + FFN, both with a query-stream residual."""

    def __init__(self, dim: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.drop_attn = nn.Dropout(dropout)

        self.ln_ff = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
        )
        self.drop_ff = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,                 # [B, T, D]
        context: Tensor,               # [B, S, D]
        *,
        key_padding_mask: Tensor | None = None,   # [B, S] bool; True -> ignore that key
    ) -> Tensor:
        kv = self.ln_kv(context)
        attn_out, _ = self.attn(
            self.ln_q(query), kv, kv,
            key_padding_mask=key_padding_mask, need_weights=False,
        )
        query = query + self.drop_attn(attn_out)
        query = query + self.drop_ff(self.ffn(self.ln_ff(query)))
        return query


class CrossAttentionBlock(nn.Module):
    """``(query [B, T, D], context [B, S, D]) -> [B, T, D]``.

    ``n_layers`` stacked :class:`_CrossAttnLayer`s (each re-attends the *same*
    ``context``), then a trailing ``LayerNorm``. Output length always equals the
    query length ``T``.
    """

    def __init__(
        self,
        *,
        dim: int = 256,
        num_heads: int = 4,
        n_layers: int = 1,
        ff_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        self.dim = dim
        self.num_heads = num_heads
        self.layers = nn.ModuleList(
            _CrossAttnLayer(dim, num_heads, ff_dim, dropout) for _ in range(n_layers)
        )
        self.ln_out = nn.LayerNorm(dim)

    def forward(
        self,
        query: Tensor,
        context: Tensor,
        *,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        if query.dim() != 3 or context.dim() != 3:
            raise ValueError(
                f"expected [B, T, D] query and [B, S, D] context, "
                f"got {tuple(query.shape)} and {tuple(context.shape)}"
            )
        if query.shape[-1] != self.dim or context.shape[-1] != self.dim:
            raise ValueError(f"query/context last dim must be {self.dim}")

        x = query
        for layer in self.layers:
            x = layer(x, context, key_padding_mask=key_padding_mask)
        return self.ln_out(x)
